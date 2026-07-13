import logging

import torch
import triton
import triton.language as tl

from flag_gems.utils.triton_version_utils import has_triton_tle

if has_triton_tle(3, 6, 0):
    try:
        import triton.experimental.tle.language as tle

        HAS_TLE = True
    except ImportError:
        tle = None
        HAS_TLE = False
else:
    tle = None
    HAS_TLE = False

THREADS_PER_BLOCK = 1024
RADIX = 256
MEDIUN_HIST_BYTES = 2 * (RADIX + 128) * 4  # 3072
MEDIUM_SCALARS_BYTES = 5 * 4  # 20
MEDIUM_HEADER_SIZE = (MEDIUN_HIST_BYTES + MEDIUM_SCALARS_BYTES + 127) &(~127)  # 3200
MAX_BUFFERED_ITEMS = 4096
SMEM_MEDIUM = MEDIUM_HEADER_SIZE + 2 * MAX_BUFFERED_ITEMS * 4  # 35968
RADIX_THRESHOLD = 32768
DECODE_BINS = 2048
HIST2048_THRESHOLD = 8192
FIXED_SMEM_LARGE = ((RADIX + RADIX + 5) * 4 + 15) & (~15)  # 2080

logger = logging.getLogger(__name__)


@triton.jit
def _convert_to_uint32_v2(x):
    bits = x.to(tl.uint32, bitcast=True)
    return tl.where((bits & 0x80000000) != 0, ~bits, (bits | 0x80000000))


@triton.jit
def _barrier_with_atomic_add(
    arrival_counter_ptr,
    zeros,
    lane,
    thresold,
):
    tl.atomic_add(
        arrival_counter_ptr + zeros,
        1,
        mask=lane == 0,
        sem="relaxed",
        scope="gpu",
    )
    # TODO: every thread query, no following debug_barrier needed
    arrival_counter = tl.atomic_add(
        arrival_counter_ptr,
        0,
        sem="relaxed",
        scope="gpu",
    )
    while arrival_counter < thresold:
        arrival_counter = tl.atomic_add(
            arrival_counter_ptr,
            0,
            sem="relaxed",
            scope="gpu",
        )


@triton.jit
def _radix_topk(
    row_input,
    row_output,
    seq_len,
    my_chunk_start,
    CHUNK_SIZE: tl.constexpr,
    local_histogram_ptr,
    suffix_sum_ptr,
    shared_scalars_ptr,
    shared_ordered_ptr,
    g_histogram_ptr,
    g_state_ptr,
    cta_in_group,
    ctas_per_group,
    barrier_phase,
    iter_idx,
    TOPK: tl.constexpr,
    VEC_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    RADIX: tl.constexpr = 256

    my_chunk_end = my_chunk_start + CHUNK_SIZE
    my_chunk_end = min(my_chunk_end, seq_len)
    actual_chunk_size = my_chunk_end - my_chunk_start if my_chunk_start < seq_len else 0
    lane = tl.arange(0, BLOCK_SIZE)
    ones = tl.full([BLOCK_SIZE], 1, tl.uint32)
    zeros = tl.zeros([BLOCK_SIZE], dtype=tl.uint32)
    zeros_2d = tl.zeros([BLOCK_SIZE, VEC_SIZE], dtype=tl.uint32)
    vec = tl.arange(0, VEC_SIZE)

    # -- Stage 1: Load chunk to shared memory as ordered uint32 --
    # TODO: remove rem_tiles, rem_elems
    n_vec_full = actual_chunk_size // (BLOCK_SIZE * VEC_SIZE)
    rem_tiles = (actual_chunk_size - n_vec_full * BLOCK_SIZE * VEC_SIZE) // BLOCK_SIZE
    rem_elems = actual_chunk_size % BLOCK_SIZE
    for t in tl.range(0, n_vec_full):
        base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
        offs = base[:, None] + vec[None, :]
        x = tl.load(row_input + offs)
        bits = _convert_to_uint32_v2(x)
        tl.store(shared_ordered_ptr + offs, bits)
    for t in tl.range(0, rem_tiles):
        offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
        x = tl.load(row_input + offs)
        bits = _convert_to_uint32_v2(x)
        tl.store(shared_ordered_ptr + offs, bits)
    if rem_elems > 0:
        offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
        in_range = lane < rem_elems
        x = tl.load(row_input + offs, mask=in_range, other=float("-inf"))
        bits = _convert_to_uint32_v2(x)
        tl.store(shared_ordered_ptr + offs, bits, mask=in_range)
    tl.debug_barrier()

    # -- Init radix select state --
    tl.store(shared_scalars_ptr + zeros, 0, mask=lane == 0)  # prefix
    tl.store(shared_scalars_ptr + 1 + zeros, TOPK, mask=lane == 0)  # remaining_k
    tl.debug_barrier()

    # -- Initial barrier --
    _barrier_with_atomic_add(
        g_state_ptr + 2,
        zeros,
        lane,
        (barrier_phase + 1) * ctas_per_group,
    )
    barrier_phase += 1

    if cta_in_group == 0:
        tl.store(g_state_ptr + 3 + zeros, 0, mask=lane == 0)  # output_counter

    # -- Stage 2: 4 rounds of radix select --
    for round_idx in tl.static_range(0, 4):
        global_round = iter_idx * 4 + round_idx
        shift_bits = 24 - round_idx * 8
        prefix = tl.load(shared_scalars_ptr)
        remaining_k = tl.load(shared_scalars_ptr + 1)

        current_hist_ptr = g_histogram_ptr + (global_round % 3) * RADIX
        next_hist_ptr = g_histogram_ptr + ((global_round + 1) % 3) * RADIX

        tl.store(local_histogram_ptr + lane, 0, mask=lane < RADIX)
        tl.debug_barrier()

        # TODO: no vec load from smem
        for t in tl.range(0, n_vec_full):
            base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
            offs = base[:, None] + vec[None, :]
            ordered = tl.load(shared_ordered_ptr + offs)
            mask = 0 if round_idx == 0 else ((~0) << (32 - round_idx * 8))
            match = (ordered & mask) == prefix
            bucket = (ordered >> shift_bits) & 0xFF
            tl.atomic_add(
                local_histogram_ptr + bucket,
                1,
                mask=match,
                sem="relaxed",
                scope="cta",
            )
        for t in tl.range(0, rem_tiles):
            offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
            ordered = tl.load(shared_ordered_ptr + offs)
            mask = 0 if round_idx == 0 else ((~0) << (32 - round_idx * 8))
            match = (ordered & mask) == prefix
            bucket = (ordered >> shift_bits) & 0xFF
            tl.atomic_add(
                local_histogram_ptr + bucket,
                1,
                mask=match,
                sem="relaxed",
                scope="cta",
            )
        if rem_elems > 0:
            offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
            in_range = lane < rem_elems
            ordered = tl.load(shared_ordered_ptr + offs, mask=in_range, other=0) 
            mask = 0 if round_idx == 0 else ((~0) << (32 - round_idx * 8))
            match = (ordered & mask) == prefix
            bucket = (ordered >> shift_bits) & 0xFF
            tl.atomic_add(
                local_histogram_ptr + bucket,
                1,
                mask=match & in_range,
                sem="relaxed",
                scope="cta",
            )
        tl.debug_barrier()

        counts = tl.load(local_histogram_ptr + lane, mask=lane < RADIX)
        tl.atomic_add(
            current_hist_ptr + zeros,
            counts,
            mask=(counts > 0) & (lane < RADIX),
            sem="relaxed",
            scope="gpu",
        )

        if cta_in_group == 0:
            tl.store(next_hist_ptr + lane, 0, mask=lane < RADIX)

        _barrier_with_atomic_add(
            g_state_ptr + 2,
            zeros,
            lane,
            (barrier_phase + 1) * ctas_per_group,
        )
        barrier_phase += 1

        g_counts = tl.load(current_hist_ptr + lane, mask=lane < RADIX)
        tl.store(suffix_sum_ptr + lane, g_counts, mask=lane < RADIX)
        tl.debug_barrier()

        # for (uint32_t stride = 1; stride < RADIX; stride *= 2) {
        for t in tl.static_range(0, 8):  # RADIX=(1 << 8)
            val = tl.load(suffix_sum_ptr + lane, mask=lane < RADIX)
            other_offs = lane + (1 << t)
            tmp = tl.load(suffix_sum_ptr + other_offs, mask=other_offs < RADIX, other=0)
            val += tmp
            tl.debug_barrier()
            tl.store(suffix_sum_ptr + lane, val, mask=lane < RADIX)
            tl.debug_barrier()

        tl.store(shared_scalars_ptr + 2 + zeros, 0, mask=lane == 0)  # threshold_bin
        tl.store(shared_scalars_ptr + 3 + zeros, remaining_k, mask=lane == 0)  # next_remaining_k
        tl.debug_barrier()

        count_ge = tl.load(suffix_sum_ptr + lane, mask=lane < RADIX)
        count_gt = tl.load(suffix_sum_ptr + lane + 1, mask=(lane + 1) < RADIX, other=0)
        threshold_mask = (count_ge >= remaining_k) & (count_gt < remaining_k)
        tl.store(shared_scalars_ptr + 2 + zeros, lane, mask=threshold_mask)
        tl.store(shared_scalars_ptr + 3 + zeros, remaining_k - count_gt, mask=threshold_mask)
        tl.debug_barrier()

        threshold_bin = tl.load(shared_scalars_ptr + 2 + zeros, mask=lane == 0, other=0) 
        new_prefix = prefix | (threshold_bin << shift_bits)
        tl.store(shared_scalars_ptr + zeros, threshold_bin, mask=lane == 0)
        next_remaining_k = tl.load(shared_scalars_ptr + 3 + zeros, mask=lane == 0, other=0)
        tl.store(shared_scalars_ptr + 1 + zeros, next_remaining_k, mask=lane == 0)
        tl.debug_barrier()
    # end 4 radix rounds

    # -- Count local > pivot elements --
    ordered_pivot = tl.load(shared_scalars_ptr)
    # no usage of suffix_sum[0]
    #tl.store(suffix_sum_ptr + zeros, 0, mask=lane == 0)
    #tl.debug_barrier()

    my_gt_count = tl.full((BLOCK_SIZE,), 0, dtype=tl.uint32)
    # TODO: no vec load from smem
    for t in tl.range(0, n_vec_full):
        base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
        offs = base[:, None] + vec[None, :]
        ordered = tl.load(shared_ordered_ptr + offs)
        gt_mask = ordered > ordered_pivot
        my_gt_count += tl.sum(gt_mask.to(tl.uint32), axis=-1)
    for t in tl.range(0, rem_tiles):
        offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
        ordered = tl.load(shared_ordered_ptr + offs)
        gt_mask = ordered > ordered_pivot
        my_gt_count += gt_mask.to(tl.uint32)
    if rem_elems > 0:
        offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
        in_range = lane < rem_elems
        ordered = tl.load(shared_ordered_ptr + offs, mask=in_range, other=0)
        gt_mask = (ordered > ordered_pivot) & in_range
        my_gt_count += gt_mask.to(tl.uint32)
    tl.debug_barrier()
    local_gt_count = tl.sum(my_gt_count)

    # -- Stage 3: Collect top-k indices --
    tl.store(local_histogram_ptr + zeros, 0, mask=lane == 0)
    # no usage of local_histogram[1]
    gt_pos = tl.atomic_add(
        g_state_ptr + 3 + zeros,
        local_gt_count,
        mask=(lane == 0) & (local_gt_count > 0),
        sem="relaxed",
        scope="gpu",
    )
    tl.debug_barrier()

    # TODO: no vec load from smem
    for t in tl.range(0, n_vec_full):
        base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
        offs = base[:, None] + vec[None, :]
        ordered = tl.load(shared_ordered_ptr + offs)
        gt_mask = ordered > ordered_pivot
        local_pos = tl.atomic_add(
            local_histogram_ptr + zeros_2d,
            1,
            mask=gt_mask,
            sem="relaxed",
            scope="cta",
        )
        tl.store(row_output + gt_pos[:, None] + local_pos, my_chunk_start + offs, mask=gt_mask) 
    for t in tl.range(0, rem_tiles):
        offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
        ordered = tl.load(shared_ordered_ptr + offs)
        gt_mask = ordered > ordered_pivot
        local_pos = tl.atomic_add(
            local_histogram_ptr + zeros,
            1,
            mask=gt_mask,
            sem="relaxed",
            scope="cta",
        )
        tl.store(row_output + gt_pos + local_pos, my_chunk_start + offs, mask=gt_mask)
    if rem_elems > 0:
        offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
        in_range = lane < rem_elems
        ordered = tl.load(shared_ordered_ptr + offs, mask=in_range, other=0)
        gt_mask = (ordered > ordered_pivot) & in_range
        local_pos = tl.atomic_add(
            local_histogram_ptr + zeros,
            1,
            mask=gt_mask,
            sem="relaxed",
            scope="cta",
        )
        tl.store(row_output + gt_pos + local_pos, my_chunk_start + offs, mask=gt_mask)

    _barrier_with_atomic_add(
        g_state_ptr + 2,
        zeros,
        lane,
        (barrier_phase + 1) * ctas_per_group,
    )
    barrier_phase += 1

    # TODO: no vec load from smem
    for t in tl.range(0, n_vec_full):
        base = t * BLOCK_SIZE * VEC_SIZE + lane * VEC_SIZE
        offs = base[:, None] + vec[None, :]
        ordered = tl.load(shared_ordered_ptr + offs)
        eq_mask = ordered == ordered_pivot
        eq_pos = tl.atomic_add(
            g_state_ptr + 3 + zeros_2d,
            1,
            mask=eq_mask,
            sem="relaxed",
            scope="gpu",
        )
        tl.store(row_output + eq_pos, my_chunk_start + offs, mask=eq_mask & (eq_pos < TOPK))
    for t in tl.range(0, rem_tiles):
        offs = (n_vec_full * VEC_SIZE + t) * BLOCK_SIZE + lane
        ordered = tl.load(shared_ordered_ptr + offs)
        eq_mask = ordered == ordered_pivot
        eq_pos = tl.atomic_add(
            g_state_ptr + 3 + zeros,
            1,
            mask=eq_mask,
            sem="relaxed",
            scope="gpu",
        )
        tl.store(row_output + eq_pos, my_chunk_start + offs, mask=eq_mask & (eq_pos < TOPK))
    if rem_elems > 0:
        offs = (n_vec_full * VEC_SIZE + rem_tiles) * BLOCK_SIZE + lane
        in_range = lane < rem_elems
        ordered = tl.load(shared_ordered_ptr + offs, mask=in_range, other=0)
        eq_mask = (ordered == ordered_pivot) & in_range
        eq_pos = tl.atomic_add(
            g_state_ptr + 3 + zeros,
            1,
            mask=eq_mask,
            sem="relaxed",
            scope="gpu",
        )
        tl.store(row_output + eq_pos, my_chunk_start + offs, mask=eq_mask & (eq_pos < TOPK))

    return barrier_phase


@triton.jit
def persistent_topk_kernel(
    logits_ptr,
    output_ptr,
    lengths_ptr,
    num_rows,
    stride,
    TOPK: tl.constexpr,
    max_seq_len,
    CHUNK_SIZE: tl.constexpr,
    ctas_per_group,
    num_groups,
    g_histogram_ptr,
    g_state_ptr,
    VEC_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    RADIX_THRESHOLD: tl.constexpr = 32768
    RADIX: tl.constexpr = 256
    HIST2048_THRESHOLD: tl.constexpr = 8192

    pid = tl.program_id(0)
    group_id = pid // ctas_per_group
    cta_in_group = pid % ctas_per_group
    if pid >= num_groups * ctas_per_group:
        return  # TODO: remove
    if cta_in_group != 0 and max_seq_len <= RADIX_THRESHOLD:
        return
    local_histogram = tle.gpu.alloc(
        [RADIX],
        dtype=tl.uint32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    suffix_sum = tle.gpu.alloc(
        [RADIX],
        dtype=tl.uint32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    # TODO:why add 5 in FIXED_SMEM_LARGE
    shared_scalars = tle.gpu.alloc(
        [4],
        dtype=tl.uint32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    shared_ordered = tle.gpu.alloc(
        [CHUNK_SIZE],
        dtype=tl.uint32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    local_histogram_ptr = tle.gpu.local_ptr(local_histogram, (0,))
    suffix_sum_ptr = tle.gpu.local_ptr(suffix_sum, (0,))
    shared_scalars_ptr = tle.gpu.local_ptr(shared_scalars, (0,))
    shared_ordered_ptr = tle.gpu.local_ptr(shared_ordered, (0,))

    g_histogram_ptr += group_id * 3 * RADIX
    g_state_ptr += group_id * 4
    barrier_phase = tl.zeros((), dtype=tl.uint32)
    total_iters = tl.cdiv(num_rows, num_groups)
    for i in tl.range(total_iters):
        row_idx = group_id + i * num_groups
        if row_idx >= num_rows:
            pass
        seq_len = tl.load(lengths_ptr + row_idx)
        row_output = output_ptr + row_idx * TOPK
        row_in = tl.multiple_of(logits_ptr + row_idx * stride, VEC_SIZE * 4)
        if seq_len <= RADIX_THRESHOLD:
            if cta_in_group == 0:
                if seq_len <= TOPK:
                    num_tiles: tl.constexpr = (TOPK + BLOCK_SIZE - 1) // BLOCK_SIZE
                    lane = tl.arange(0, BLOCK_SIZE)
                    for tile_idx in tl.static_range(0, num_tiles):
                        pos = tile_idx * BLOCK_SIZE + lane
                        take_row = pos < seq_len
                        tl.store(
                            row_output + pos,
                            pos.to(tl.int32),
                            mask=take_row,
                        )
                        take_pad = (pos >= seq_len) & (pos < TOPK)
                        tl.store(row_output + pos, -1, mask=take_pad)
                elif seq_len <= HIST2048_THRESHOLD:
                    pass  # TODO: histogram_2048_topk
                else:
                    pass  # TODO: histogram_256_topk
            pass
        else:
            my_chunk_start = cta_in_group * CHUNK_SIZE
            barrier_phase = _radix_topk(
                row_in,
                row_output,
                seq_len,
                my_chunk_start,
                CHUNK_SIZE,
                local_histogram_ptr,
                suffix_sum_ptr,
                shared_scalars_ptr,
                shared_ordered_ptr,
                g_histogram_ptr,
                g_state_ptr,
                cta_in_group,
                ctas_per_group,
                barrier_phase,
                i,
                TOPK,
                VEC_SIZE,
                BLOCK_SIZE,
            )
    return


def persistent_topk(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    k: int,
    max_seq_len: int,
) -> None:
    """vLLM-compatible persistent topk decode.

    Args:
        logits:  [num_rows, stride] float32.
        lengths: [num_rows] int32, or [B, next_n] int32 for MTP.
        output:  [num_rows, k] int32 — pre-allocated output buffer.
        workspace: uint8 buffer (required). Used for internal
                   scratch if provided. Enables CUDAGraph compatibility
                   by avoiding internal torch.zeros allocation.
        k:       number of top elements to select. Must be 512/1024/2048.
        max_seq_len: global max seq_len across all rows.
    """
    assert logits.is_cuda, "persistent_topk: logits must be CUDA tensor"
    assert lengths.is_cuda, "persistent_topk: lengths must be CUDA tensor"
    assert output.is_cuda, "persistent_topk: output must be CUDA tensor"
    assert logits.dtype == torch.float32, "persistent_topk: only float32 supported"
    assert lengths.dtype == torch.int32, "persistent_topk: lengths must be int32"
    assert output.dtype == torch.int32, "persistent_topk: output must be int32"
    assert logits.dim() == 2, "persistent_topk: logits must be 2D"
    assert lengths.dim() in (1, 2), "persistent_topk: lengths must be 1D or 2D"
    assert lengths.is_contiguous(), "persistent_topk: lengths must be contiguous"
    assert output.dim() == 2, "persistent_topk: output must be 2D"

    num_rows = logits.size(0)
    stride = logits.stride(0)
    assert lengths.numel() == num_rows, f"persistent_topk: lengths size mismatch: {lengths.numel()} vs {num_rows}"
    assert output.size(0) == num_rows and output.size(1) == k, (
        f"persistent_topk: output size mismatch: ({output.size(0)}, {output.size(1)}) vs ({num_rows}, {k})"
    )
    assert k in (512, 1024, 2048), (
        f"persistent_topk supports k=512, k=1024, or k=2048, got k={k}"
    )
    max_seq_len = min(logits.shape[1], max_seq_len)
    device = logits.device
    device_props = torch.cuda.get_device_properties(device.index)
    num_sms = device_props.multi_processor_count
    max_smem_per_block = device_props.shared_memory_per_block_optin
    if num_rows <= 4:
        effective_max_smem = min(max_smem_per_block, SMEM_MEDIUM)
    elif num_rows <= 8:
        effective_max_smem = min(max_smem_per_block, 48 *1024)
    else:
        effective_max_smem = max_smem_per_block
    available_for_ordered = effective_max_smem - FIXED_SMEM_LARGE
    max_chunk_elements = available_for_ordered // 4  # sizeof(uint32)
    vec_size = 1
    if stride % 4 == 0:
        vec_size = 4
    elif stride % 2 == 0:
        vec_size = 2

    max_chunk_elements = (max_chunk_elements // vec_size) * vec_size
    min_chunk = vec_size * THREADS_PER_BLOCK
    max_chunk_elements = max(max_chunk_elements, min_chunk)
    max_chunk_elements = triton.next_power_of_2(max_chunk_elements)

    ctas_per_group = (stride + max_chunk_elements - 1) // max_chunk_elements
    chunk_size = (stride + ctas_per_group - 1) // ctas_per_group
    chunk_size = ((chunk_size + vec_size - 1) // vec_size) * vec_size
    chunk_size = triton.next_power_of_2(chunk_size)
    chunk_size = min(max_chunk_elements, chunk_size)
    assert chunk_size >= available_for_ordered // 4, "fail to get chunk_size"

    smem_size = FIXED_SMEM_LARGE + chunk_size * 4  # sizeof(uint32)
    smem_size = max(SMEM_MEDIUM, smem_size)

    max_threads_per_block = device_props.max_threads_per_block
    occupancy = max(1, max_threads_per_block // THREADS_PER_BLOCK)  # 1
 
    needs_cooperative = max_seq_len > RADIX_THRESHOLD
    hw_resident_cap = num_sms * occupancy
    max_resident_ctas = hw_resident_cap
    if needs_cooperative:
        headroom = num_sms if occupancy > 1 else 1
        if max_resident_ctas >= headroom + ctas_per_group:
            max_resident_ctas -= headroom
    num_groups = min(max_resident_ctas // ctas_per_group, num_rows)
    num_groups = max(1, num_groups)
    total_ctas = num_groups * ctas_per_group

    if needs_cooperative and total_ctas > hw_resident_cap:
        assert 0, "too many chunk"
    # RadixRowState layout:
    #     uint32_t histogram[3][256];
    #     uint32_t remaining_k;
    #     uint32_t prefix;
    #     int arrival_counter;
    #     int output_counter;
    histogram_bytes = RADIX * 3 * 4
    radix_row_state_bytes = histogram_bytes + 4 * 4
    workspace[:(num_groups * radix_row_state_bytes)] = 0
    g_histogram_size = num_groups * histogram_bytes
    g_state_size = num_groups * 4 * 4
    g_histogram = workspace[:g_histogram_size].view(torch.uint32).view(num_groups, 3, RADIX)
    g_state = workspace[g_histogram_size:g_histogram_size + g_state_size].view(torch.int32).view(num_groups, 4)

    persistent_topk_kernel[(total_ctas,)](
        logits,
        output,
        lengths,
        num_rows,
        stride,
        k,
        max_seq_len,
        chunk_size,
        ctas_per_group,
        num_groups,
        g_histogram,
        g_state,
        VEC_SIZE=vec_size,
        BLOCK_SIZE=THREADS_PER_BLOCK,
        num_warps=THREADS_PER_BLOCK // 32,
    )
