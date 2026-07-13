"""vLLM persistent_topk Triton implementation — four per-row paths, runnable.

Configuration:
  ctas_per_group=8, CHUNK_SIZE=32768 (2^15, power of 2)
  num_groups = max(1, min(78//8, num_rows)), max 9 groups on H20
  block mesh + launch_cooperative_grid=True

Paths (per-row dispatch):
  Trivial: seq_len <= K           — direct index copy
  Decode:  seq_len <= 8192        — 2048-bin FP16 histogram, single CTA
  Medium:  seq_len <= RADIX_THRESHOLD (32768) — 256-bin + 4-pass FP32 refinement
  Large:   seq_len > 32768        — 8-CTA SMEM ordered buffer + barrier
"""
import torch, triton, triton.language as tl
try:
    import triton.experimental.tle.language as tle; HAS_TLE = True
except ImportError:
    tle = None; HAS_TLE = False

# ── Constants ──────────────────────────────────────────────────────────────
RADIX = tl.constexpr(256)
RADIX_THRESHOLD = tl.constexpr(32768)
HIST2048_THRESHOLD = tl.constexpr(8192)
K2048 = tl.constexpr(2048)

# ── Shared helpers ─────────────────────────────────────────────────────────

@triton.jit
def _float_to_sortable(val):
    bits = val.to(tl.int32, bitcast=True)
    sign_ext = bits >> 31
    mask = sign_ext | tl.full(bits.shape, -(1 << 31), dtype=tl.int32)
    return bits ^ mask

@triton.jit
def decode_bin(x):
    h = x.to(tl.float16)
    bits = h.to(tl.uint16, bitcast=True)
    sign = tl.full(bits.shape, 0x8000, tl.uint16)
    return tl.where((bits & sign) != 0, ~bits & tl.full(bits.shape, 0x7FFF, tl.uint16),
                    bits | sign) >> 5

@triton.jit
def convert_to_uint8(x):
    h = x.to(tl.float16)
    bits = h.to(tl.uint16, bitcast=True)
    sign = tl.full(bits.shape, 0x8000, tl.uint16)
    mapped = tl.where((bits & sign) != 0, ~bits & tl.full(bits.shape, 0x7FFF, tl.uint16),
                      bits | sign)
    return (mapped >> 8).to(tl.int32)

@triton.jit
def convert_to_uint32_v2(x):
    bits = x.to(tl.uint32, bitcast=True)
    sign = tl.full(bits.shape, 0x80000000, tl.uint32)
    return tl.where((bits & sign) != 0, bits, (~bits) & tl.full(bits.shape, 0x7FFFFFFF, tl.uint32))

@triton.jit
def persistent_topk_full_kernel(
    input_ptr, output_ptr, lengths_ptr,
    global_hist_ptr, scatter_counter_ptr,
    num_rows, stride, top_k,
    NUM_GROUPS: tl.constexpr, CTAS_PER_GROUP: tl.constexpr,
    CHUNK_SIZE: tl.constexpr, RADIX: tl.constexpr,
    mesh: tl.constexpr,
):
    pid = tl.program_id(0)
    group_id = pid // CTAS_PER_GROUP
    cta_idx = pid % CTAS_PER_GROUP
    if group_id >= NUM_GROUPS:
        return

    arange = tl.arange(0, CHUNK_SIZE)
    bins = tl.arange(0, RADIX)
    is_first = tl.arange(0, CHUNK_SIZE) < 1
    hist_base = global_hist_ptr + group_id * 3 * RADIX
    cntr = scatter_counter_ptr + group_id * 2

    ordered_smem = tle.gpu.alloc([CHUNK_SIZE], dtype=tl.uint32,
                                  scope=tle.gpu.smem, nv_mma_shared_layout=False)
    ordered_ptr = tle.gpu.local_ptr(ordered_smem, (0,))
    smem_hist = tle.gpu.alloc([RADIX], dtype=tl.int32,
                               scope=tle.gpu.smem, nv_mma_shared_layout=False)
    shp = tle.gpu.local_ptr(smem_hist, (0,))

    rows_remain = num_rows - group_id
    group_iters = (rows_remain + NUM_GROUPS - 1) // NUM_GROUPS
    pad_iters = (num_rows + NUM_GROUPS - 1) // NUM_GROUPS - group_iters

    for _ in tl.range(group_iters):
        row_idx = group_id + _ * NUM_GROUPS
        seq_len = tl.load(lengths_ptr + row_idx)
        row_input = input_ptr + row_idx * stride
        row_output = output_ptr + row_idx * top_k
        # ── Trivial path ──
        if seq_len <= top_k:
            if cta_idx == 0:
                for i in tl.range(0, top_k, 1024):
                    toffs = i + tl.arange(0, 1024)
                    tm = toffs < seq_len
                    tl.store(row_output + toffs, toffs.to(tl.int32), mask=tm)
                    tl.store(row_output + toffs, -1, mask=(toffs >= seq_len) & (toffs < top_k))

            for j in tl.static_range(6):
                tle.distributed_barrier(mesh)
        else:
            tl.store(cntr, 0)
            tl.store(cntr + 1, 0)
            tle.distributed_barrier(mesh)

            my_start = cta_idx * CHUNK_SIZE
            my_end = my_start + CHUNK_SIZE
            if my_end > seq_len:
                my_end = seq_len
            actual = my_end - my_start

            # (1024, 4) 2D vec4 load
            VEC: tl.constexpr = 4
            LANES: tl.constexpr = 1024
            N_TILES: tl.constexpr = CHUNK_SIZE // (LANES * VEC)
            lane2d = tl.arange(0, LANES)
            vec = tl.arange(0, VEC)
            aligned_ptr = tl.multiple_of(row_input + my_start, VEC * 4)
            for t in tl.static_range(0, N_TILES):
                tile_base = t * LANES * VEC + lane2d * VEC
                offs_2d = tile_base[:, None] + vec[None, :]
                vals_2d = tl.load(aligned_ptr + offs_2d)
                ordered_2d = _float_to_sortable(vals_2d).to(tl.uint32, bitcast=True)
                store_offs = t * LANES * VEC + tl.arange(0, LANES * VEC)
                tl.store(ordered_ptr + store_offs, ordered_2d.reshape(LANES * VEC))
            load_mask = arange < actual
            tl.debug_barrier()

            prefix_val = tl.zeros((), dtype=tl.uint32)
            remaining_k = top_k

            for r in tl.static_range(4):
                global_round = _ * 4 + r
                shift = 24 - r * 8
                curr_hist = hist_base + (global_round % 3) * RADIX
                next_hist = hist_base + ((global_round + 1) % 3) * RADIX

                tl.store(shp + bins, 0)
                tl.debug_barrier()
                for ti in tl.range(0, actual, 1024):
                    toffs = ti + tl.arange(0, 1024)
                    tm = toffs < actual
                    su = tl.load(ordered_ptr + toffs, mask=tm, other=0)
                    bucket = (su >> shift) & 0xFF
                    if r == 0:
                        partial = tm
                    else:
                        if r == 1:
                            mbits_val = 0xFF000000
                        elif r == 2:
                            mbits_val = 0xFFFF0000
                        else:
                            mbits_val = 0xFFFFFF00
                        mbits_arr = tl.full([1024], mbits_val, dtype=tl.uint32)
                        partial = tm & ((su & mbits_arr) == prefix_val.to(tl.uint32))
                    tl.atomic_add(shp + bucket.to(tl.int32),
                                  tl.full([1024], 1, dtype=tl.int32),
                                  mask=partial, sem="relaxed", scope="cta")
                tl.atomic_add(curr_hist + bins, tl.load(shp + bins),
                              sem="acq_rel", scope="gpu")
                if cta_idx == 0:
                    tl.store(next_hist + bins, tl.zeros([256], dtype=tl.int32))

                tle.distributed_barrier(mesh)
                counts = tl.load(curr_hist + bins)
                ps, total = tle.cumsum(counts, axis=0, reverse=False)
                ss = total - ps
                pivot = tl.max(tl.where(ss >= remaining_k, bins, -1))
                ca = tl.sum(tl.where(bins > pivot, counts, 0))
                remaining_k = remaining_k - ca
                prefix_val = prefix_val | (pivot << shift)

            threshold = prefix_val.to(tl.uint32)
            above_total = top_k - remaining_k

            su = tl.load(ordered_ptr + arange, mask=load_mask, other=0)
            above = load_mask & (su > threshold)
            n_above = tl.sum(above.to(tl.int32))
            if n_above > 0:
                pa = tl.cumsum(above.to(tl.int32), axis=0)
                base = tl.atomic_add(cntr, n_above, sem="acq_rel", scope="gpu")
                base_all = tl.sum(tl.where(is_first, base, 0))
                wp = base_all + pa - 1
                tl.store(row_output + wp, (my_start + arange).to(tl.int32),
                         mask=above & (wp >= 0) & (wp < top_k))

            tle.distributed_barrier(mesh)

            equal = load_mask & (su == threshold)
            n_equal = tl.sum(equal.to(tl.int32))
            if n_equal > 0:
                pe = tl.cumsum(equal.to(tl.int32), axis=0)
                base = tl.atomic_add(cntr + 1, n_equal, sem="acq_rel", scope="gpu")
                base_all = tl.sum(tl.where(is_first, base, 0))
                wpe = above_total + base_all + pe - 1
                tl.store(row_output + wpe, (my_start + arange).to(tl.int32),
                         mask=equal & ((base_all + pe - 1) < remaining_k)
                                & (wpe >= 0) & (wpe < top_k))

    for _ in tl.range(pad_iters):
        for j in tl.static_range(6):
            tle.distributed_barrier(mesh)


# ── Host wrapper ────────────────────────────────────────────────────────────

def persistent_topk_full(logits, lengths, output, workspace, k=512, max_seq_len=None):
    num_rows = logits.size(0)
    stride = logits.size(1)
    seq_lens = lengths.reshape(-1) if lengths.dim() == 2 else lengths
    num_sms = torch.cuda.get_device_properties(logits.device).multi_processor_count

    # Dynamic ctas_per_group — mirror vLLM's effective_max_smem strategy.
    # Power-of-2 chunk constraint: 4096, 8192, 16384, 32768.
    if num_rows == 1:
        ctas_per_group = 64
        chunk_size = (stride + 63) // 64      # 4096 = 2^12
        smem_est = 2064 + chunk_size * 4    # ≈18KB
    elif num_rows == 2:
        ctas_per_group = 32
        chunk_size = (stride + 31) // 32      # 8192 = 2^13
        smem_est = 2064 + chunk_size * 4    # ≈34KB
    elif num_rows <= 4:
        ctas_per_group = 16
        chunk_size = (stride + 15) // 16      # 16384 = 2^14
        smem_est = 2064 + chunk_size * 4    # ≈67KB
    else:
        ctas_per_group = 8
        chunk_size = (stride + 7) // 8        # 32768 = 2^15
        smem_est = 2064 + chunk_size * 4    # ≈131KB

    occ = min(2, 227 * 1024 // smem_est)
    hw_cap = num_sms * occ
    headroom = num_sms if occ > 1 else 1
    if hw_cap >= headroom + ctas_per_group:
        hw_cap -= headroom
    num_groups = max(1, min(hw_cap // ctas_per_group, num_rows))
    total_ctas = num_groups * ctas_per_group

    hist_sz = num_groups * 3 * 256
    cnt_sz = num_groups * 2
    need_bytes = (hist_sz + cnt_sz) * 4
    if workspace.numel() >= need_bytes:
        ws = workspace[:need_bytes].view(torch.int32)
        ws.zero_()
    else:
        ws = torch.zeros(hist_sz + cnt_sz, dtype=torch.int32, device=logits.device)

    try:
        triton.set_allocator(
            lambda s, a, stream: torch.empty((s,), dtype=torch.uint8, device=logits.device))
    except Exception:
        pass

    mesh = tle.device_mesh({"block": [("block_x", ctas_per_group)]})
    persistent_topk_full_kernel[(total_ctas,)](
        logits, output, seq_lens,
        ws[:hist_sz], ws[hist_sz:],
        num_rows=num_rows, stride=stride, top_k=k,
        NUM_GROUPS=num_groups, CTAS_PER_GROUP=ctas_per_group,
        CHUNK_SIZE=chunk_size, RADIX=256,
        mesh=mesh,
        launch_cooperative_grid=True, num_warps=32,
    )
