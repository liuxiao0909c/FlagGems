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
def persistent_topk_kernel(
    logits_ptr,
    output_ptr,
    lengths_ptr,
    num_rows,
    stride,
    topk: tl.constexpr,
    max_seq_len,
    chunk_size,
    ctas_per_group,
    g_histogram_ptr,
    g_state_ptr,
    VEC_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):

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

    ctas_per_group = (stride + max_chunk_elements - 1) // max_chunk_elements
    chunk_size = (stride + ctas_per_group - 1) // ctas_per_group
    chunk_size = ((chunk_size + vec_size - 1) // vec_size) * vec_size
    chunk_size = min(max_chunk_elements, chunk_size)

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
        g_histogram,
        g_state,
        VEC_SIZE=vec_size,
        BLOCK_SIZE=THREADS_PER_BLOCK,
        num_warps=THREADS_PER_BLOCK // 32,
    )
