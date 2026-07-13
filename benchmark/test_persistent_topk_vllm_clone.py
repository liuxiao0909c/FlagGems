"""Benchmark vLLM clone persistent topk vs vLLM.

Speedup = latency_vllm / latency_clone.
> 1.0 → clone faster; < 1.0 → vLLM faster.
"""

import pytest
import torch

import flag_gems
from flag_gems.fused.persistent_topk_vllm_clone import persistent_topk_full

from . import base

device = flag_gems.device

HAS_VLLM = False
try:
    import vllm._custom_ops  # noqa: F401

    HAS_VLLM = True
except (ImportError, AttributeError):
    pass

STRIDE = 262144
K = 512


def _baseline_persistent_topk(logits, lengths, indices, workspace, max_seq_len, seq_lens):
    """vLLM CUDA kernel — baseline."""
    torch.ops._C.persistent_topk(logits, lengths, indices, workspace, K, max_seq_len)
    return indices


def _clone_persistent_topk(logits, lengths, indices, workspace, max_seq_len, seq_lens):
    """vLLM clone in Triton (16 CTA/group, large path)."""
    persistent_topk_full(logits, lengths, indices, workspace, k=K, max_seq_len=max_seq_len)
    #flag_gems.persistent_topk(logits, lengths, indices, workspace, k=K, max_seq_len=max_seq_len)
    return indices


class PersistentTopKCloneBenchmark(base.Benchmark):
    DEFAULT_SHAPE_DESC = "num_rows, seq_len, max_seq_len"

    def set_shapes(self, shape_file_path=None):
        #self.shapes = [(n, 262144, 1048576) for n in range(1, 33)]
        self.shapes = [(n, 262144, 1048576) for n in range(1, 3)]

    def get_input_iter(self, dtype):
        for num_rows, seq_len, max_seq_len in self.shapes:
            logits = torch.full(
                (num_rows, STRIDE), float("-inf"), dtype=torch.float32, device=self.device
            )
            logits[:, :seq_len] = torch.randn(num_rows, seq_len, device=self.device)

            lengths = torch.full((num_rows,), seq_len, dtype=torch.int32, device=self.device)
            indices = torch.empty((num_rows, K), dtype=torch.int32, device=self.device)
            workspace = torch.empty(1024 * 1024, dtype=torch.uint8, device=self.device)
            seq_lens = torch.full((num_rows,), seq_len, dtype=torch.int32, device=self.device)

            yield logits, lengths, indices, workspace, max_seq_len, seq_lens


@pytest.mark.skipif(not HAS_VLLM, reason="vLLM not installed")
@pytest.mark.persistent_topk
def test_persistent_topk_clone():
    bench = PersistentTopKCloneBenchmark(
        op_name="persistent_topk_clone",
        torch_op=_baseline_persistent_topk,
        gems_op=_clone_persistent_topk,
        dtypes=[torch.float32],
    )
    bench.run()
