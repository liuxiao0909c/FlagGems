"""Correctness tests for vLLM clone persistent topk — large path + trivial path.

Test shapes: num_rows ∈ {2,4,8,16,32}, stride=262144, k=512.
"""

import pytest
import torch

import flag_gems
from flag_gems.fused.persistent_topk import persistent_topk as persistent_topk_full

device = flag_gems.device

STRIDE = 262144
K = 512

SHAPES = [
    (1, 262144, 1048576),
    (2, 262144, 1048576),
    (4, 262144, 1048576),
    (8, 262144, 1048576),
    (16, 262144, 1048576),
    (32, 262144, 1048576),
]


def _make_inputs(num_rows, seq_len, seed=42):
    torch.manual_seed(seed)
    logits = torch.full(
        (num_rows, STRIDE), float("-inf"), dtype=torch.float32, device=device
    )
    logits[:, :seq_len] = torch.randn(num_rows, seq_len, device=device)
    lengths = torch.full((num_rows,), seq_len, dtype=torch.int32, device=device)
    return logits, lengths


def _selected_values(logits, indices):
    vals = []
    for i in range(indices.shape[0]):
        valid = indices[i][indices[i] >= 0].long()
        if valid.numel() > 0:
            vals.append(logits[i].gather(0, valid).sort(descending=True)[0])
    if vals:
        return torch.cat(vals)
    return torch.empty(0, device=device)


@pytest.mark.parametrize("num_rows, seq_len, max_seq_len", SHAPES)
@torch.inference_mode()
def test_vllm_clone_vs_torch(num_rows, seq_len, max_seq_len):
    logits, lengths = _make_inputs(num_rows, seq_len)
    output = torch.empty((num_rows, K), dtype=torch.int32, device=device)
    ws = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)

    #import pdb; pdb.set_trace()
    persistent_topk_full(logits, lengths, output, ws, k=K, max_seq_len=max_seq_len)
    #import pdb; pdb.set_trace()
    #assert(output.max() == 0)
    torch.accelerator.synchronize()

    ref = torch.empty((num_rows, K), dtype=torch.int32, device=device)
    for i in range(num_rows):
        k = min(K, lengths[i].item())
        ref[i, :k] = logits[i, : lengths[i].item()].topk(k, dim=-1)[1]
        ref[i, k:] = -1

    vals_g = _selected_values(logits, output)
    vals_t = _selected_values(logits, ref)
    torch.testing.assert_close(vals_g, vals_t, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("num_rows, seq_len, max_seq_len", SHAPES)
@torch.inference_mode()
def test_vllm_clone_multi_seed(num_rows, seq_len, max_seq_len):
    for seed in (42, 99, 123, 456, 789):
        logits, lengths = _make_inputs(num_rows, seq_len, seed=seed)
        output = torch.empty((num_rows, K), dtype=torch.int32, device=device)
        ws = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)

        persistent_topk_full(logits, lengths, output, ws, k=K, max_seq_len=max_seq_len)
        torch.accelerator.synchronize()

        ref = torch.empty((num_rows, K), dtype=torch.int32, device=device)
        for i in range(num_rows):
            k = min(K, lengths[i].item())
            ref[i, :k] = logits[i, : lengths[i].item()].topk(k, dim=-1)[1]
            ref[i, k:] = -1

        vals_g = _selected_values(logits, output)
        vals_t = _selected_values(logits, ref)
        torch.testing.assert_close(vals_g, vals_t, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize(
    "num_rows, seq_len",
    [
        (4, 256),    # all trivial
        (8, 262144),  # all large
    ],
)
@torch.inference_mode()
def test_vllm_clone_reuse_workspace(num_rows, seq_len):
    """Verify workspace reuse across calls (CUDAGraph simulation)."""
    logits, lengths = _make_inputs(num_rows, seq_len)
    output = torch.empty((num_rows, K), dtype=torch.int32, device=device)
    ws = torch.empty(1024 * 1024, dtype=torch.uint8, device=device)

    persistent_topk_full(logits, lengths, output, ws, k=K, max_seq_len=seq_len)
    torch.accelerator.synchronize()

    ref = torch.empty((num_rows, K), dtype=torch.int32, device=device)
    for i in range(num_rows):
        k = min(K, lengths[i].item())
        ref[i, :k] = logits[i, : lengths[i].item()].topk(k, dim=-1)[1]
        ref[i, k:] = -1

    vals_g1 = _selected_values(logits, output)
    vals_t = _selected_values(logits, ref)
    torch.testing.assert_close(vals_g1, vals_t, atol=1e-4, rtol=1e-4)

    # Second pass: fresh data, SAME workspace (no zero_())
    logits2, lengths2 = _make_inputs(num_rows, seq_len, seed=99)
    output2 = torch.empty((num_rows, K), dtype=torch.int32, device=device)
    persistent_topk_full(logits2, lengths2, output2, ws, k=K, max_seq_len=seq_len)
    torch.accelerator.synchronize()

    ref2 = torch.empty((num_rows, K), dtype=torch.int32, device=device)
    for i in range(num_rows):
        k = min(K, lengths2[i].item())
        ref2[i, :k] = logits2[i, : lengths2[i].item()].topk(k, dim=-1)[1]
        ref2[i, k:] = -1

    vals_g2 = _selected_values(logits2, output2)
    vals_t2 = _selected_values(logits2, ref2)
    torch.testing.assert_close(vals_g2, vals_t2, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    #test_vllm_clone_vs_torch(5, 262144, 1048576)
    test_vllm_clone_vs_torch(5, 262144, 1048576)

