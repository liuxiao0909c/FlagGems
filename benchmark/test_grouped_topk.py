# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

import flag_gems

from . import base, utils


MAX_IDX = 0xFFFF
SIGN_MASK_INT32 = torch.tensor(0x80000000, dtype=torch.uint32).view(torch.int32)
SIGN_MASK_INT64 = torch.tensor(0x80000000, dtype=torch.int64)


def _pack_val_idx_fp32(val: torch.Tensor, idx: torch.Tensor):
    bits = val.view(torch.int32)
    sign = bits & SIGN_MASK_INT32
    key = torch.where(sign != 0, ~bits, bits).to(torch.int64)
    key = torch.where(sign != 0, key, key | SIGN_MASK_INT64)
    high = key << 16
    low = (0xFFFF & (MAX_IDX - idx)).to(torch.int64)
    return high | low


def _unpack_val_idx_fp32(pair: torch.Tensor):
    key = pair >> 16
    sign = key & SIGN_MASK_INT64
    bits = torch.where(sign != 0, key ^ SIGN_MASK_INT64, key).to(torch.int32)
    bits = torch.where(sign != 0, bits, ~bits)
    val = bits.view(torch.float32)
    idx = (MAX_IDX - (pair & 0xFFFF)).to(torch.int32)
    return val, idx


def torch_grouped_topk(
    scores: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
):
    """Adapted from vLLM: vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py"""
    scores = scores.float()
    if scoring_func == 1:
        scores = scores.sigmoid()

    num_token = scores.size(0)
    original_scores = scores
    scores = scores + bias.unsqueeze(0)
    group_scores = (
        scores.view(num_token, num_expert_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
    )

    use_sorted = True
    # torch.topk is not stable, pack with id before topk
    tmp_group_ids = torch.arange(0, num_expert_group, dtype=torch.int32, device=scores.device)
    tmp_group_ids = tmp_group_ids[None, :].expand(num_token, -1)
    group_pairs = _pack_val_idx_fp32(group_scores, tmp_group_ids)
    top_group_pairs = torch.topk(group_pairs, k=topk_group, dim=-1, sorted=use_sorted)[0]
    _top_group_scores, group_idx = _unpack_val_idx_fp32(top_group_pairs) # [n, top_k_group]
    group_mask = torch.zeros_like(group_scores)  # [n, n_group]
    group_mask.scatter_(1, group_idx, 1)  # [n, n_group]
    score_mask = (
        group_mask.unsqueeze(-1)
        .expand(num_token, num_expert_group, scores.size(-1) // num_expert_group)
        .reshape(num_token, -1)
    )  # [n, e]
    tmp_scores = scores.masked_fill(~score_mask.bool(), float("-inf"))  # [n, e]
    tmp_ids = torch.arange(0, scores.size(1), dtype=torch.int32, device=scores.device)
    tmp_ids = tmp_ids[None, :].expand(num_token, -1)
    pairs = _pack_val_idx_fp32(tmp_scores, tmp_ids)
    top_pairs = torch.topk(pairs, k=topk, dim=-1, sorted=use_sorted)[0]
    if bias is not None:
        _, topk_ids = _unpack_val_idx_fp32(top_pairs)
        topk_weights = original_scores.gather(1, topk_ids)
    else:
        topk_weights, topk_ids = _unpack_val_idx_fp32(top_pairs)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    if routed_scaling_factor != 1.0:
        topk_weights = topk_weights * routed_scaling_factor
    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)


vendor_name = flag_gems.vendor_name

try:
    if vendor_name == "metax":
        from vllm_metax._custom_ops import grouped_topk as ref_grouped_topk
        HAS_VLLM = True
    elif vendor_name == "mthreads" or vendor_name == "hygon" or vendor_name == "ascend":
        # vllm/_custom_ops.py:
        #    The fused grouped_topk kernel is only available on CUDA platforms
        ref_grouped_topk = torch_grouped_topk
        HAS_VLLM = False
    else:
        from vllm._custom_ops import grouped_topk as ref_grouped_topk
        HAS_VLLM = True
except (ImportError, AttributeError):
    ref_grouped_topk = torch_grouped_topk
    HAS_VLLM = False


pytestmark = pytest.mark.skipif(
    HAS_VLLM and (utils.SkipVersion("vllm", "<0.9") or utils.SkipVersion("torch", "<2.7")),
    reason="vLLM or PyTorch version is too low when taking vLLM as reference.",
)


class GroupedTopKBenchmark(base.Benchmark):
    def __init__(
        self,
        op_name,
        torch_op,
        dtypes,
        renormalize=True,
        routed_scaling_factor=1.0,
        scoring_func=0,
    ):
        super().__init__(op_name=op_name, torch_op=torch_op, dtypes=dtypes)
        self.renormalize = renormalize
        self.routed_scaling_factor = routed_scaling_factor
        self.scoring_func = scoring_func

    def set_shapes(self, shape_file_path=None):
        grouped_topk_configs = [
            # Deepseek-3.2
            (num_tokens, num_experts, n_group, topk_group, topk)
            for num_tokens in [1, 8, 32, 64, 128, 256, 496, 512, 16384]
            for num_experts in [256]
            for n_group in [8]
            for topk_group in [4]
            for topk in [8]
        ]
        self.shapes = grouped_topk_configs

    def get_input_iter(self, dtype):
        for config in self.shapes:
            yield from self.grouped_topk_input_fn(config, dtype, self.device)

    def grouped_topk_input_fn(self, config, dtype, device):
        num_tokens, num_experts, n_group, topk_group, topk = config

        scores = torch.randn(num_tokens, num_experts, device=device, dtype=dtype)
        bias = torch.randn(num_experts, device=device, dtype=torch.float32)

        yield (
            scores,
            n_group,
            topk_group,
            topk,
            self.renormalize,
            self.routed_scaling_factor,
            bias,
            self.scoring_func,
        )


@pytest.mark.grouped_topk
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="#2891: Not working")
@pytest.mark.skipif(vendor_name == "iluvatar", reason="#2891: Not working")
@pytest.mark.skipif(flag_gems.vendor_name == "cambricon", reason="#2891: TypeError")
def test_grouped_topk_no_renorm():
    bench = GroupedTopKBenchmark(
        op_name="grouped_topk",
        torch_op=ref_grouped_topk,
        dtypes=[torch.bfloat16],
        renormalize=False,
        scoring_func=0,
    )

    bench.set_gems(flag_gems.grouped_topk)
    bench.run()


@pytest.mark.grouped_topk
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="#2891: Not working ")
@pytest.mark.skipif(vendor_name == "iluvatar", reason="#2891: Not working")
@pytest.mark.skipif(flag_gems.vendor_name == "cambricon", reason="#2891: TypeError")
def test_grouped_topk_score_0():
    bench = GroupedTopKBenchmark(
        op_name="grouped_topk",
        torch_op=ref_grouped_topk,
        dtypes=[torch.bfloat16],
        renormalize=True,
        scoring_func=0,
    )

    bench.set_gems(flag_gems.grouped_topk)
    bench.run()


@pytest.mark.grouped_topk
@pytest.mark.skipif(vendor_name == "kunlunxin", reason="#2891: Not working")
@pytest.mark.skipif(vendor_name == "iluvatar", reason="#2891: Not working")
@pytest.mark.skipif(flag_gems.vendor_name == "cambricon", reason="#2891: TypeError")
def test_grouped_topk_score_1():
    bench = GroupedTopKBenchmark(
        op_name="grouped_topk",
        torch_op=ref_grouped_topk,
        dtypes=[torch.bfloat16],
        renormalize=True,
        scoring_func=1,
    )

    bench.set_gems(flag_gems.grouped_topk)
    bench.run()
