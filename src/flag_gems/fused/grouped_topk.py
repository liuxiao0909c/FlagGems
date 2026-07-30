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

import logging

import torch
import triton
import triton.language as tl
import triton.experimental.tle.language as tle
from flag_gems.utils import tl_extra_shim

logger = logging.getLogger(__name__)


@triton.jit
def topk_with_k2_triton(
    scores_ptr,
    bias_ptr,
    group_scores_ptr,
    num_experts_per_group,
    n_group,
    stride_scores_token,
    stride_group_scores_token,
    BLOCK_SIZE: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
):
    pid = tl.program_id(0)

    token_id = pid // n_group
    group_id = pid % n_group

    lane = tl.arange(0, BLOCK_SIZE)
    mask = lane < num_experts_per_group

    scores_offset = token_id * stride_scores_token + group_id * num_experts_per_group
    bias_offset = group_id * num_experts_per_group

    x = tl.load(
        scores_ptr + scores_offset + lane,
        mask=mask,
        other=-float("inf"),
    )

    b = tl.load(
        bias_ptr + bias_offset + lane,
        mask=mask,
        other=0.0,
    )

    x = x + b

    x_f32 = x.to(tl.float32)

    max1 = tl.max(x_f32, axis=0)
    is_max1 = (x_f32 == max1) & mask
    count_max1 = tl.sum(is_max1.to(tl.int32), axis=0)

    x2 = tl.where(
        is_max1 & (count_max1 == 1),
        -float("inf"),
        x_f32,
    )
    max2 = tl.max(x2, axis=0)

    group_scores_offset = token_id * stride_group_scores_token + group_id
    tl.store(
        group_scores_ptr + group_scores_offset,
        (max1 + max2).to(INPUT_DTYPE),
    )


@triton.jit
def group_idx_and_topk_triton(
    scores_ptr,
    group_scores_ptr,
    topk_values_ptr,
    topk_indices_ptr,
    bias_ptr,
    num_tokens,
    n_group,
    topk_group,
    topk,
    num_experts,
    num_experts_per_group,
    routed_scaling_factor,
    stride_scores_token,
    stride_group_scores_token,
    stride_out_token,
    N_GROUP: tl.constexpr,
    TOPK_GROUP: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_GROUP: tl.constexpr,
    BLOCK_EXPERT: tl.constexpr,
    INPUT_DTYPE: tl.constexpr,
    renormalize: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= num_tokens:
        return

    neg_inf = -float("inf")

    group_offsets = tl.arange(0, BLOCK_GROUP)
    valid_group = group_offsets < n_group

    group_scores = tl.load(
        group_scores_ptr + pid * stride_group_scores_token + group_offsets,
        mask=valid_group,
        other=neg_inf,
    )

    group_scores_f32 = group_scores.to(tl.float32)
    is_finite = (group_scores_f32 == group_scores_f32) & (
        group_scores_f32 != float("inf")
    )
    group_scores_f32 = tl.where(is_finite & valid_group, group_scores_f32, neg_inf)

    max_group_score = tl.max(group_scores_f32, axis=0)
    if_proceed = max_group_score != neg_inf

    value = group_scores_f32
    target_num_min = BLOCK_GROUP - n_group + topk_group
    count_equal_to_top_value = BLOCK_GROUP - n_group
    pre_count_equal_to_top_value = 0
    topk_group_value = neg_inf

    for _ in range(TOPK_GROUP):
        need = count_equal_to_top_value < target_num_min
        max_val = tl.max(value, axis=0)

        is_max = need & (value == max_val)
        value = tl.where(is_max, neg_inf, value)

        newly = tl.sum(is_max.to(tl.int32), axis=0)

        pre_count_equal_to_top_value = tl.where(
            need, count_equal_to_top_value, pre_count_equal_to_top_value
        )
        count_equal_to_top_value = tl.where(
            need, count_equal_to_top_value + newly, count_equal_to_top_value
        )
        topk_group_value = tl.where(need, max_val, topk_group_value)

    num_equalto_topkth_group = target_num_min - pre_count_equal_to_top_value

    group_gt = group_scores_f32 > topk_group_value
    group_eq = group_scores_f32 == topk_group_value

    eq_i = group_eq.to(tl.int32)
    prefix_eq = tl.cumsum(eq_i, axis=0) - eq_i

    group_selected = (
        group_gt | (group_eq & (prefix_eq < num_equalto_topkth_group))
    ) & valid_group

    expert_offsets = tl.arange(0, BLOCK_EXPERT)
    valid_expert = expert_offsets < num_experts
    expert_group = expert_offsets // num_experts_per_group

    expert_in_group = expert_group[:, None] == group_offsets[None, :]
    expert_selected = (
        tl.sum((expert_in_group & group_selected[None, :]).to(tl.int32), axis=1) > 0
    ) & valid_expert

    scored = tl.load(
        scores_ptr + pid * stride_scores_token + expert_offsets,
        mask=expert_selected,
        other=neg_inf,
    )

    expert_bias = tl.load(
        bias_ptr + expert_offsets,
        mask=valid_expert,
        other=0.0,
    )

    selection_scores_native = scored + expert_bias

    selection_scores = tl.where(
        expert_selected,
        selection_scores_native.to(tl.float32),
        neg_inf,
    )

    topk_vals = tl.full([TOPK], 0.0, tl.float32)
    topk_idx = tl.full([TOPK], 0, tl.int32)
    pos_range = tl.arange(0, TOPK)

    for i in range(TOPK):
        max_val = tl.max(selection_scores, axis=0)
        is_max = selection_scores == max_val

        candidate_idx = tl.where(is_max, expert_offsets, num_experts + 1)
        selected_idx = tl.min(candidate_idx, axis=0)

        selected_score = tl.load(
            scores_ptr + pid * stride_scores_token + selected_idx,
            mask=selected_idx < num_experts,
            other=neg_inf,
        ).to(tl.float32)

        topk_vals = tl.where(pos_range == i, selected_score, topk_vals)
        topk_idx = tl.where(pos_range == i, selected_idx.to(tl.int32), topk_idx)

        selection_scores = tl.where(
            expert_offsets == selected_idx, neg_inf, selection_scores
        )

    if renormalize == 1:
        topk_sum = tl.sum(topk_vals, axis=0) + 1e-20
        scale = routed_scaling_factor / topk_sum
    else:
        scale = routed_scaling_factor

    topk_vals = topk_vals * scale

    default_idx = pos_range.to(tl.int32)
    default_vals = tl.full([TOPK], 1.0 / topk, tl.float32)

    final_vals = tl.where(if_proceed, topk_vals, default_vals)
    final_idx = tl.where(if_proceed, topk_idx, default_idx)

    tl.store(
        topk_values_ptr + pid * stride_out_token + pos_range,
        final_vals,
        mask=pos_range < topk,
    )

    tl.store(
        topk_indices_ptr + pid * stride_out_token + pos_range,
        final_idx,
        mask=pos_range < topk,
    )


@triton.jit
def _sigmoid(x):
    log2e: tl.constexpr = 1.4426950408889634
    return 1 / (1 + tl_extra_shim.exp2(-x * log2e))


@triton.jit
def _pack_val_idx_fp32(val, idx):
    MAX_IDX: tl.constexpr = 0xFFFF
    bits = val.to(tl.uint32, bitcast=True)
    sign_bit = tl.full(val.shape, 0x80000000, tl.uint32)
    high = tl.where(bits & sign_bit, ~bits, bits | sign_bit).to(tl.uint64) << 32
    low = (0xFFFF & (MAX_IDX - idx)).to(tl.uint64)
    return high | low


@triton.jit
def _unpack_val_idx_fp32(pair):
    MAX_IDX: tl.constexpr = 0xFFFF
    idx = (MAX_IDX - (pair & 0xFFFF)).to(tl.uint32)
    sign_bit = tl.full(pair.shape, 0x80000000, tl.uint32)
    enc = (pair >> 32).to(tl.uint32)
    bits = tl.where(enc >= sign_bit, enc ^ sign_bit, ~enc)
    val = bits.to(tl.float32, bitcast=True)
    return val, idx


@triton.jit
def triton_grouped_topk_fused_small_expert_count_kernel(
    scores_ptr,
    topk_values_ptr,
    topk_indices_ptr,
    routing_bias_ptr,
    num_tokens,
    num_groups,
    topk_group,
    topk,
    num_experts,
    num_experts_per_group,
    renormalize,
    routed_scaling_factor,
    scores_stride0,
    SCORING_FUNC: tl.constexpr,
    MESH: tl.constexpr,
    CLUSTER_SIZE: tl.constexpr,
    MAX_NUM_EXPERTS: tl.constexpr,
    N_GROUPS_P: tl.constexpr,
):
    cluster_pid = tl.program_id(0)
    cluster_rank = tle.shard_id(MESH, "cluster_x")
    token_id = cluster_pid // CLUSTER_SIZE
    is_rank0 = cluster_rank == 0
    valid_rank = cluster_rank < num_groups  # always true?
    neg_inf: tl.constexpr = float("-inf")

    DUMP_FLAG = False #token_id == 2
    scores_ptr += token_id * scores_stride0
    topk_values_ptr += token_id * topk
    topk_indices_ptr += token_id * topk

    s_score_sigmoid = tle.gpu.alloc(
        [32],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_score_bias = tle.gpu.alloc(
        [32],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_group_scores = tle.gpu.alloc(
        [N_GROUPS_P],
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_top_group_idx = tle.gpu.alloc(
        [4], # MAX_NUM_TOP_GROUPS
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_expert_score_group = tle.gpu.alloc(
        [32], # WARP_SIZE
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_expert_idx_group = tle.gpu.alloc(
        [32], # WARP_SIZE
        dtype=tl.int32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )
    s_top_score_sigmoid = tle.gpu.alloc(
        [8], # MAX_NUM_TOP_EXPERTS
        dtype=tl.float32,
        layout=None,
        scope=tle.gpu.smem,
        nv_mma_shared_layout=False,
    )

    s_score_sigmoid_ptr = tle.gpu.local_ptr(s_score_sigmoid, (0,))
    s_score_bias_ptr = tle.gpu.local_ptr(s_score_bias, (0,))
    s_group_scores_ptr = tle.gpu.local_ptr(s_group_scores, (0,))
    s_group_scores_rank0_ptr = tle.remote(s_group_scores_ptr, 0, scope=MESH)
    s_top_group_idx_ptr = tle.gpu.local_ptr(s_top_group_idx, (0,))
    s_top_group_idx_rank0_ptr = tle.remote(s_top_group_idx_ptr, 0, scope=MESH)
    s_expert_score_group_ptr = tle.gpu.local_ptr(s_expert_score_group, (0,))
    s_expert_score_group_rank0_ptr = tle.remote(s_expert_score_group_ptr, 0, scope=MESH)
    s_expert_idx_group_ptr = tle.gpu.local_ptr(s_expert_idx_group, (0,))
    s_expert_idx_group_rank0_ptr = tle.remote(s_expert_idx_group_ptr, 0, scope=MESH)
    s_top_score_sigmoid_ptr = tle.gpu.local_ptr(s_top_score_sigmoid, (0,))
    s_top_score_sigmoid_rank0_ptr = tle.remote(s_top_score_sigmoid_ptr, 0, scope=MESH)

    # step1: load score/bias to smem
    lane = tl.arange(0, 32)
    zeros = tl.zeros([32], dtype=tl.int32)
    score = tl.load(
        scores_ptr + num_experts_per_group * cluster_rank + lane,
        mask=valid_rank & (lane < num_experts_per_group),
        other=neg_inf,
    ).to(tl.float32)
    if DUMP_FLAG:
        tl.device_print("score_f32:", score)
    if SCORING_FUNC == 1:
        score_sigmoid = _sigmoid(score)
    else:
        score_sigmoid = score
    if DUMP_FLAG:
        tl.device_print("score_sigmoid:", score_sigmoid)
    tl.store(s_score_sigmoid_ptr + lane, score_sigmoid, mask=valid_rank & (lane < num_experts_per_group))

    bias_val = tl.load(
        routing_bias_ptr + num_experts_per_group * cluster_rank + lane,
        mask=valid_rank & (lane < num_experts_per_group),
        other=neg_inf,
    ).to(tl.float32)
    score_bias = score_sigmoid + bias_val
    if DUMP_FLAG:
        tl.device_print("score_bias:", score_bias)
    tl.store(s_score_bias_ptr + lane, score_bias, mask=valid_rank & (lane < num_experts_per_group))

    # step2: each rank get top2 as group_scores
    MAX_IDX: tl.constexpr = 65535
    min_val = tl.full((32,), neg_inf, dtype=tl.float32)
    comp_val_idx = _pack_val_idx_fp32(score_bias, lane)
    packed_max1 = tl.max(comp_val_idx)
    val_max1, _1 = _unpack_val_idx_fp32(packed_max1)
    comp_val_idx = tl.where(
        comp_val_idx == packed_max1,
        _pack_val_idx_fp32(min_val, lane),
        comp_val_idx,
    )
    packed_max2 = tl.max(comp_val_idx)
    val_max2, _1 = _unpack_val_idx_fp32(packed_max2)
    group_score = val_max1 + val_max2
    if DUMP_FLAG:
        tl.device_print("group_score:", group_score)
    tl.store(s_group_scores_rank0_ptr + cluster_rank + zeros, group_score, mask=valid_rank & (lane == 0))
    tle.distributed_barrier(MESH)

    # step3: rank0 get topk_group
    if is_rank0:
        group_scores = tl.load(s_group_scores_ptr + lane, mask=lane < num_groups, other=neg_inf)
        if DUMP_FLAG:
            tl.device_print("group_scores_gbl:", group_scores)
        comp_val_idx = _pack_val_idx_fp32(group_scores, lane)
        for t in tl.range(0, topk_group):
            packed_max = tl.max(comp_val_idx)
            comp_val_idx = tl.where(
                comp_val_idx == packed_max,
                _pack_val_idx_fp32(min_val, lane),
                comp_val_idx,
            )
            _, top_group_idx = _unpack_val_idx_fp32(packed_max)
            if DUMP_FLAG:
                tl.device_print("top_group_idx:", top_group_idx)
            tl.store(s_top_group_idx_ptr + t + zeros, top_group_idx, mask=lane == t)
    tle.distributed_barrier(MESH)

    # step4: each topk_group(<=4) get topk(<=8), also copy score_sigmoid
    if valid_rank:
        top_group_idx_gbl = tl.load(s_top_group_idx_rank0_ptr + lane, mask=lane < topk_group, other=MAX_IDX)
        group_idx_match = top_group_idx_gbl == cluster_rank
        top_group_idx_gbl = tl.where(group_idx_match, lane, MAX_IDX)
        group_offset = tl.min(top_group_idx_gbl)
        if group_offset < MAX_IDX:
            if DUMP_FLAG:
                tl.device_print("top_group_idx_gbl:", top_group_idx_gbl)
            score_bias = tl.load(s_score_bias_ptr + lane, mask=valid_rank & (lane < num_experts_per_group), other=neg_inf) # TODO: just use previous score_bias?
            idx = cluster_rank * num_experts_per_group + lane
            comp_val_idx = _pack_val_idx_fp32(score_bias, idx)
            for t in tl.range(0, topk):
                packed_max = tl.max(comp_val_idx)
                comp_val_idx = tl.where(
                    comp_val_idx == packed_max,
                    _pack_val_idx_fp32(min_val, idx),
                    comp_val_idx,
                )
                top_val, top_idx = _unpack_val_idx_fp32(packed_max)
                tl.store(s_expert_idx_group_rank0_ptr + group_offset * topk + t + zeros, top_idx, mask=lane == t)
                tl.store(s_expert_score_group_rank0_ptr + group_offset * topk + t + zeros, top_val, mask=lane == t)
    tle.distributed_barrier(MESH)

    # step5: rank0 get final topk, just use tl.sort, not array of 4 like cuda
    if is_rank0:
        top_idxs = tl.load(s_expert_idx_group_ptr + lane, mask=lane < topk_group * topk, other=MAX_IDX)
        top_vals = tl.load(s_expert_score_group_ptr + lane, mask=lane < topk_group * topk, other=neg_inf)
        comp_val_idx = _pack_val_idx_fp32(top_vals, top_idxs)
        comp_val_idx = tl.sort(comp_val_idx, dim=0, descending=True)
        _, top_idxs = _unpack_val_idx_fp32(comp_val_idx)
        if DUMP_FLAG:
            tl.device_print("top_idxs:", top_idxs)
        tl.store(s_expert_idx_group_ptr + lane, top_idxs, mask=lane < topk)
    tle.distributed_barrier(MESH)

    if renormalize:
        # step6_1: copy score_sigmoid for tl.sum
        if valid_rank:
            top_idxs_gbl = tl.load(s_expert_idx_group_rank0_ptr + lane, mask=lane < topk, other=MAX_IDX)
            top_idxs_local = top_idxs_gbl - cluster_rank * num_experts_per_group
            mask = (top_idxs_local >= 0) & (top_idxs_local < num_experts_per_group)
            # output topk_indices
            tl.store(topk_indices_ptr + lane, top_idxs_gbl, mask=valid_rank & mask)
            score_norm_local = tl.load(s_score_sigmoid_ptr + top_idxs_local, mask=valid_rank & mask, other=0.0)
            if DUMP_FLAG:
                tl.device_print("top_idxs_gbl:", top_idxs_gbl)
                tl.device_print("top_idxs_gbl_mask:", valid_rank & mask)
                tl.device_print("score_norm_local:", score_norm_local)
            tl.store(s_top_score_sigmoid_rank0_ptr + lane, score_norm_local, mask=valid_rank & mask)
        tle.distributed_barrier(MESH)
        # step7: rank0 output
        if is_rank0:
            score_norm_gbl = tl.load(s_top_score_sigmoid_ptr + lane, mask=lane < topk, other=0.0)
            red_norm = tl.sum(score_norm_gbl)
            final_score = score_norm_gbl / (red_norm + 1e-20)
            # output topk_values
            tl.store(topk_values_ptr + lane, final_score, mask=lane < topk)
            if DUMP_FLAG:
                tl.device_print("score_norm_gbl:", score_norm_gbl)
                tl.device_print("final_score:", final_score)
    else:
        if valid_rank:
            top_idxs_gbl = tl.load(s_expert_idx_group_rank0_ptr + lane, mask=lane < topk, other=MAX_IDX)
            top_idxs_local = top_idxs_gbl - cluster_rank * num_experts_per_group
            mask = (top_idxs_local >= 0) & (top_idxs_local < num_experts_per_group)
            # output topk_indices
            tl.store(topk_indices_ptr + lane, top_idxs_gbl, mask=valid_rank & mask)
            score_norm_local = tl.load(s_score_sigmoid_ptr + top_idxs_local, mask=valid_rank & mask, other=0.0)
            # output topk_values
            tl.store(topk_values_ptr + lane, score_norm_local, mask=valid_rank & mask)


def grouped_topk(
    scores: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    renormalize: bool,
    routed_scaling_factor: float,
    bias: torch.Tensor,
    scoring_func: int = 0,
):
    logger.debug("GEMS GROUPED TOPK")
    if scores.ndim != 2:
        raise ValueError("scores must be a 2D Tensor")
    num_tokens, num_experts = scores.shape
    if num_experts % n_group != 0:
        raise ValueError("num_experts must be divisible by n_group")
    if n_group > 32:
        raise ValueError("n_group should be smaller than or equal to 32")
    if topk > 32:
        raise ValueError("topk should be smaller than or equal to 32 for now")
    if scoring_func not in (0, 1):
        raise ValueError("scoring_func must be 0 (none) or 1 (sigmoid)")

    if bias.dtype != scores.dtype:
        bias = bias.to(scores.dtype)
    if bias.ndim != 1:
        bias = bias.flatten()
    if len(bias) != num_experts:
        raise ValueError(
            f"bias length ({len(bias)}) must match num_experts ({num_experts})"
        )

    num_experts_per_group = num_experts // n_group

    if scores.dtype == torch.float32:
        INPUT_DTYPE = tl.float32
    elif scores.dtype == torch.float16:
        INPUT_DTYPE = tl.float16
    elif scores.dtype == torch.bfloat16:
        INPUT_DTYPE = tl.bfloat16
    else:
        raise ValueError(f"Unsupported dtype: {scores.dtype}")

    if ((n_group > 1) & (n_group <= 32) & (num_experts <= 256) & (num_experts_per_group <= 32) &
        (num_experts_per_group * topk_group <= 128) & (topk <= 8) &
        (topk_group <= 4)):
        #import pdb; pdb.set_trace()
        topk_values = torch.empty(
            (num_tokens, topk),
            device=scores.device,
            dtype=torch.float32,
        )    
        topk_indices = torch.empty(
            (num_tokens, topk),
            device=scores.device,
            dtype=torch.int32,
        )
        TLE_SMEM_CLUSTER_SIZE = n_group
        BLOCK_CLUSTER_MESH = tle.device_mesh({"block_cluster": [("cluster_x", TLE_SMEM_CLUSTER_SIZE)]})
        triton_grouped_topk_fused_small_expert_count_kernel[(num_tokens,)](
            scores,
            topk_values,
            topk_indices,
            bias,
            num_tokens,
            n_group,
            topk_group,
            topk,
            num_experts,
            num_experts_per_group,
            renormalize,
            routed_scaling_factor,
            scores.stride(0),
            SCORING_FUNC=scoring_func,
            MESH=BLOCK_CLUSTER_MESH,
            CLUSTER_SIZE=TLE_SMEM_CLUSTER_SIZE,
            MAX_NUM_EXPERTS=256,
            N_GROUPS_P=triton.next_power_of_2(n_group),
            num_warps=TLE_SMEM_CLUSTER_SIZE,
        )
        return topk_values, topk_indices

    if scoring_func == 1:
        scores_processed = torch.sigmoid(scores.float()).to(scores.dtype)
    else:
        scores_processed = scores

    group_scores = torch.empty(
        (num_tokens, n_group),
        device=scores.device,
        dtype=scores.dtype,
    )

    topk_values = torch.empty(
        (num_tokens, topk),
        device=scores.device,
        dtype=torch.float32,
    )

    topk_indices = torch.empty(
        (num_tokens, topk),
        device=scores.device,
        dtype=torch.int32,
    )

    BLOCK1 = triton.next_power_of_2(num_experts_per_group)
    grid1 = (num_tokens * n_group,)

    topk_with_k2_triton[grid1](
        scores_processed,
        bias,
        group_scores,
        num_experts_per_group,
        n_group,
        scores_processed.stride(0),
        group_scores.stride(0),
        BLOCK_SIZE=BLOCK1,
        INPUT_DTYPE=INPUT_DTYPE,
    )

    BLOCK_GROUP = triton.next_power_of_2(n_group)
    BLOCK_EXPERT = triton.next_power_of_2(num_experts)
    grid2 = (num_tokens,)

    group_idx_and_topk_triton[grid2](
        scores_processed,
        group_scores,
        topk_values,
        topk_indices,
        bias,
        num_tokens,
        n_group,
        topk_group,
        topk,
        num_experts,
        num_experts_per_group,
        routed_scaling_factor,
        scores_processed.stride(0),
        group_scores.stride(0),
        topk_values.stride(0),
        N_GROUP=n_group,
        TOPK_GROUP=topk_group,
        TOPK=topk,
        BLOCK_GROUP=BLOCK_GROUP,
        BLOCK_EXPERT=BLOCK_EXPERT,
        INPUT_DTYPE=INPUT_DTYPE,
        renormalize=int(renormalize),
    )

    return topk_values, topk_indices
