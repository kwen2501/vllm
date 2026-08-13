# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Symmetric-memory DCP All-to-All.

A fused variant of the ``a2a`` DCP combine in ``dcp_alltoall.py``. Rather than
packing into a staging buffer, handing that to NCCL, and unpacking the result,
the pack kernel stores each destination rank's slice straight into that rank's
symmetric-memory buffer over NVLink. One barrier later, the combine kernel
reads the local buffer and does the LSE-weighted reduction.

Per layer this is 2 kernels + 1 barrier instead of 2 kernels + an NCCL
all-to-all, and the partial outputs cross HBM twice instead of six times. The
win is largest at small decode batches, where the NCCL group launch dominates;
at large batch both paths move the same NVLink bytes and the gap narrows.

Falls back to the NCCL ``a2a`` path when symmetric memory is unavailable, the
DCP group spans nodes, or the batch exceeds the preallocated buffers.

Usage:
    vllm serve model --tp 8 --dcp 8 --dcp-comm-backend a2a_symm
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm import envs
from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator
    from vllm.v1.attention.ops.common import CPTritonContext

logger = init_logger(__name__)

try:
    import torch.distributed._symmetric_memory as torch_symm_mem

    _symm_mem_available = True
except ImportError:
    _symm_mem_available = False

# Two slots, used alternately, so a single barrier per call is enough. With one
# slot, rank r's put for call i+1 (issued after r clears barrier i) could land
# before peer p's combine for call i, which p only issues after that same
# barrier -- a write-after-read race. Alternating means the previous user of a
# slot is two calls back, and that read is ordered before the intervening
# barrier on p's own stream.
_NUM_SLOTS = 2


@triton.jit
def _dcp_symm_pack_put_kernel(
    out_ptr,
    lse_ptr,
    data_peer_ptrs,
    lse_peer_ptrs,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    lse_stride_B,
    lse_stride_H,
    data_stride_N,
    data_stride_B,
    data_stride_H,
    lse_buf_stride_N,
    lse_buf_stride_B,
    slot_data_offset,
    slot_lse_offset,
    RANK: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    H_PER_RANK: tl.constexpr,
):
    """Store this rank's partial output for each peer into that peer's buffer."""
    batch_idx = tl.program_id(0).to(tl.int64)
    local_head_idx = tl.program_id(1).to(tl.int64)
    dst_rank = tl.program_id(2).to(tl.int64)

    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < HEAD_DIM
    src_head_idx = dst_rank * H_PER_RANK + local_head_idx

    vals = tl.load(
        out_ptr
        + batch_idx * out_stride_B
        + src_head_idx * out_stride_H
        + d_offsets * out_stride_D,
        mask=d_mask,
    )
    lse_val = tl.load(
        lse_ptr + batch_idx * lse_stride_B + src_head_idx * lse_stride_H
    ).to(tl.float32)

    data_base = tl.load(data_peer_ptrs.to(tl.pointer_type(tl.uint64)) + dst_rank).to(
        tl.pointer_type(out_ptr.dtype.element_ty)
    )
    lse_base = tl.load(lse_peer_ptrs.to(tl.pointer_type(tl.uint64)) + dst_rank).to(
        tl.pointer_type(tl.float32)
    )

    tl.store(
        data_base
        + slot_data_offset
        + RANK * data_stride_N
        + batch_idx * data_stride_B
        + local_head_idx * data_stride_H
        + d_offsets,
        vals,
        mask=d_mask,
    )
    tl.store(
        lse_base
        + slot_lse_offset
        + RANK * lse_buf_stride_N
        + batch_idx * lse_buf_stride_B
        + local_head_idx,
        lse_val,
    )


@triton.jit
def _dcp_symm_combine_kernel(
    data_ptr,
    lse_buf_ptr,
    out_ptr,
    out_lse_ptr,
    data_stride_N,
    data_stride_B,
    data_stride_H,
    lse_buf_stride_N,
    lse_buf_stride_B,
    out_stride_B,
    out_stride_H,
    out_stride_D,
    out_lse_stride_B,
    out_lse_stride_H,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    IS_BASE_E: tl.constexpr,
    RETURN_LSE: tl.constexpr,
):
    """LSE-weighted reduction of the N partials this rank received."""
    batch_idx = tl.program_id(0).to(tl.int64)
    head_idx = tl.program_id(1).to(tl.int64)
    d_block = tl.program_id(2).to(tl.int64)

    n_offsets = tl.arange(0, BLOCK_N)
    n_mask = n_offsets < N
    d_offsets = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
    d_mask = d_offsets < HEAD_DIM

    lse = tl.load(
        lse_buf_ptr
        + n_offsets * lse_buf_stride_N
        + batch_idx * lse_buf_stride_B
        + head_idx,
        mask=n_mask,
        other=-float("inf"),
    )
    lse = tl.where((lse != lse) | (lse == float("inf")), -float("inf"), lse)

    lse_max = tl.max(lse, axis=0)
    lse_max = tl.where(lse_max == -float("inf"), 0.0, lse_max)
    exps = tl.exp(lse - lse_max) if IS_BASE_E else tl.exp2(lse - lse_max)
    exps = tl.where(n_mask, exps, 0.0)
    denom = tl.sum(exps, axis=0)
    weights = exps / tl.where(denom == 0.0, 1.0, denom)

    vals = tl.load(
        data_ptr
        + n_offsets[:, None] * data_stride_N
        + batch_idx * data_stride_B
        + head_idx * data_stride_H
        + d_offsets[None, :],
        mask=n_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    acc = tl.sum(vals.to(tl.float32) * weights[:, None], axis=0)

    tl.store(
        out_ptr
        + batch_idx * out_stride_B
        + head_idx * out_stride_H
        + d_offsets * out_stride_D,
        acc,
        mask=d_mask,
    )

    if RETURN_LSE and d_block == 0:
        if IS_BASE_E:  # noqa: SIM108
            global_lse = tl.log(denom) + lse_max
        else:
            global_lse = tl.log2(denom) + lse_max
        tl.store(
            out_lse_ptr + batch_idx * out_lse_stride_B + head_idx * out_lse_stride_H,
            global_lse,
        )


class _SymmA2AContext:
    """Persistent symmetric buffers for one DCP group and layer geometry."""

    def __init__(
        self,
        cp_group: GroupCoordinator,
        dtype: torch.dtype,
        h_per_rank: int,
        head_dim: int,
        device: torch.device,
        max_tokens: int,
    ):
        self.world_size = cp_group.world_size
        self.rank = cp_group.rank_in_group
        self.max_tokens = max_tokens

        self.data = torch_symm_mem.empty(
            _NUM_SLOTS,
            self.world_size,
            max_tokens,
            h_per_rank,
            head_dim,
            dtype=dtype,
            device=device,
        )
        self.lse = torch_symm_mem.empty(
            _NUM_SLOTS,
            self.world_size,
            max_tokens,
            h_per_rank,
            dtype=torch.float32,
            device=device,
        )
        group_name = cp_group.device_group.group_name
        self._data_handle = torch_symm_mem.rendezvous(self.data, group_name)
        self._lse_handle = torch_symm_mem.rendezvous(self.lse, group_name)
        self.data_peer_ptrs = self._data_handle.buffer_ptrs_dev
        self.lse_peer_ptrs = self._lse_handle.buffer_ptrs_dev
        # Advanced identically on every rank -- see the _NUM_SLOTS comment for
        # why lockstep is what makes a single barrier sufficient.
        self._slot = 0

    def next_slot(self) -> int:
        slot = self._slot
        self._slot = (self._slot + 1) % _NUM_SLOTS
        return slot

    def barrier(self) -> None:
        """Publish this rank's remote stores and wait for every peer's."""
        self._data_handle.barrier()


_contexts: dict[tuple, _SymmA2AContext | None] = {}


def _budget_max_tokens(
    world_size: int, h_per_rank: int, head_dim: int, itemsize: int
) -> int:
    """Largest batch the symmetric budget covers, across both slots.

    Batches beyond this fall back to the NCCL path, which is the right
    tradeoff: the fused path wins on latency at decode-sized batches, while
    large prefill batches are bandwidth-bound either way.
    """
    budget = envs.VLLM_DCP_SYMM_A2A_MAX_MB * 1024 * 1024
    per_token = _NUM_SLOTS * world_size * h_per_rank * (head_dim * itemsize + 4)
    return budget // per_token


def _get_context(
    cp_group: GroupCoordinator,
    dtype: torch.dtype,
    h_per_rank: int,
    head_dim: int,
    device: torch.device,
) -> _SymmA2AContext | None:
    """Return the context for this geometry, creating it once, or None.

    Every decision here is a function of rank-invariant values (group, layer
    geometry, config), so all DCP ranks agree on whether to build a context.
    They must: creation rendezvouses, and a rank that declined while others
    build would hang them.
    """
    key = (cp_group.device_group.group_name, dtype, h_per_rank, head_dim, device)
    if key in _contexts:
        return _contexts[key]

    if torch.cuda.is_current_stream_capturing():
        # rendezvous is a host-side collective and cannot be captured. Warmup
        # normally builds the context first; if it did not, decline for this
        # graph rather than corrupt the capture. Not cached, so a later eager
        # call can still build it.
        logger.warning_once(
            "DCP a2a_symm: no symmetric buffers for this shape at capture "
            "time; the captured graph will use the NCCL all-to-all."
        )
        return None

    ctx = _try_create_context(cp_group, dtype, h_per_rank, head_dim, device)
    _contexts[key] = ctx
    return ctx


def _try_create_context(
    cp_group: GroupCoordinator,
    dtype: torch.dtype,
    h_per_rank: int,
    head_dim: int,
    device: torch.device,
) -> _SymmA2AContext | None:
    from vllm.config import get_current_vllm_config_or_none
    from vllm.distributed.parallel_state import in_the_same_node_as

    if not _symm_mem_available:
        logger.warning_once(
            "DCP a2a_symm: torch symmetric memory is unavailable; "
            "falling back to the NCCL all-to-all."
        )
        return None

    max_tokens = _budget_max_tokens(
        cp_group.world_size, h_per_rank, head_dim, dtype.itemsize
    )
    vllm_config = get_current_vllm_config_or_none()
    if vllm_config is not None:
        max_tokens = min(
            max_tokens, vllm_config.scheduler_config.max_num_batched_tokens
        )
    if max_tokens < 1:
        logger.warning_once(
            "DCP a2a_symm: VLLM_DCP_SYMM_A2A_MAX_MB is too small for this "
            "layer geometry; falling back to the NCCL all-to-all."
        )
        return None

    if not all(in_the_same_node_as(cp_group.cpu_group)):
        logger.warning_once(
            "DCP a2a_symm: the DCP group spans nodes; falling back to the "
            "NCCL all-to-all."
        )
        return None

    try:
        ctx = _SymmA2AContext(cp_group, dtype, h_per_rank, head_dim, device, max_tokens)
    except (RuntimeError, AttributeError) as e:
        logger.warning_once(
            "DCP a2a_symm: symmetric memory setup failed (%s); falling back "
            "to the NCCL all-to-all.",
            str(e),
        )
        return None

    logger.info_once(
        "DCP a2a_symm: symmetric buffers ready for up to %d tokens (world size %d).",
        max_tokens,
        cp_group.world_size,
    )
    return ctx


def dcp_symm_a2a_lse_reduce(
    cp_attn_out: torch.Tensor,
    cp_attn_lse: torch.Tensor,
    cp_group: GroupCoordinator,
    ctx: CPTritonContext | None = None,
    return_lse: bool = False,
    is_lse_base_on_e: bool = True,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Combine partial attention outputs across DCP ranks over NVLink.

    Same contract as
    [`dcp_a2a_lse_reduce`][vllm.v1.attention.ops.dcp_alltoall.dcp_a2a_lse_reduce],
    which this transparently falls back to when the fused path does not apply.

    Args:
        cp_attn_out: [B, H, D] where B=num_tokens, H=total_heads, D=head_dim
        cp_attn_lse: [B, H] log-sum-exp values
        cp_group: GroupCoordinator for DCP communication
        ctx: CPTritonContext (unused, for signature compatibility)
        return_lse: If True, also return the combined global LSE
        is_lse_base_on_e: If True, LSE is base e; if False, base 2

    Returns:
        Combined output [B, H/N, D] (head-scattered)
        If return_lse=True, also returns global_lse [B, H/N]
    """
    world_size = cp_group.world_size

    if world_size == 1:
        if return_lse:
            return cp_attn_out, cp_attn_lse
        return cp_attn_out

    B, H, D = cp_attn_out.shape
    if H % world_size != 0:
        raise ValueError(f"H={H} must be divisible by DCP world size {world_size}.")
    H_per_rank = H // world_size

    symm = _get_context(cp_group, cp_attn_out.dtype, H_per_rank, D, cp_attn_out.device)
    # B == 0 goes to the NCCL path too: a zero-sized all-to-all is legal, a
    # zero-block grid is not.
    if symm is None or not 0 < B <= symm.max_tokens:
        return dcp_a2a_lse_reduce(
            cp_attn_out,
            cp_attn_lse,
            cp_group,
            ctx=ctx,
            return_lse=return_lse,
            is_lse_base_on_e=is_lse_base_on_e,
        )

    slot = symm.next_slot()
    # The put kernel moves a whole head row per program; the combine kernel
    # holds an [N, BLOCK_D] tile, so it splits wide heads to bound registers.
    put_block_d = triton.next_power_of_2(D)
    block_d = min(put_block_d, 128)

    _dcp_symm_pack_put_kernel[(B, H_per_rank, world_size)](
        cp_attn_out,
        cp_attn_lse,
        symm.data_peer_ptrs,
        symm.lse_peer_ptrs,
        cp_attn_out.stride(0),
        cp_attn_out.stride(1),
        cp_attn_out.stride(2),
        cp_attn_lse.stride(0),
        cp_attn_lse.stride(1),
        symm.data.stride(1),
        symm.data.stride(2),
        symm.data.stride(3),
        symm.lse.stride(1),
        symm.lse.stride(2),
        slot * symm.data.stride(0),
        slot * symm.lse.stride(0),
        RANK=symm.rank,
        HEAD_DIM=D,
        BLOCK_D=put_block_d,
        H_PER_RANK=H_per_rank,
    )

    symm.barrier()

    out = torch.empty(
        (B, H_per_rank, D), device=cp_attn_out.device, dtype=cp_attn_out.dtype
    )
    out_lse = torch.empty(
        (B, H_per_rank) if return_lse else (1, 1),
        device=cp_attn_out.device,
        dtype=torch.float32,
    )
    _dcp_symm_combine_kernel[(B, H_per_rank, triton.cdiv(D, block_d))](
        symm.data[slot],
        symm.lse[slot],
        out,
        out_lse,
        symm.data.stride(1),
        symm.data.stride(2),
        symm.data.stride(3),
        symm.lse.stride(1),
        symm.lse.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out_lse.stride(0),
        out_lse.stride(1),
        N=world_size,
        BLOCK_N=triton.next_power_of_2(world_size),
        HEAD_DIM=D,
        BLOCK_D=block_d,
        IS_BASE_E=is_lse_base_on_e,
        RETURN_LSE=return_lse,
    )

    if return_lse:
        return out, out_lse
    return out
