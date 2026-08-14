# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for weight transfer engine backends.

Unit tests for engine classes (parsing, validation, registry).
Integration tests for NCCL and IPC weight transfer between processes using Ray.
"""

import pickle
import sys
import threading
import time
import types
from unittest.mock import MagicMock

import pybase64 as base64
import pytest
import ray
import torch
from torch.multiprocessing.reductions import reduce_tensor

from vllm.config.parallel import ParallelConfig
from vllm.config.weight_transfer import WeightTransferConfig
from vllm.distributed.weight_transfer import (
    HTTPVLLMWeightSyncClient,
    ModuleSource,
    ParamMeta,
    RayVLLMWeightSyncClient,
    TrainerWeightTransferEngine,
    VLLMWeightSyncClient,
    WeightSource,
    WeightTransferEngineFactory,
    WeightTransferTrainerFactory,
)
from vllm.distributed.weight_transfer.base import (
    TrainerInitInfo,
    WeightTransferInitRequest,
    WeightTransferUpdateRequest,
)
from vllm.distributed.weight_transfer.ipc_engine import (
    IPCTrainerInitInfo,
    IPCTrainerWeightTransferEngine,
    IPCWeightTransferEngine,
    IPCWeightTransferInitInfo,
    IPCWeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.m2n_common import (
    REPLICATE,
    REPLICATED,
    M2NMesh,
    check_placements,
    check_transferable,
    resolve_layout,
    validate_layout,
)
from vllm.distributed.weight_transfer.m2n_engine import (
    M2NWeightTransferEngine,
    M2NWeightTransferInitInfo,
    M2NWeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.nccl_engine import (
    NCCLTrainerInitInfo,
    NCCLTrainerWeightTransferEngine,
    NCCLWeightTransferEngine,
    NCCLWeightTransferInitInfo,
    NCCLWeightTransferUpdateInfo,
)
from vllm.distributed.weight_transfer.packed_tensor import (
    DEFAULT_PACKED_BUFFER_SIZE_BYTES,
    DEFAULT_PACKED_NUM_BUFFERS,
)
from vllm.distributed.weight_transfer.sparse_nccl_engine import (
    SparseNCCLTrainerInitInfo,
    SparseNCCLTrainerWeightTransferEngine,
    SparseNCCLWeightTransferEngine,
    SparseNCCLWeightTransferUpdateInfo,
    SparseWeightPatch,
)
from vllm.platforms import current_platform
from vllm.utils.network_utils import get_open_port


def _init_ray_for_weight_transfer() -> None:
    if ray.is_initialized():
        return
    ray.init(
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {
                "RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_HIP_VISIBLE_DEVICES": "1",
                "RAY_EXPERIMENTAL_NOSET_ROCR_VISIBLE_DEVICES": "1",
            }
        },
    )


def _get_ray_assigned_device() -> torch.device:
    gpu_ids = ray.get_gpu_ids()
    if not gpu_ids:
        return torch.device("cuda:0")
    return torch.device(f"cuda:{int(gpu_ids[0])}")


def _set_ray_assigned_device() -> torch.device:
    device = _get_ray_assigned_device()
    current_platform.set_device(device)
    return device


def create_mock_parallel_config(
    rank: int = 0,
    world_size: int = 1,
    dp_rank: int = 0,
) -> ParallelConfig:
    """Create a mock ParallelConfig for testing."""
    config = MagicMock(spec=ParallelConfig)
    config.rank = rank
    config.world_size = world_size
    config.data_parallel_rank = dp_rank
    config.data_parallel_index = dp_rank
    return config


def create_mock_vllm_config(
    rank: int = 0,
    world_size: int = 1,
    dp_rank: int = 0,
) -> MagicMock:
    """Create a mock VllmConfig exposing parallel_config and model_config."""
    vllm_config = MagicMock()
    vllm_config.parallel_config = create_mock_parallel_config(rank, world_size, dp_rank)
    vllm_config.model_config = MagicMock()
    return vllm_config


# --- Unit Tests: NCCLWeightTransferUpdateInfo Validation ---


class TestNCCLWeightTransferUpdateInfoValidation:
    """Test NCCLWeightTransferUpdateInfo dataclass validation."""

    def test_valid_update_info(self):
        info = NCCLWeightTransferUpdateInfo(
            names=["layer.weight", "layer.bias"],
            dtype_names=["float32", "float32"],
            shapes=[[10, 10], [10]],
        )
        assert info.names == ["layer.weight", "layer.bias"]
        assert info.dtype_names == ["float32", "float32"]
        assert info.shapes == [[10, 10], [10]]

    def test_mismatched_dtype_names_raises(self):
        with pytest.raises(ValueError, match="dtype_names"):
            NCCLWeightTransferUpdateInfo(
                names=["layer.weight", "layer.bias"],
                dtype_names=["float32"],  # Only one dtype
                shapes=[[10, 10], [10]],
            )

    def test_mismatched_shapes_raises(self):
        with pytest.raises(ValueError, match="shapes"):
            NCCLWeightTransferUpdateInfo(
                names=["layer.weight", "layer.bias"],
                dtype_names=["float32", "float32"],
                shapes=[[10, 10]],  # Only one shape
            )

    def test_empty_lists_valid(self):
        info = NCCLWeightTransferUpdateInfo(names=[], dtype_names=[], shapes=[])
        assert len(info.names) == 0


# --- Unit Tests: SparseNCCLWeightTransferUpdateInfo Validation ---


class TestSparseNCCLWeightTransferUpdateInfoValidation:
    """Test SparseNCCLWeightTransferUpdateInfo dataclass validation."""

    def test_valid_sparse_update_info(self):
        info = SparseNCCLWeightTransferUpdateInfo(
            names=["layer.weight", "layer.bias"],
            dtype_names=["float32", "bfloat16"],
            shapes=[[10, 10], [10]],
            num_updates_list=[4, 2],
        )
        assert info.num_updates_list == [4, 2]

    def test_mismatched_dtype_names_raises(self):
        with pytest.raises(ValueError, match="dtype_names"):
            SparseNCCLWeightTransferUpdateInfo(
                names=["layer.weight", "layer.bias"],
                dtype_names=["float32"],
                shapes=[[10, 10], [10]],
                num_updates_list=[4, 2],
            )

    def test_rejects_empty_num_updates_list(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            SparseNCCLWeightTransferUpdateInfo(
                names=[],
                dtype_names=[],
                shapes=[],
                num_updates_list=[],
            )

    def test_rejects_mismatched_num_updates(self):
        with pytest.raises(ValueError, match="`num_updates_list`"):
            SparseNCCLWeightTransferUpdateInfo(
                names=["layer.weight", "layer.bias"],
                dtype_names=["float32", "float32"],
                shapes=[[10, 10], [10]],
                num_updates_list=[3],
            )

    def test_rejects_negative_num_updates(self):
        with pytest.raises(ValueError, match="non-negative"):
            SparseNCCLWeightTransferUpdateInfo(
                names=["layer.weight"],
                dtype_names=["float32"],
                shapes=[[10, 10]],
                num_updates_list=[-1],
            )


# --- Unit Tests: Engine Parsing ---


class TestNCCLEngineParsing:
    """Test NCCLWeightTransferEngine parsing methods."""

    def _make_engine(self):
        config = WeightTransferConfig(backend="nccl")
        return NCCLWeightTransferEngine(
            config,
            create_mock_vllm_config(),
            torch.device("cuda"),
            MagicMock(spec=torch.nn.Module),
        )

    def test_parse_init_info_valid(self):
        engine = self._make_engine()
        init_info = engine.parse_init_info(
            {
                "master_address": "127.0.0.1",
                "master_port": 12345,
                "rank_offset": 1,
                "world_size": 3,
            }
        )
        assert isinstance(init_info, NCCLWeightTransferInitInfo)
        assert init_info.master_address == "127.0.0.1"
        assert init_info.master_port == 12345
        assert init_info.rank_offset == 1
        assert init_info.world_size == 3

    def test_parse_init_info_missing_field_raises(self):
        engine = self._make_engine()
        with pytest.raises(ValueError, match="Invalid init_info"):
            engine.parse_init_info({"master_address": "127.0.0.1"})

    def test_parse_update_info_valid(self):
        engine = self._make_engine()
        update_info = engine.parse_update_info(
            {
                "names": ["w1", "w2"],
                "dtype_names": ["float32", "bfloat16"],
                "shapes": [[100, 100], [50]],
            }
        )
        assert isinstance(update_info, NCCLWeightTransferUpdateInfo)
        assert update_info.names == ["w1", "w2"]
        assert update_info.dtype_names == ["float32", "bfloat16"]
        assert update_info.shapes == [[100, 100], [50]]


# --- Unit Tests: Engine Registry ---


class TestEngineRegistry:
    """Test weight transfer engine registry."""

    def test_create_engine_nccl(self):
        config = WeightTransferConfig(backend="nccl")
        engine = WeightTransferEngineFactory.create_engine(
            config,
            create_mock_vllm_config(),
            torch.device("cuda"),
            MagicMock(spec=torch.nn.Module),
        )
        assert isinstance(engine, NCCLWeightTransferEngine)

    def test_create_engine_ipc(self):
        config = WeightTransferConfig(backend="ipc")
        engine = WeightTransferEngineFactory.create_engine(
            config,
            create_mock_vllm_config(),
            torch.device("cuda"),
            MagicMock(spec=torch.nn.Module),
        )
        assert isinstance(engine, IPCWeightTransferEngine)

    def test_create_engine_sparse_nccl(self):
        config = WeightTransferConfig(backend="sparse_nccl")
        engine = WeightTransferEngineFactory.create_engine(
            config,
            create_mock_vllm_config(),
            torch.device("cuda"),
            MagicMock(spec=torch.nn.Module),
        )
        assert isinstance(engine, SparseNCCLWeightTransferEngine)

    def test_create_engine_invalid_backend(self):
        config = WeightTransferConfig(backend="invalid")
        with pytest.raises(ValueError, match="Invalid weight transfer backend"):
            WeightTransferEngineFactory.create_engine(
                config,
                create_mock_vllm_config(),
                torch.device("cuda"),
                MagicMock(spec=torch.nn.Module),
            )

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError, match="already registered"):
            WeightTransferEngineFactory.register_engine(
                "nccl", NCCLWeightTransferEngine
            )

    def test_worker_registry_exposes_nccl_m2n(self):
        assert "nccl_m2n" in WeightTransferEngineFactory._registry


# --- Unit Tests: Sparse patch application (CPU) ---


class TestSparseNCCLPatchApplication:
    """Test SparseNCCLWeightTransferEngine._apply_patch on a real param."""

    def _make_engine(self, model):
        config = WeightTransferConfig(backend="sparse_nccl")
        return SparseNCCLWeightTransferEngine(
            config, create_mock_vllm_config(), torch.device("cpu"), model
        )

    def _make_model(self, numel: int = 8):
        model = torch.nn.Module()
        model.register_parameter(
            "w", torch.nn.Parameter(torch.zeros(numel), requires_grad=False)
        )

        def get_parameter(name):
            assert name == "w"
            return model.w

        model.get_parameter = get_parameter
        return model

    def test_apply_patch_updates_only_selected_entries(self):
        model = self._make_model(8)
        engine = self._make_engine(model)
        engine._apply_patch(
            SparseWeightPatch(
                name="w",
                indices=torch.tensor([1, 3], dtype=torch.int32),
                values=torch.tensor([5.0, 7.0], dtype=torch.float32),
            )
        )
        expected = torch.zeros(8)
        expected[1] = 5.0
        expected[3] = 7.0
        assert torch.equal(model.w.data, expected)

    def test_apply_patch_rejects_mismatched_lengths(self):
        model = self._make_model(8)
        engine = self._make_engine(model)
        with pytest.raises(ValueError, match="matching lengths"):
            engine._apply_patch(
                SparseWeightPatch(
                    name="w",
                    indices=torch.tensor([1, 3], dtype=torch.int32),
                    values=torch.tensor([5.0], dtype=torch.float32),
                )
            )

    def test_apply_patch_rejects_non_int32_indices(self):
        model = self._make_model(8)
        engine = self._make_engine(model)
        with pytest.raises(ValueError, match="int32 indices"):
            engine._apply_patch(
                SparseWeightPatch(
                    name="w",
                    indices=torch.tensor([1], dtype=torch.int64),
                    values=torch.tensor([5.0], dtype=torch.float32),
                )
            )

    def test_apply_patch_rejects_dtype_mismatch(self):
        model = self._make_model(8)
        engine = self._make_engine(model)
        with pytest.raises(ValueError, match="does not match"):
            engine._apply_patch(
                SparseWeightPatch(
                    name="w",
                    indices=torch.tensor([1], dtype=torch.int32),
                    values=torch.tensor([5.0], dtype=torch.bfloat16),
                )
            )

    def test_apply_patch_rejects_non_contiguous_param(self):
        model = torch.nn.Module()
        model.register_parameter(
            "w",
            torch.nn.Parameter(
                torch.arange(12, dtype=torch.float32).view(3, 4).t(),
                requires_grad=False,
            ),
        )
        model.get_parameter = lambda name: model.w
        engine = self._make_engine(model)
        with pytest.raises(NotImplementedError, match="contiguous params"):
            engine._apply_patch(
                SparseWeightPatch(
                    name="w",
                    indices=torch.tensor([1], dtype=torch.int32),
                    values=torch.tensor([1.0], dtype=torch.float32),
                )
            )


# --- Test receive_weights without init raises ---


def test_nccl_receive_weights_without_init_raises():
    """Test that receive_weights raises if init_transfer_engine wasn't called."""
    if torch.accelerator.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    config = WeightTransferConfig(backend="nccl")
    engine = NCCLWeightTransferEngine(
        config,
        create_mock_vllm_config(),
        torch.device("cuda"),
        MagicMock(spec=torch.nn.Module),
    )

    update_info = NCCLWeightTransferUpdateInfo(
        names=["w"], dtype_names=["float32"], shapes=[[10]]
    )

    with pytest.raises(RuntimeError, match="not initialized"):
        engine.receive_weights(update_info)


def test_sparse_nccl_receive_weights_without_init_raises():
    """Test that sparse receive raises if init_transfer_engine wasn't called."""
    if torch.accelerator.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    config = WeightTransferConfig(backend="sparse_nccl")
    engine = SparseNCCLWeightTransferEngine(
        config,
        create_mock_vllm_config(),
        torch.device("cuda"),
        MagicMock(spec=torch.nn.Module),
    )

    update_info = SparseNCCLWeightTransferUpdateInfo(
        names=["w"],
        dtype_names=["float32"],
        shapes=[[10]],
        num_updates_list=[2],
    )

    with pytest.raises(RuntimeError, match="not initialized"):
        engine.receive_weights(update_info)


# --- Integration Test: NCCL Weight Transfer Between Ray Tasks ---


@ray.remote(num_gpus=1)
def trainer_broadcast_tensor(
    master_address: str,
    master_port: int,
    world_size: int,
    tensor_shape: list[int],
    tensor_dtype: str,
) -> bool:
    """Trainer task that broadcasts a tensor via NCCL."""
    import torch

    device = _set_ray_assigned_device()

    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    # Create process group as rank 0 (trainer)
    pg = StatelessProcessGroup.create(
        host=master_address,
        port=master_port,
        rank=0,
        world_size=world_size,
    )
    comm = PyNcclCommunicator(pg, device=device.index)

    # Create and broadcast the tensor
    dtype = getattr(torch, tensor_dtype)
    tensor_to_send = torch.ones(tensor_shape, dtype=dtype, device=device)
    comm.broadcast(tensor_to_send, src=0, stream=torch.cuda.current_stream())
    torch.accelerator.synchronize()

    return True


@ray.remote(num_gpus=1)
def inference_receive_tensor(
    master_address: str,
    master_port: int,
    world_size: int,
    tensor_shape: list[int],
    tensor_dtype: str,
) -> dict:
    """Inference task that receives tensor via NCCLWeightTransferEngine."""
    import contextlib
    from unittest.mock import MagicMock

    import torch

    _set_ray_assigned_device()

    from vllm.config.parallel import ParallelConfig
    from vllm.config.weight_transfer import WeightTransferConfig
    from vllm.distributed.weight_transfer.nccl_engine import (
        NCCLWeightTransferEngine,
        NCCLWeightTransferInitInfo,
        NCCLWeightTransferUpdateInfo,
    )

    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.received = []

        def load_weights(self, weights):
            for name, tensor in weights:
                self.received.append((name, tensor.clone()))

    config = WeightTransferConfig(backend="nccl")
    vllm_config = MagicMock()
    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.rank = 0
    parallel_config.world_size = 1
    parallel_config.data_parallel_rank = 0
    parallel_config.data_parallel_index = 0
    vllm_config.parallel_config = parallel_config
    vllm_config.model_config = MagicMock()

    recorder = Recorder()
    engine = NCCLWeightTransferEngine(
        config, vllm_config, torch.device("cuda"), recorder
    )
    # Transport-only test: bypass the set_current_vllm_config context that
    # receive_weights enters, since vllm_config here is a mock.
    import vllm.config as _vllm_config_mod

    _vllm_config_mod.set_current_vllm_config = lambda cfg: contextlib.nullcontext()

    # Initialize the engine (joins as rank 1)
    # Trainer broadcasts a single tensor unpacked, so the worker must not
    # expect the packed wire format (packed is a must-agree wire param shipped
    # on the init info).
    init_info = NCCLWeightTransferInitInfo(
        master_address=master_address,
        master_port=master_port,
        rank_offset=1,  # Trainer is rank 0, we become rank 1
        world_size=world_size,
        packed=False,
    )
    engine.init_transfer_engine(init_info)

    update_info = NCCLWeightTransferUpdateInfo(
        names=["test.weight"],
        dtype_names=[tensor_dtype],
        shapes=[tensor_shape],
    )
    engine.receive_weights(update_info)
    torch.accelerator.synchronize()

    # Verify we received the tensor
    success = False
    received_shape = None
    received_sum = None

    if len(recorder.received) == 1:
        name, tensor = recorder.received[0]
        received_shape = list(tensor.shape)
        received_sum = tensor.sum().item()
        if received_shape == tensor_shape:
            expected_sum = 1.0 * torch.tensor(tensor_shape).prod().item()
            if abs(received_sum - expected_sum) < 0.01:
                success = True

    engine.shutdown()

    return {
        "success": success,
        "received_shape": received_shape,
        "received_sum": received_sum,
    }


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run NCCL weight transfer test.",
)
def test_nccl_weight_transfer_between_processes():
    """Test NCCL weight transfer from trainer to inference process using Ray.

    This test verifies that the NCCLWeightTransferEngine can receive
    tensors broadcast by a trainer process via NCCL.
    """
    _init_ray_for_weight_transfer()

    master_address = "127.0.0.1"
    master_port = get_open_port()
    world_size = 2  # 1 trainer + 1 inference worker

    tensor_shape = [100, 100]
    tensor_dtype = "float32"

    inference_future = inference_receive_tensor.remote(
        master_address, master_port, world_size, tensor_shape, tensor_dtype
    )
    trainer_future = trainer_broadcast_tensor.remote(
        master_address, master_port, world_size, tensor_shape, tensor_dtype
    )

    trainer_result, result = ray.get([trainer_future, inference_future])

    assert trainer_result, "Trainer should complete successfully"
    assert result["success"], (
        f"Weight transfer failed. "
        f"Received shape: {result['received_shape']}, "
        f"Received sum: {result['received_sum']}"
    )


@ray.remote(num_gpus=1)
def trainer_broadcast_sparse_tensor(
    master_address: str,
    master_port: int,
    world_size: int,
) -> bool:
    """Trainer task that broadcasts sparse patches via the trainer engine.

    The worker task drives its own init/receive directly (it is not an RPC
    endpoint), so the engine gets a no-op control-plane client; the NCCL
    rendezvous and the patch broadcasts are the real thing.
    """
    import torch

    device = _set_ray_assigned_device()

    from vllm.distributed.weight_transfer import WeightTransferTrainerFactory
    from vllm.distributed.weight_transfer.sparse_nccl_engine import (
        SparseNCCLTrainerInitInfo,
        SparseWeightPatch,
    )

    class NoopClient:
        def init_weight_transfer_engine(self, init_info):
            pass

        def start_weight_update(self):
            pass

        def update_weights(self, update_info):
            pass

        def finish_weight_update(self):
            pass

    patch = SparseWeightPatch(
        name="test.weight",
        indices=torch.tensor([1, 7, 25], dtype=torch.int32, device=device),
        values=torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32, device=device),
        full_shape=(10, 10),
    )
    engine = WeightTransferTrainerFactory.trainer_init(
        init_info=SparseNCCLTrainerInitInfo(
            master_address=master_address,
            master_port=master_port,
            world_size=world_size,
            rank=0,
        ),
        client=NoopClient(),
    )
    engine.send_weights([patch])
    torch.accelerator.synchronize()
    engine.shutdown()
    return True


@ray.remote(num_gpus=1)
def inference_receive_sparse_tensor(
    master_address: str,
    master_port: int,
    world_size: int,
) -> dict:
    """Inference task that receives sparse patches via the sparse engine."""
    from unittest.mock import MagicMock

    import torch

    device = _set_ray_assigned_device()

    from vllm.config.parallel import ParallelConfig
    from vllm.config.weight_transfer import WeightTransferConfig
    from vllm.distributed.weight_transfer.sparse_nccl_engine import (
        SparseNCCLWeightTransferEngine,
        SparseNCCLWeightTransferUpdateInfo,
    )

    config = WeightTransferConfig(backend="sparse_nccl")
    vllm_config = MagicMock()
    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.rank = 0
    parallel_config.world_size = 1
    parallel_config.data_parallel_rank = 0
    parallel_config.data_parallel_index = 0
    vllm_config.parallel_config = parallel_config
    vllm_config.model_config = MagicMock()

    # Real module holding the target parameter the patch will modify.
    model = torch.nn.Module()
    model.register_parameter(
        "w", torch.nn.Parameter(torch.zeros(30, device="cuda"), requires_grad=False)
    )
    model.get_parameter = lambda name: model.w

    update_info = SparseNCCLWeightTransferUpdateInfo(
        names=["w"],
        dtype_names=["float32"],
        shapes=[[30]],
        num_updates_list=[3],
    )

    engine = SparseNCCLWeightTransferEngine(
        config, vllm_config, torch.device("cuda"), model
    )
    from vllm.distributed.weight_transfer.nccl_common import (
        NCCLWeightTransferInitInfo,
    )

    engine.init_transfer_engine(
        NCCLWeightTransferInitInfo(
            master_address=master_address,
            master_port=master_port,
            rank_offset=1,
            world_size=world_size,
        )
    )
    engine.receive_weights(update_info)
    torch.accelerator.synchronize()

    expected = torch.zeros(30, dtype=torch.float32, device=device)
    expected[[1, 7, 25]] = torch.tensor(
        [10.0, 20.0, 30.0], dtype=torch.float32, device=device
    )
    success = torch.equal(model.w.data, expected)
    engine.shutdown()
    return {
        "success": success,
        "selected_values": model.w.data[[1, 7, 25]].cpu().tolist(),
    }


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run NCCL sparse weight transfer test.",
)
def test_nccl_sparse_weight_transfer_between_processes():
    """Test NCCL sparse weight transfer from trainer to inference process."""
    _init_ray_for_weight_transfer()

    master_address = "127.0.0.1"
    master_port = get_open_port()
    world_size = 2

    inference_future = inference_receive_sparse_tensor.remote(
        master_address, master_port, world_size
    )
    trainer_future = trainer_broadcast_sparse_tensor.remote(
        master_address, master_port, world_size
    )

    trainer_result, result = ray.get([trainer_future, inference_future])

    assert trainer_result, "Trainer should complete successfully"
    assert result["success"], (
        "Sparse weight transfer failed. "
        f"Received selected values: {result['selected_values']}"
    )


# --- Unit Tests: IPCWeightTransferUpdateInfo Validation ---


class TestIPCWeightTransferUpdateInfoValidation:
    """Test IPCWeightTransferUpdateInfo dataclass validation."""

    def test_valid_update_info(self):
        if torch.accelerator.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        dummy_tensor = torch.ones(10, 10, device="cuda:0")
        _, ipc_handle = reduce_tensor(dummy_tensor)
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        ipc_handles = [{gpu_uuid: ipc_handle}]

        info = IPCWeightTransferUpdateInfo(
            names=["layer.weight"],
            dtype_names=["float32"],
            shapes=[[10, 10]],
            ipc_handles=ipc_handles,
        )
        assert info.names == ["layer.weight"]
        assert info.dtype_names == ["float32"]
        assert info.shapes == [[10, 10]]
        assert len(info.ipc_handles) == 1

    def test_mismatched_dtype_names_raises(self):
        if torch.accelerator.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        dummy_tensor = torch.ones(10, 10, device="cuda:0")
        _, ipc_handle = reduce_tensor(dummy_tensor)
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        ipc_handles = [{gpu_uuid: ipc_handle}, {gpu_uuid: ipc_handle}]

        with pytest.raises(ValueError, match="dtype_names"):
            IPCWeightTransferUpdateInfo(
                names=["layer.weight", "layer.bias"],
                dtype_names=["float32"],  # Only one dtype
                shapes=[[10, 10], [10]],
                ipc_handles=ipc_handles,
            )

    def test_mismatched_shapes_raises(self):
        if torch.accelerator.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        dummy_tensor = torch.ones(10, 10, device="cuda:0")
        _, ipc_handle = reduce_tensor(dummy_tensor)
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        ipc_handles = [{gpu_uuid: ipc_handle}, {gpu_uuid: ipc_handle}]

        with pytest.raises(ValueError, match="shapes"):
            IPCWeightTransferUpdateInfo(
                names=["layer.weight", "layer.bias"],
                dtype_names=["float32", "float32"],
                shapes=[[10, 10]],  # Only one shape
                ipc_handles=ipc_handles,
            )

    def test_mismatched_ipc_handles_raises(self):
        if torch.accelerator.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        dummy_tensor = torch.ones(10, 10, device="cuda:0")
        _, ipc_handle = reduce_tensor(dummy_tensor)
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        ipc_handles = [{gpu_uuid: ipc_handle}]  # Only one handle

        with pytest.raises(ValueError, match="ipc_handles"):
            IPCWeightTransferUpdateInfo(
                names=["layer.weight", "layer.bias"],
                dtype_names=["float32", "float32"],
                shapes=[[10, 10], [10]],
                ipc_handles=ipc_handles,
            )

    def test_valid_update_info_from_pickled(self, monkeypatch):
        if torch.accelerator.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        dummy_tensor = torch.ones(10, 10, device="cuda:0")
        ipc_handle = reduce_tensor(dummy_tensor)
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        ipc_handles = [{gpu_uuid: ipc_handle}]

        pickled = base64.b64encode(pickle.dumps(ipc_handles)).decode("utf-8")

        info = IPCWeightTransferUpdateInfo(
            names=["layer.weight"],
            dtype_names=["float32"],
            shapes=[[10, 10]],
            ipc_handles_pickled=pickled,
        )
        assert info.ipc_handles == ipc_handles
        assert info.ipc_handles_pickled is None

    def test_pickled_requires_insecure_serialization_flag(self, monkeypatch):
        monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0")

        with pytest.raises(ValueError, match="VLLM_ALLOW_INSECURE_SERIALIZATION=1"):
            IPCWeightTransferUpdateInfo(
                names=[],
                dtype_names=[],
                shapes=[],
                ipc_handles_pickled=base64.b64encode(pickle.dumps([])).decode("utf-8"),
            )

    def test_both_handles_and_pickled_raises(self):
        if torch.accelerator.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        dummy_tensor = torch.ones(10, 10, device="cuda:0")
        ipc_handle = reduce_tensor(dummy_tensor)
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        ipc_handles = [{gpu_uuid: ipc_handle}]

        pickled = base64.b64encode(pickle.dumps(ipc_handles)).decode("utf-8")

        with pytest.raises(ValueError, match="Cannot specify both"):
            IPCWeightTransferUpdateInfo(
                names=["layer.weight"],
                dtype_names=["float32"],
                shapes=[[10, 10]],
                ipc_handles=ipc_handles,
                ipc_handles_pickled=pickled,
            )

    def test_neither_handles_nor_pickled_raises(self):
        with pytest.raises(ValueError, match="must be provided"):
            IPCWeightTransferUpdateInfo(
                names=["layer.weight"],
                dtype_names=["float32"],
                shapes=[[10, 10]],
            )

    def test_empty_lists_valid(self):
        info = IPCWeightTransferUpdateInfo(
            names=[],
            dtype_names=[],
            shapes=[],
            ipc_handles=[],
        )
        assert len(info.names) == 0


# --- Unit Tests: IPC Engine Parsing ---


class TestIPCEngineParsing:
    """Test IPCWeightTransferEngine parsing methods."""

    def _make_engine(self):
        config = WeightTransferConfig(backend="ipc")
        return IPCWeightTransferEngine(
            config,
            create_mock_vllm_config(),
            torch.device("cuda"),
            MagicMock(spec=torch.nn.Module),
        )

    def test_parse_update_info_valid(self):
        if torch.accelerator.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        engine = self._make_engine()

        dummy_tensor1 = torch.ones(100, 100, device="cuda:0")
        dummy_tensor2 = torch.ones(50, device="cuda:0")
        _, ipc_args1 = reduce_tensor(dummy_tensor1)
        _, ipc_args2 = reduce_tensor(dummy_tensor2)
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        ipc_handles = [{gpu_uuid: ipc_args1}, {gpu_uuid: ipc_args2}]

        update_info = engine.parse_update_info(
            {
                "names": ["w1", "w2"],
                "dtype_names": ["float32", "bfloat16"],
                "shapes": [[100, 100], [50]],
                "ipc_handles": ipc_handles,
            }
        )

        assert isinstance(update_info, IPCWeightTransferUpdateInfo)
        assert update_info.names == ["w1", "w2"]
        assert update_info.dtype_names == ["float32", "bfloat16"]
        assert update_info.shapes == [[100, 100], [50]]
        assert len(update_info.ipc_handles) == 2

    def test_parse_update_info_pickled(self, monkeypatch):
        if torch.accelerator.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

        engine = self._make_engine()

        dummy_tensor1 = torch.ones(100, 100, device="cuda:0")
        dummy_tensor2 = torch.ones(50, device="cuda:0")
        _, ipc_args1 = reduce_tensor(dummy_tensor1)
        _, ipc_args2 = reduce_tensor(dummy_tensor2)
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        ipc_handles = [{gpu_uuid: ipc_args1}, {gpu_uuid: ipc_args2}]

        pickled = base64.b64encode(pickle.dumps(ipc_handles)).decode("utf-8")

        update_info = engine.parse_update_info(
            {
                "names": ["w1", "w2"],
                "dtype_names": ["float32", "bfloat16"],
                "shapes": [[100, 100], [50]],
                "ipc_handles_pickled": pickled,
            }
        )

        assert isinstance(update_info, IPCWeightTransferUpdateInfo)
        assert update_info.names == ["w1", "w2"]
        assert len(update_info.ipc_handles) == 2
        assert gpu_uuid in update_info.ipc_handles[0]
        assert gpu_uuid in update_info.ipc_handles[1]

    def test_parse_update_info_ignores_none_pickled_handles(self):
        engine = self._make_engine()
        ipc_handles = [{"gpu-uuid": ("ipc-args",)}]

        update_info = engine.parse_update_info(
            {
                "names": ["w1"],
                "dtype_names": ["float32"],
                "shapes": [[1]],
                "ipc_handles": ipc_handles,
                "ipc_handles_pickled": None,
            }
        )

        assert isinstance(update_info, IPCWeightTransferUpdateInfo)
        assert update_info.ipc_handles == ipc_handles

    def test_parse_update_info_both_handles_and_pickled_raises(self):
        if torch.accelerator.device_count() < 1:
            pytest.skip("Need at least 1 GPU for this test")

        engine = self._make_engine()

        dummy_tensor = torch.ones(10, 10, device="cuda:0")
        _, ipc_handle = reduce_tensor(dummy_tensor)
        gpu_uuid = str(torch.cuda.get_device_properties(0).uuid)
        ipc_handles = [{gpu_uuid: ipc_handle}]

        pickled = base64.b64encode(pickle.dumps(ipc_handles)).decode("utf-8")

        with pytest.raises(ValueError, match="Cannot specify both"):
            engine.parse_update_info(
                {
                    "names": ["layer.weight"],
                    "dtype_names": ["float32"],
                    "shapes": [[10, 10]],
                    "ipc_handles": ipc_handles,
                    "ipc_handles_pickled": pickled,
                }
            )


# --- Integration Test: IPC Weight Transfer Between Ray Tasks ---


def get_physical_gpu_id(device_index: int = 0) -> str:
    """Get physical GPU UUID for a device."""
    props = torch.cuda.get_device_properties(device_index)
    return str(props.uuid)


@ray.remote(num_gpus=0.5)
class TrainerActor:
    """Trainer actor that creates and holds CUDA IPC handles."""

    def __init__(self, tensor_shape: list[int], tensor_dtype: str):
        device = _set_ray_assigned_device()

        # Create tensor on GPU and keep it alive
        dtype = getattr(torch, tensor_dtype)
        self.tensor = torch.ones(tensor_shape, dtype=dtype, device=device)
        self.tensor.fill_(42.0)  # Fill with 42 to verify correct transfer

        _, ipc_args = reduce_tensor(self.tensor)
        gpu_uuid = get_physical_gpu_id(device.index)

        torch.accelerator.synchronize()

        self.ipc_handle_dict = {
            "ipc_handle": ipc_args,
            "gpu_uuid": gpu_uuid,
            "shape": tensor_shape,
            "dtype": tensor_dtype,
        }

    def get_ipc_handle_dict(self) -> dict:
        """Return IPC handle dict. Tensor stays alive in this actor."""
        return self.ipc_handle_dict


@ray.remote(num_gpus=0.5)
def inference_receive_ipc_tensor(
    ipc_handle_dict: dict,
    mode: str = "ray",
) -> dict:
    """Inference task that receives tensor via IPCWeightTransferEngine."""
    import contextlib
    import os

    # Worker-side: ipc_handles_pickled is deserialized via pickle.
    if mode == "http":
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

    from unittest.mock import MagicMock

    import torch

    device = _set_ray_assigned_device()

    from vllm.config.parallel import ParallelConfig
    from vllm.config.weight_transfer import WeightTransferConfig
    from vllm.distributed.weight_transfer.ipc_engine import (
        IPCWeightTransferEngine,
    )

    class Recorder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.received = []

        def load_weights(self, weights):
            for name, tensor in weights:
                self.received.append((name, tensor.clone()))

    # Trainer sends unpacked IPC handles; the worker learns packed=False from
    # the init handshake below (IPCWeightTransferInitInfo defaults to False).
    config = WeightTransferConfig(backend="ipc")
    vllm_config = MagicMock()
    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.rank = 0
    parallel_config.world_size = 1
    parallel_config.data_parallel_rank = 0
    parallel_config.data_parallel_index = 0
    vllm_config.parallel_config = parallel_config
    vllm_config.model_config = MagicMock()

    recorder = Recorder()
    engine = IPCWeightTransferEngine(config, vllm_config, device, recorder)
    # Transport-only test: bypass the set_current_vllm_config context that
    # receive_weights enters, since vllm_config here is a mock.
    import vllm.config as _vllm_config_mod

    _vllm_config_mod.set_current_vllm_config = lambda cfg: contextlib.nullcontext()

    init_info = IPCWeightTransferInitInfo()
    engine.init_transfer_engine(init_info)

    ipc_handles = [{ipc_handle_dict["gpu_uuid"]: ipc_handle_dict["ipc_handle"]}]

    if mode == "ray":
        update_dict: dict = {
            "names": ["test.weight"],
            "dtype_names": [ipc_handle_dict["dtype"]],
            "shapes": [ipc_handle_dict["shape"]],
            "ipc_handles": ipc_handles,
        }
    elif mode == "http":
        pickled = base64.b64encode(pickle.dumps(ipc_handles)).decode("utf-8")
        update_dict = {
            "names": ["test.weight"],
            "dtype_names": [ipc_handle_dict["dtype"]],
            "shapes": [ipc_handle_dict["shape"]],
            "ipc_handles_pickled": pickled,
        }
    else:
        raise ValueError(f"Unknown mode: {mode}")

    update_info = engine.parse_update_info(update_dict)
    engine.receive_weights(update_info)
    torch.accelerator.synchronize()

    success = False
    received_shape = None
    received_sum = None

    if len(recorder.received) == 1:
        name, tensor = recorder.received[0]
        received_shape = list(tensor.shape)
        received_sum = tensor.sum().item()
        if received_shape == ipc_handle_dict["shape"]:
            expected_sum = 42.0 * torch.tensor(ipc_handle_dict["shape"]).prod().item()
            if abs(received_sum - expected_sum) < 0.01:
                success = True

    engine.shutdown()

    return {
        "success": success,
        "received_shape": received_shape,
        "received_sum": received_sum,
    }


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1,
    reason="Need at least 1 GPU to run IPC weight transfer test.",
)
@pytest.mark.parametrize("mode", ["ray", "http"])
def test_ipc_weight_transfer_between_processes(mode: str):
    """Test IPC weight transfer from trainer to inference process using Ray."""
    from ray.util.placement_group import placement_group
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    _init_ray_for_weight_transfer()

    pg = placement_group([{"GPU": 1, "CPU": 2}])
    ray.get(pg.ready())

    scheduling_strategy = PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_capture_child_tasks=True,
    )

    tensor_shape = [100, 100]
    tensor_dtype = "float32"

    trainer_actor = TrainerActor.options(  # type: ignore[attr-defined]
        scheduling_strategy=scheduling_strategy
    ).remote(tensor_shape, tensor_dtype)

    ipc_handle_dict = ray.get(trainer_actor.get_ipc_handle_dict.remote())

    inference_result = ray.get(
        inference_receive_ipc_tensor.options(
            scheduling_strategy=scheduling_strategy
        ).remote(ipc_handle_dict, mode=mode)
    )

    assert inference_result["success"], (
        f"IPC weight transfer failed (mode={mode}). "
        f"Received shape: {inference_result['received_shape']}, "
        f"Received sum: {inference_result['received_sum']}"
    )


def test_ipc_receive_weights_missing_gpu_uuid_raises():
    """Test that receive_weights raises if GPU UUID not found in IPC handles."""
    if torch.accelerator.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    config = WeightTransferConfig(backend="ipc")
    engine = IPCWeightTransferEngine(
        config,
        create_mock_vllm_config(),
        torch.device("cuda:0"),
        MagicMock(spec=torch.nn.Module),
    )
    # No init handshake here, so the engine keeps its default packed=False.

    dummy_tensor = torch.ones(10, 10, device="cuda:0")
    _, ipc_handle = reduce_tensor(dummy_tensor)
    wrong_uuid = "wrong-uuid-12345"
    ipc_handles = [{wrong_uuid: ipc_handle}]

    update_info = IPCWeightTransferUpdateInfo(
        names=["w"],
        dtype_names=["float32"],
        shapes=[[10, 10]],
        ipc_handles=ipc_handles,
    )

    with pytest.raises(ValueError, match="IPC handle not found"):
        engine.receive_weights(update_info)


class RecordingClient:
    """A fake VLLMWeightSyncClient that records the order of calls."""

    def __init__(self):
        self.order: list[str] = []
        self.last_init_info: dict | None = None
        self.last_update_info: dict | None = None

    def init_weight_transfer_engine(self, init_info: dict) -> None:
        self.order.append("init")
        self.last_init_info = init_info

    def start_weight_update(self) -> None:
        self.order.append("start")

    def update_weights(self, update_info: dict) -> None:
        self.order.append("update")
        self.last_update_info = update_info

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        self.order.append("finish")


def _module_with(*pairs):
    """A tiny nn.Module exposing the given (name, tensor) pairs as parameters,
    so trainer tests can build a ModuleSource without a real model."""
    module = torch.nn.Module()
    for name, tensor in pairs:
        module.register_parameter(name, torch.nn.Parameter(tensor, requires_grad=False))
    return module


class _DummyTrainerEngine(TrainerWeightTransferEngine):
    """Minimal concrete trainer engine to exercise base-class + factory."""

    @classmethod
    def trainer_init(cls, init_info, *, client, source):
        return cls(client=client, source=source)

    def send_weights(self):
        pass


class TestTrainerClients:
    """Structural protocol conformance for the built-in clients."""

    def test_recording_client_is_protocol(self):
        assert isinstance(RecordingClient(), VLLMWeightSyncClient)

    def test_http_client_is_protocol(self):
        assert isinstance(
            HTTPVLLMWeightSyncClient("http://localhost:8000"), VLLMWeightSyncClient
        )

    def test_ray_client_is_protocol(self):
        assert isinstance(RayVLLMWeightSyncClient(MagicMock()), VLLMWeightSyncClient)

    def test_ray_client_sends_typed_requests(self, monkeypatch):
        """Ray client must hand the actor typed Request objects, not raw dicts."""
        import ray

        monkeypatch.setattr(ray, "get", lambda refs: None)
        handle = MagicMock()
        client = RayVLLMWeightSyncClient(handle)

        client.init_weight_transfer_engine({"master_addr": "x"})
        (init_req,), _ = handle.init_weight_transfer_engine.remote.call_args
        assert isinstance(init_req, WeightTransferInitRequest)
        assert init_req.init_info == {"master_addr": "x"}

        client.update_weights({"names": ["w"]})
        (update_req,), _ = handle.update_weights.remote.call_args
        assert isinstance(update_req, WeightTransferUpdateRequest)
        assert update_req.update_info == {"names": ["w"]}

        client.finish_weight_update("step-42")
        handle.finish_weight_update.remote.assert_called_once_with()
        handle.update_weight_version.remote.assert_called_once_with("step-42")

    def test_http_client_pickles_ipc_handles_for_json(self, monkeypatch):
        """HTTP update_weights must encode raw ipc_handles as a base64 pickle."""
        captured = {}

        def fake_post(self, path, json=None):
            captured["path"] = path
            captured["json"] = json

        monkeypatch.setattr(HTTPVLLMWeightSyncClient, "_post", fake_post)
        client = HTTPVLLMWeightSyncClient("http://localhost:8000")
        client.update_weights({"names": ["w"], "ipc_handles": [{"gpu": ("args",)}]})
        sent = captured["json"]["update_info"]
        assert "ipc_handles" not in sent
        assert "ipc_handles_pickled" in sent
        assert pickle.loads(base64.b64decode(sent["ipc_handles_pickled"])) == [
            {"gpu": ("args",)}
        ]

    def test_http_client_passes_through_nccl_update_info(self, monkeypatch):
        """NCCL update_info has only JSON-native fields and passes unchanged."""
        captured = {}

        def fake_post(self, path, json=None):
            captured["json"] = json

        monkeypatch.setattr(HTTPVLLMWeightSyncClient, "_post", fake_post)
        client = HTTPVLLMWeightSyncClient("http://localhost:8000")
        update_info = {"names": ["w"], "dtype_names": ["float32"], "shapes": [[4]]}
        client.update_weights(update_info)
        assert captured["json"]["update_info"] == update_info

        client.finish_weight_update("step-42")
        assert captured["json"] == {"weight_version": "step-42"}


class TestModuleSource:
    """`ModuleSource` metadata vs. materialized iteration (dense, no GPU)."""

    def test_metadata_reads_shape_and_dtype(self):
        source = ModuleSource(
            _module_with(("w", torch.zeros(2, 3)), ("b", torch.zeros(3)))
        )
        meta = source.metadata()
        assert [m.name for m in meta] == ["w", "b"]
        assert [m.shape for m in meta] == [(2, 3), (3,)]
        assert all(m.dtype == torch.float32 for m in meta)

    def test_iteration_yields_materialized_tensors(self):
        w = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        source = ModuleSource(_module_with(("w", w)))
        pairs = list(source)
        assert [name for name, _ in pairs] == ["w"]
        assert torch.equal(pairs[0][1], w)

    def test_source_is_reiterable(self):
        source = ModuleSource(_module_with(("w", torch.zeros(2))))
        assert [n for n, _ in source] == [n for n, _ in source] == ["w"]

    def test_metadata_agrees_with_iteration(self):
        """The two channels must line up element-for-element: engines declare
        the round from `metadata()` and then send what iteration yields."""
        source = ModuleSource(
            _module_with(("w", torch.zeros(2, 3)), ("b", torch.zeros(3)))
        )
        meta = source.metadata()
        pairs = list(source)
        assert [m.name for m in meta] == [name for name, _ in pairs]
        assert [m.dtype for m in meta] == [t.dtype for _, t in pairs]
        assert [m.shape for m in meta] == [tuple(t.shape) for _, t in pairs]


class TestTrainerFactory:
    """WeightTransferTrainerFactory registry mechanics."""

    def test_registry_has_all_backends(self):
        assert "nccl" in WeightTransferTrainerFactory._registry
        assert "ipc" in WeightTransferTrainerFactory._registry
        assert "sparse_nccl" in WeightTransferTrainerFactory._registry

    def test_register_and_dispatch(self):
        saved = dict(WeightTransferTrainerFactory._registry)
        try:
            WeightTransferTrainerFactory.register_engine("dummy", _DummyTrainerEngine)
            engine = WeightTransferTrainerFactory.trainer_init(
                MagicMock(backend="dummy"),  # backend read from the init info
                client=RecordingClient(),
                source=ModuleSource(_module_with(("w", torch.zeros(2)))),
            )
            assert isinstance(engine, _DummyTrainerEngine)
            with pytest.raises(ValueError, match="already registered"):
                WeightTransferTrainerFactory.register_engine(
                    "dummy", _DummyTrainerEngine
                )
        finally:
            WeightTransferTrainerFactory._registry = saved

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Invalid weight transfer backend"):
            WeightTransferTrainerFactory.trainer_init(
                MagicMock(backend="nope"),
                client=RecordingClient(),
                source=ModuleSource(_module_with(("w", torch.zeros(2)))),
            )

    def test_ipc_init_info_declares_backend(self):
        assert IPCTrainerInitInfo.backend == "ipc"

    def test_nccl_init_info_declares_backend(self):
        assert NCCLTrainerInitInfo.backend == "nccl"

    def test_sparse_nccl_init_info_declares_backend(self):
        assert SparseNCCLTrainerInitInfo.backend == "sparse_nccl"

    def test_trainer_init_info_subclass_must_set_backend(self):
        with pytest.raises(TypeError, match="class-level `backend`"):

            class _NoBackend(TrainerInitInfo):
                pass


class TestTrainerEngineBase:
    """Base-class construction (no GPU)."""

    def test_source_stored_and_sender_by_default(self):
        engine = _DummyTrainerEngine(
            client=RecordingClient(),
            source=ModuleSource(_module_with(("w", torch.zeros(2)))),
        )
        assert engine.is_sender is True
        assert [name for name, _ in engine.source] == ["w"]

    def test_shutdown_default_is_noop(self):
        engine = _DummyTrainerEngine(
            client=RecordingClient(),
            source=ModuleSource(_module_with(("w", torch.zeros(2)))),
            is_sender=False,
        )
        assert engine.is_sender is False
        engine.shutdown()  # must not raise


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1,
    reason="Need at least 1 GPU (CUDA IPC handles).",
)
def test_ipc_trainer_send_weights_drives_client_in_order():
    """send_weights issues start -> update -> finish and ships per-round metadata;
    the packed wire param rides the init info, not the per-round update_info."""
    client = RecordingClient()
    engine = IPCTrainerWeightTransferEngine(
        client=client,
        source=ModuleSource(_module_with(("w", torch.ones(4, device="cuda")))),
        packed=False,
    )

    engine.send_weights()

    assert client.order == ["start", "update", "finish"]
    assert client.last_update_info is not None
    assert client.last_update_info["names"] == ["w"]
    assert client.last_update_info["shapes"] == [[4]]
    assert "packed" not in client.last_update_info


def test_ipc_trainer_init_ships_packed_to_worker():
    """trainer_init drives the inference-side init handshake and propagates the
    must-agree `packed` flag to the worker."""
    if torch.accelerator.device_count() < 1:
        pytest.skip("Need at least 1 GPU (CUDA IPC handles).")

    client = RecordingClient()
    engine = WeightTransferTrainerFactory.trainer_init(
        init_info=IPCTrainerInitInfo(rank=0, packed=True),  # backend from init info
        client=client,
        source=ModuleSource(_module_with(("w", torch.ones(4, device="cuda")))),
    )

    assert isinstance(engine, IPCTrainerWeightTransferEngine)
    assert engine.is_sender is True
    assert engine.packed is True
    assert client.order == ["init"]
    assert client.last_init_info == {"packed": True}


def test_nccl_trainer_init_ships_worker_init_info(monkeypatch):
    """The sender's trainer_init drives the inference-side init handshake with
    the worker-shaped init info (rank_offset=1) while opening its own endpoint,
    and propagates the must-agree wire params to the worker."""
    import vllm.distributed.weight_transfer.nccl_engine as nccl_engine_mod

    # Bypass the real NCCL rendezvous.
    monkeypatch.setattr(
        nccl_engine_mod, "open_trainer_endpoint", lambda info: MagicMock()
    )

    client = RecordingClient()
    engine = WeightTransferTrainerFactory.trainer_init(
        init_info=NCCLTrainerInitInfo(
            master_address="127.0.0.1",
            master_port=29500,
            world_size=3,
            rank=0,
            packed=True,
            packed_buffer_size_bytes=1024,
            packed_num_buffers=3,
        ),
        client=client,
        source=ModuleSource(_module_with(("w", torch.zeros(4)))),
    )

    assert isinstance(engine, NCCLTrainerWeightTransferEngine)
    assert engine.is_sender is True
    assert engine.packed is True
    assert client.order == ["init"]
    assert client.last_init_info == {
        "master_address": "127.0.0.1",
        "master_port": 29500,
        "rank_offset": 1,
        "world_size": 3,
        "packed": True,
        "packed_buffer_size_bytes": 1024,
        "packed_num_buffers": 3,
    }


def test_nccl_worker_learns_wire_params_from_init_handshake(monkeypatch):
    """The worker engine reads packed + buffer geometry from the
    trainer-supplied init info at the handshake, not from the config or the
    per-round update info."""
    import vllm.distributed.weight_transfer.nccl_engine as nccl_engine_mod

    monkeypatch.setattr(
        nccl_engine_mod, "worker_init_process_group", lambda info, pc: MagicMock()
    )

    engine = NCCLWeightTransferEngine(
        WeightTransferConfig(backend="nccl"),
        create_mock_vllm_config(),
        torch.device("cuda:0"),
        MagicMock(spec=torch.nn.Module),
    )
    assert engine.packed is False  # pre-handshake default (legacy unpacked)
    engine.init_transfer_engine(
        NCCLWeightTransferInitInfo(
            master_address="127.0.0.1",
            master_port=29500,
            rank_offset=1,
            world_size=2,
            packed=True,
            packed_buffer_size_bytes=2048,
            packed_num_buffers=4,
        )
    )

    assert engine.packed is True
    assert engine.packed_buffer_size_bytes == 2048
    assert engine.packed_num_buffers == 4


def test_nccl_trainer_init_non_sender_skips_rendezvous_and_client():
    """Non-sender trainer ranks build an engine without opening an endpoint or
    touching the client; they only join the collectives in send_weights."""
    client = RecordingClient()
    engine = WeightTransferTrainerFactory.trainer_init(
        init_info=NCCLTrainerInitInfo(
            master_address="127.0.0.1",
            master_port=29500,
            world_size=3,
            rank=1,
        ),
        client=client,
        source=ModuleSource(_module_with(("w", torch.zeros(4)))),
    )

    assert engine.is_sender is False
    assert engine.model_update_group is None
    assert client.order == []

    # send_weights on a non-sender only iterates the source (packed mode needs
    # no CUDA stream on non-senders), never the client.
    engine.send_weights()
    assert client.order == []


@pytest.mark.skipif(
    torch.accelerator.device_count() < 1,
    reason="Need at least 1 GPU (NCCL broadcast / CUDA stream).",
)
def test_nccl_trainer_send_weights_drives_client_in_order():
    """send_weights issues start -> update -> finish and ships per-round
    metadata; the packed wire params ride the init handshake, not the
    per-round update_info."""
    client = RecordingClient()
    engine = NCCLTrainerWeightTransferEngine(
        client=client,
        source=ModuleSource(_module_with(("w", torch.zeros(4, device="cuda")))),
        packed=False,
    )
    # Bypass the real NCCL rendezvous; broadcast is a no-op.
    engine.model_update_group = MagicMock()

    engine.send_weights()

    assert client.order == ["start", "update", "finish"]
    assert client.last_update_info is not None
    assert client.last_update_info["names"] == ["w"]
    assert client.last_update_info["shapes"] == [[4]]
    assert "packed" not in client.last_update_info


class _ScriptedSource(WeightSource):
    """Declares `meta` but yields whatever `pairs` says — used to drive the
    metadata/iteration agreement checks."""

    def __init__(self, meta, pairs):
        self._meta = meta
        self._pairs = pairs

    def metadata(self):
        return list(self._meta)

    def __iter__(self):
        yield from self._pairs


def _mock_group_engine(source, monkeypatch, **kwargs):
    """Unpacked trainer engine with a mocked group and stream (no GPU needed)."""
    engine = NCCLTrainerWeightTransferEngine(
        client=RecordingClient(), source=source, packed=False, **kwargs
    )
    engine.model_update_group = MagicMock()
    monkeypatch.setattr(torch.cuda, "current_stream", MagicMock())
    return engine


def test_nccl_trainer_init_requires_source():
    """NCCL is a full-resync backend: it cannot run without a WeightSource."""
    with pytest.raises(ValueError, match="requires a WeightSource"):
        NCCLTrainerWeightTransferEngine.trainer_init(
            NCCLTrainerInitInfo(
                master_address="127.0.0.1", master_port=29500, world_size=2, rank=0
            ),
            client=RecordingClient(),
        )


def test_nccl_trainer_send_weights_rejects_reordered_source(monkeypatch):
    """The worker sizes its buffers (and cuts packed chunks) from metadata(), so
    iteration disagreeing with it must raise rather than corrupt the stream."""
    meta = [
        ParamMeta("w", torch.float32, (4,)),
        ParamMeta("b", torch.float32, (2,)),
    ]
    reordered = [("b", torch.zeros(2)), ("w", torch.zeros(4))]
    engine = _mock_group_engine(_ScriptedSource(meta, reordered), monkeypatch)

    with pytest.raises(ValueError, match="disagrees with iteration at index 0"):
        engine.send_weights()


def test_nccl_trainer_send_weights_rejects_dtype_disagreement(monkeypatch):
    """A source that declares one wire dtype and materializes another would make
    the two sides disagree on every byte offset."""
    meta = [ParamMeta("w", torch.float32, (4,))]
    engine = _mock_group_engine(
        _ScriptedSource(meta, [("w", torch.zeros(4, dtype=torch.bfloat16))]),
        monkeypatch,
    )

    with pytest.raises(ValueError, match="disagrees with iteration"):
        engine.send_weights()


def test_nccl_trainer_send_weights_rejects_truncated_source(monkeypatch):
    """Yielding fewer parameters than declared leaves the worker waiting."""
    meta = [
        ParamMeta("w", torch.float32, (4,)),
        ParamMeta("b", torch.float32, (2,)),
    ]
    engine = _mock_group_engine(
        _ScriptedSource(meta, [("w", torch.zeros(4))]), monkeypatch
    )

    with pytest.raises(ValueError, match="yielded 1 parameters"):
        engine.send_weights()


def test_nccl_trainer_send_weights_broadcasts_contiguous(monkeypatch):
    """NCCL sends numel elements from data_ptr(), so a non-contiguous view must
    be linearized first or the worker receives unrelated memory."""
    base = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    view = base.t()  # non-contiguous
    meta = [ParamMeta("w", torch.float32, tuple(view.shape))]
    engine = _mock_group_engine(_ScriptedSource(meta, [("w", view)]), monkeypatch)

    engine.send_weights()

    sent = engine.model_update_group.broadcast.call_args.args[0]
    assert sent.is_contiguous()
    assert torch.equal(sent, view)


def test_nccl_trainer_send_weights_raises_instead_of_hanging(monkeypatch):
    """A failed broadcast must surface even while the inference-side
    update_weights is still blocked in its matching NCCL call.

    Joining the RPC thread there would deadlock: the worker only returns once
    the broadcast it is waiting for arrives, which never happens.
    """
    rpc_entered = threading.Event()
    release_rpc = threading.Event()

    class _BlockingClient(RecordingClient):
        def update_weights(self, update_info):
            rpc_entered.set()
            release_rpc.wait(timeout=30)  # stands in for a wedged NCCL recv
            super().update_weights(update_info)

    class _FailingSource(WeightSource):
        def metadata(self):
            return [ParamMeta("w", torch.float32, (4,))]

        def __iter__(self):
            # Fail only once the RPC is provably in flight.
            rpc_entered.wait(timeout=60)
            raise RuntimeError("broadcast blew up")

    engine = NCCLTrainerWeightTransferEngine(
        client=_BlockingClient(), source=_FailingSource(), packed=False
    )
    engine.model_update_group = MagicMock()
    monkeypatch.setattr(torch.cuda, "current_stream", MagicMock())

    started = time.perf_counter()
    try:
        with pytest.raises(RuntimeError, match="broadcast blew up"):
            engine.send_weights()
        elapsed = time.perf_counter() - started
        assert rpc_entered.is_set(), "the RPC was never in flight"
        assert not release_rpc.is_set(), "send_weights waited for the wedged RPC"
        # Regression guard: joining the RPC thread would park here until the
        # client's own timeout expires instead of raising immediately.
        assert elapsed < 5.0, f"send_weights blocked {elapsed:.1f}s on the RPC"
    finally:
        release_rpc.set()


def _sparse_patch(device: str = "cpu") -> SparseWeightPatch:
    return SparseWeightPatch(
        name="w",
        indices=torch.tensor([1, 3], dtype=torch.int32, device=device),
        values=torch.tensor([1.0, 2.0], dtype=torch.float32, device=device),
        full_shape=(4, 4),
    )


def test_sparse_nccl_trainer_init_ships_worker_init_info(monkeypatch):
    """The sender's trainer_init drives the init handshake with the
    worker-shaped init info; sparse ships no packed wire params, so the worker
    keeps its unpacked defaults. Sparse takes no `source`."""
    import vllm.distributed.weight_transfer.sparse_nccl_engine as sparse_mod

    monkeypatch.setattr(sparse_mod, "open_trainer_endpoint", lambda info: MagicMock())

    client = RecordingClient()
    engine = WeightTransferTrainerFactory.trainer_init(
        init_info=SparseNCCLTrainerInitInfo(
            master_address="127.0.0.1",
            master_port=29500,
            world_size=2,
            rank=0,
        ),
        client=client,
    )

    assert isinstance(engine, SparseNCCLTrainerWeightTransferEngine)
    assert client.order == ["init"]
    assert client.last_init_info == {
        "master_address": "127.0.0.1",
        "master_port": 29500,
        "rank_offset": 1,
        "world_size": 2,
        "packed": False,
        "packed_buffer_size_bytes": DEFAULT_PACKED_BUFFER_SIZE_BYTES,
        "packed_num_buffers": DEFAULT_PACKED_NUM_BUFFERS,
    }


def test_sparse_nccl_trainer_send_weights_drives_client_in_order(monkeypatch):
    """send_weights takes the round's patches and ships per-patch metadata
    (names / shapes / num_updates_list) + broadcasts indices + values each."""
    client = RecordingClient()
    engine = SparseNCCLTrainerWeightTransferEngine(client=client)
    engine.model_update_group = MagicMock()
    # The group is a mock, so the stream is just a handle it is handed (and a
    # handle _post_send_sync can synchronize).
    monkeypatch.setattr(torch.cuda, "current_stream", MagicMock())

    engine.send_weights([_sparse_patch()])

    assert client.order == ["start", "update", "finish"]
    assert client.last_update_info is not None
    assert client.last_update_info["names"] == ["w"]
    assert client.last_update_info["shapes"] == [[4, 4]]
    assert client.last_update_info["num_updates_list"] == [2]
    # One broadcast for indices + one for values per patch.
    assert engine.model_update_group.broadcast.call_count == 2


def test_sparse_nccl_trainer_send_weights_empty_round_is_noop():
    """A round with no patches must not touch the client (an empty sparse
    update info is invalid by construction)."""
    client = RecordingClient()
    engine = SparseNCCLTrainerWeightTransferEngine(client=client)
    engine.model_update_group = MagicMock()

    engine.send_weights([])
    engine.send_weights()  # no argument is also a no-op round

    assert client.order == []


def test_sparse_nccl_trainer_send_weights_requires_full_shape():
    patch = _sparse_patch()
    patch.full_shape = None
    engine = SparseNCCLTrainerWeightTransferEngine(client=RecordingClient())
    engine.model_update_group = MagicMock()

    with pytest.raises(ValueError, match="full_shape"):
        engine.send_weights([patch])


def test_sparse_nccl_trainer_rejects_source():
    """Sparse is a delta backend; a WeightSource would silently never be sent."""
    with pytest.raises(ValueError, match="takes no WeightSource"):
        SparseNCCLTrainerWeightTransferEngine(
            client=RecordingClient(),
            source=ModuleSource(_module_with(("w", torch.zeros(2)))),
        )


def test_sparse_nccl_trainer_validates_patch_before_any_rpc():
    """Malformed patches must fail on the trainer, before start_weight_update:
    the worker's own checks only run once the broadcasts are already in flight,
    where a size mismatch wedges both sides instead of raising."""
    client = RecordingClient()
    engine = SparseNCCLTrainerWeightTransferEngine(client=client)
    engine.model_update_group = MagicMock()

    mismatched = SparseWeightPatch(
        name="w",
        indices=torch.tensor([1, 3], dtype=torch.int32),
        values=torch.tensor([1.0], dtype=torch.float32),
        full_shape=(4, 4),
    )
    with pytest.raises(ValueError, match="matching lengths"):
        engine.send_weights([mismatched])

    wrong_index_dtype = SparseWeightPatch(
        name="w",
        indices=torch.tensor([1, 3], dtype=torch.int64),
        values=torch.tensor([1.0, 2.0], dtype=torch.float32),
        full_shape=(4, 4),
    )
    with pytest.raises(ValueError, match="int32 indices"):
        engine.send_weights([wrong_index_dtype])

    assert client.order == []


def test_sparse_nccl_trainer_non_sender_skips_client():
    client = RecordingClient()
    engine = WeightTransferTrainerFactory.trainer_init(
        init_info=SparseNCCLTrainerInitInfo(
            master_address="127.0.0.1",
            master_port=29500,
            world_size=2,
            rank=1,
        ),
        client=client,
    )

    assert engine.is_sender is False
    assert engine.model_update_group is None
    assert isinstance(engine, SparseNCCLTrainerWeightTransferEngine)
    engine.send_weights([_sparse_patch()])
    assert client.order == []


# --- NCCL M2N backend ---
#
# The real `nccl.m2n` runtime (nccl-extensions) is out of tree and not
# installed in CI. Tests that reach `init_transfer_engine` install a dumb
# stand-in that records handles and reshard calls and validates nothing, so
# it can never compensate for a missing vLLM-side check.


class _FakeM2NHandle:
    def __init__(self):
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


def _make_fake_m2n():
    m2n = types.ModuleType("nccl.m2n")
    m2n.handles = []
    m2n.reshard_calls = []

    class Mesh:
        def __init__(self, dims, start_rank=0):
            self.dims = tuple(dims)
            self.start_rank = start_rank

    class Shard:
        def __init__(self, dim):
            self.dim = dim

    class Replicate:
        pass

    class Config:
        def __init__(self, max_cta=None):
            self.max_cta = max_cta

    class Handle:
        @staticmethod
        def create(config):
            handle = _FakeM2NHandle()
            m2n.handles.append(handle)
            return handle

    def reshard(*args, **kwargs):
        m2n.reshard_calls.append((args, kwargs))

    m2n.Mesh = Mesh
    m2n.Shard = Shard
    m2n.Replicate = Replicate
    m2n.Config = Config
    m2n.Handle = Handle
    m2n.reshard = reshard
    return m2n


def _install_fake_m2n(monkeypatch):
    m2n = _make_fake_m2n()
    nccl_pkg = types.ModuleType("nccl")
    nccl_pkg.m2n = m2n
    monkeypatch.setitem(sys.modules, "nccl", nccl_pkg)
    monkeypatch.setitem(sys.modules, "nccl.m2n", m2n)
    return m2n


def _make_m2n_engine(model=None):
    vllm_config = create_mock_vllm_config()
    vllm_config.parallel_config.data_parallel_size = 1
    return M2NWeightTransferEngine(
        WeightTransferConfig(backend="nccl_m2n"),
        vllm_config,
        torch.device("cuda"),
        model if model is not None else MagicMock(spec=torch.nn.Module),
    )


def _m2n_init_info(**overrides):
    """A plan consistent with `_make_m2n_engine`'s single-worker deployment."""
    fields = dict(
        master_address="127.0.0.1",
        master_port=29500,
        rank_offset=1,
        world_size=2,
        src_mesh_dims=[1, 1],
        dst_mesh_dims=[1, 1],
        names=["w"],
        dtype_names=["float32"],
        shapes=[[4, 4]],
        src_placements=[None],
    )
    fields.update(overrides)
    return M2NWeightTransferInitInfo(**fields)


class TestM2NLayout:
    def test_replicated_splits_nothing(self):
        """A replicated tensor imposes no divisibility constraint. The shape
        dims are deliberately coprime with the mesh size: a layout that actually
        split the tensor would reject them."""
        mesh = M2NMesh((2, 2), start_rank=1)
        indivisible_shape = (7, 13)  # neither dim divisible by any mesh axis

        resolved_mesh, placements = resolve_layout(mesh, REPLICATED)
        validate_layout(resolved_mesh, placements, indivisible_shape, "destination")

    def test_replicated_keeps_the_same_ranks(self):
        """Replication re-factors the mesh to get a size-1 axis for its no-op
        shard. That is only sound if it still covers exactly the same GPUs."""
        mesh = M2NMesh((2, 3), start_rank=4)
        resolved_mesh, _ = resolve_layout(mesh, REPLICATED)
        assert resolved_mesh.size == mesh.size
        assert resolved_mesh.start_rank == mesh.start_rank

    def test_sharded_keeps_its_own_factorization(self):
        """Rank order decides who owns which shard, so a sharded tensor must
        not be re-factored the way a replicated one is."""
        mesh = M2NMesh((2, 3), start_rank=4)
        resolved_mesh, placements = resolve_layout(mesh, (REPLICATE, 0))
        assert resolved_mesh == mesh
        assert placements == (REPLICATE, 0)

    def test_two_shard_axes_rejected(self):
        """One axis has to replicate; a 2-D mesh that shards both is not
        something a single reshard can express."""
        with pytest.raises(ValueError, match="shards both"):
            check_placements((0, 1))

    def test_negative_placement_code_rejected(self):
        """Only REPLICATE (-1) may be negative. Fails if a code like -2 again
        passes validation — Python negative indexing made validate_layout
        check the wrong tensor dim — and flows into m2n.Shard(-2) instead of
        failing the init RPC."""
        with pytest.raises(ValueError, match="non-negative tensor dim"):
            check_placements((-2, REPLICATE))

    def test_shard_dim_must_exist(self):
        with pytest.raises(ValueError, match="rank 2"):
            validate_layout(M2NMesh((1, 2), 0), (REPLICATE, 2), (8, 16), "source")

    def test_shard_must_divide_evenly(self):
        with pytest.raises(ValueError, match="does not divide evenly"):
            validate_layout(M2NMesh((1, 3), 0), (REPLICATE, 0), (8, 16), "source")


class TestM2NTransferable:
    def test_unsupported_dtype_names_the_parameter(self):
        with pytest.raises(ValueError, match="'w'"):
            check_transferable("w", torch.complex64, (4,))

    def test_rank_four_rejected(self):
        with pytest.raises(ValueError, match="rank 4"):
            check_transferable("w", torch.bfloat16, (2, 2, 2, 2))


class TestM2NWireTypes:
    def _init_info(self, **overrides):
        fields = dict(
            master_address="127.0.0.1",
            master_port=1234,
            rank_offset=1,
            world_size=3,
            src_mesh_dims=[1, 1],
            dst_mesh_dims=[2, 1],
            names=["w"],
            dtype_names=["bfloat16"],
            shapes=[[16, 16]],
            src_placements=[None],
        )
        fields.update(overrides)
        return M2NWeightTransferInitInfo(**fields)

    def test_accepts_a_consistent_plan(self):
        assert self._init_info().names == ["w"]

    def test_ragged_plan_rejected(self):
        with pytest.raises(ValueError, match="`shapes`"):
            self._init_info(shapes=[])

    def test_destination_mesh_must_cover_the_workers(self):
        """The trainer declares the inference mesh, so one that does not cover
        the workers is a config error — and it has to fail the init RPC, since
        a mismatched mesh would otherwise surface as a hung collective."""
        with pytest.raises(ValueError, match="dst_mesh_dims"):
            self._init_info(dst_mesh_dims=[3, 1])  # 3 != the 2 workers

    def test_world_must_hold_a_trainer_and_a_worker(self):
        with pytest.raises(ValueError, match="rank_offset"):
            self._init_info(rank_offset=3, world_size=3)


def test_m2n_unknown_dtype_name_fails_init_with_value_error(monkeypatch):
    """An unknown dtype name must fail the init RPC with a ValueError naming
    the parameter; fails if it again leaks AttributeError from
    getattr(torch, name) with no pointer to the offending parameter."""
    _install_fake_m2n(monkeypatch)
    engine = _make_m2n_engine()

    with pytest.raises(ValueError, match="'w' has unknown dtype name 'bfloat61'"):
        engine.init_transfer_engine(_m2n_init_info(dtype_names=["bfloat61"]))


def test_m2n_worker_count_mismatch_fails_before_rendezvous(monkeypatch):
    """A declared worker count that disagrees with the deployment must fail
    the init RPC; fails if it again reaches the rendezvous, which waits
    forever for ranks that never join."""
    import vllm.distributed.weight_transfer.m2n_engine as m2n_engine_mod

    _install_fake_m2n(monkeypatch)
    rendezvous = MagicMock()
    monkeypatch.setattr(m2n_engine_mod, "worker_init_process_group", rendezvous)
    engine = _make_m2n_engine()  # this deployment has exactly 1 worker

    with pytest.raises(ValueError, match="wait forever"):
        engine.init_transfer_engine(_m2n_init_info(world_size=3, dst_mesh_dims=[2, 1]))
    rendezvous.assert_not_called()


def test_m2n_reinit_releases_previous_handle_and_communicator(monkeypatch):
    """Re-init (a trainer restart) must destroy the previous m2n handle — it
    holds the staging pool — and NCCL communicator; fails if either again
    leaks when init_transfer_engine overwrites them."""
    if torch.accelerator.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    import vllm.distributed.weight_transfer.m2n_engine as m2n_engine_mod

    m2n = _install_fake_m2n(monkeypatch)
    comms = [MagicMock(), MagicMock()]
    monkeypatch.setattr(
        m2n_engine_mod, "worker_init_process_group", lambda info, pc: comms.pop(0)
    )
    engine = _make_m2n_engine()

    engine.init_transfer_engine(_m2n_init_info())
    first_comm = engine.model_update_group
    engine.init_transfer_engine(_m2n_init_info())

    assert [handle.destroyed for handle in m2n.handles] == [True, False]
    first_comm.destroy.assert_called_once()
    engine.model_update_group.destroy.assert_not_called()


def test_m2n_undeclared_name_fails_round_before_any_reshard(monkeypatch):
    """A round naming an undeclared parameter must fail atomically; fails if
    the check again runs inside the loop, where earlier parameters are
    already resharded and loaded while the trainer stays blocked mid-round."""
    if torch.accelerator.device_count() < 1:
        pytest.skip("Need at least 1 GPU for this test")

    import vllm.distributed.weight_transfer.m2n_engine as m2n_engine_mod

    m2n = _install_fake_m2n(monkeypatch)
    comm = MagicMock()
    comm.comm = 1234  # comm_ptr resolves the raw handle before the name check
    monkeypatch.setattr(
        m2n_engine_mod, "worker_init_process_group", lambda info, pc: comm
    )

    loaded = []

    class Recorder(torch.nn.Module):
        def load_weights(self, weights):
            loaded.extend(name for name, _ in weights)

    engine = _make_m2n_engine(model=Recorder())
    engine.init_transfer_engine(_m2n_init_info())

    with pytest.raises(ValueError, match="not declared at init"):
        engine.receive_weights(M2NWeightTransferUpdateInfo(names=["w", "undeclared"]))
    assert m2n.reshard_calls == []
    assert loaded == []


# --- Integration Test: M2N Weight Transfer Between Ray Tasks ---


def _m2n_param_tensor(shape, offset, device):
    """Distinct deterministic contents per parameter, so a swapped or
    reordered delivery cannot bitwise-match."""
    numel = 1
    for dim in shape:
        numel *= dim
    return (torch.arange(numel, dtype=torch.float32, device=device) + offset).reshape(
        shape
    )


@ray.remote(num_gpus=1)
def m2n_trainer_broadcast_weights(
    master_address: str,
    master_port: int,
    params: list[tuple[str, list[int], float]],
) -> bool:
    """Trainer task: joins the shared communicator as rank 0 and plays the
    trainer half of the faked reshard — one broadcast per parameter, in
    declared order."""
    import torch

    device = _set_ray_assigned_device()

    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup

    pg = StatelessProcessGroup.create(
        host=master_address, port=master_port, rank=0, world_size=2
    )
    comm = PyNcclCommunicator(pg, device=device.index)

    stream = torch.cuda.current_stream()
    for _, shape, offset in params:
        comm.broadcast(_m2n_param_tensor(shape, offset, device), src=0, stream=stream)
    torch.accelerator.synchronize()
    return True


@ray.remote(num_gpus=1)
def m2n_worker_receive_weights(
    master_address: str,
    master_port: int,
    params: list[tuple[str, list[int], float]],
) -> dict:
    """Worker task: runs the real M2NWeightTransferEngine end to end, with
    only the out-of-tree reshard kernel replaced by a real NCCL broadcast on
    the engine's own live communicator."""
    import sys
    from unittest.mock import MagicMock

    import torch

    device = _set_ray_assigned_device()

    from vllm.config.parallel import ParallelConfig
    from vllm.config.weight_transfer import WeightTransferConfig
    from vllm.distributed.weight_transfer.m2n_engine import (
        M2NWeightTransferEngine,
        M2NWeightTransferInitInfo,
        M2NWeightTransferUpdateInfo,
    )

    live = {}  # filled after init; the fake reshard broadcasts on this comm
    m2n = _make_fake_m2n()

    def reshard(send, buffer, comm, stream, **kwargs):
        m2n.reshard_calls.append(buffer.shape)
        live["comm"].broadcast(buffer, src=0, stream=stream)

    m2n.reshard = reshard
    nccl_pkg = types.ModuleType("nccl")
    nccl_pkg.m2n = m2n
    sys.modules["nccl"] = nccl_pkg
    sys.modules["nccl.m2n"] = m2n

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            for name, shape, _ in params:
                self.register_parameter(
                    name,
                    torch.nn.Parameter(
                        torch.zeros(shape, device=device), requires_grad=False
                    ),
                )

        def load_weights(self, weights):
            for name, tensor in weights:
                self.get_parameter(name).data.copy_(tensor)

    vllm_config = MagicMock()
    parallel_config = MagicMock(spec=ParallelConfig)
    parallel_config.rank = 0
    parallel_config.world_size = 1
    parallel_config.data_parallel_rank = 0
    parallel_config.data_parallel_index = 0
    parallel_config.data_parallel_size = 1
    vllm_config.parallel_config = parallel_config
    vllm_config.model_config = MagicMock()

    model = TinyModel()
    engine = M2NWeightTransferEngine(
        WeightTransferConfig(backend="nccl_m2n"), vllm_config, device, model
    )
    engine.init_transfer_engine(
        M2NWeightTransferInitInfo(
            master_address=master_address,
            master_port=master_port,
            rank_offset=1,
            world_size=2,
            src_mesh_dims=[1, 1],
            dst_mesh_dims=[1, 1],
            names=[name for name, _, _ in params],
            dtype_names=["float32"] * len(params),
            shapes=[shape for _, shape, _ in params],
            src_placements=[None] * len(params),
        )
    )
    live["comm"] = engine.model_update_group

    engine.receive_weights(
        M2NWeightTransferUpdateInfo(names=[name for name, _, _ in params])
    )
    torch.accelerator.synchronize()

    params_match = all(
        torch.equal(
            model.get_parameter(name).data, _m2n_param_tensor(shape, offset, device)
        )
        for name, shape, offset in params
    )
    handle = m2n.handles[0]
    comm = engine.model_update_group
    engine.shutdown()
    return {
        "params_match": params_match,
        "num_reshards": len(m2n.reshard_calls),
        "handle_destroyed": handle.destroyed,
        "comm_destroyed": comm.disabled,
        "group_cleared": engine.model_update_group is None,
    }


@pytest.mark.skipif(
    torch.accelerator.device_count() < 2,
    reason="Need at least 2 GPUs to run the M2N weight transfer test.",
)
def test_m2n_weight_transfer_between_processes():
    """End-to-end worker-side nccl_m2n round over a live 2-GPU communicator.

    Real StatelessProcessGroup rendezvous, real PyNcclCommunicator, real
    engine init/receive/shutdown; only the out-of-tree `nccl.m2n` kernel is
    replaced by a fake whose reshard delivers the trainer's bytes via a real
    NCCL broadcast on the engine's own communicator. Covers the init
    handshake wire types, rendezvous and rank assignment, `comm_ptr` on a
    live communicator, per-parameter buffer dtype/shape wiring, bitwise
    `load_weights` delivery in declared order, and `shutdown()` destroying
    the live handle and communicator (the fix that made shutdown release
    instead of just dropping the reference). The m2n reshard math itself is
    NOT covered here; it is exercised out of tree.
    """
    _init_ray_for_weight_transfer()

    master_address = "127.0.0.1"
    master_port = get_open_port()
    params = [("w", [4, 6], 1000.0), ("b", [3], 2000.0)]

    inference_future = m2n_worker_receive_weights.remote(
        master_address, master_port, params
    )
    trainer_future = m2n_trainer_broadcast_weights.remote(
        master_address, master_port, params
    )
    trainer_result, result = ray.get([trainer_future, inference_future])

    assert trainer_result, "Trainer should complete successfully"
    assert result["params_match"], f"Bitwise mismatch after the round: {result}"
    assert result["num_reshards"] == len(params)
    assert result["handle_destroyed"], "shutdown() must destroy the m2n handle"
    assert result["comm_destroyed"], (
        "shutdown() must destroy the NCCL communicator, not just drop it"
    )
    assert result["group_cleared"]
