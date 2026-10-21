from typing import NamedTuple

import pytest
import torch
from torch.optim import Optimizer

import prime_rl.trainer.optim.state_offload as state_offload_module
from prime_rl.trainer.optim.state_offload import CPUOffloadOptimizer

pytestmark = [pytest.mark.gpu]


class TrainedPair(NamedTuple):
    reference_model: torch.nn.Module
    reference_optimizer: Optimizer
    offloaded_model: torch.nn.Module
    offloaded_optimizer: CPUOffloadOptimizer


def make_model(seed: int) -> torch.nn.Sequential:
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(64, 128), torch.nn.GELU(), torch.nn.Linear(128, 32))
    return model.cuda()


def run_steps(model: torch.nn.Module, optimizer: Optimizer, num_steps: int, seed: int) -> None:
    torch.manual_seed(seed)
    for _ in range(num_steps):
        batch = torch.randn(16, 64, device="cuda")
        loss = model(batch).square().mean()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def assert_params_equal(reference_model: torch.nn.Module, model: torch.nn.Module) -> None:
    for (name, reference_param), (_, param) in zip(reference_model.named_parameters(), model.named_parameters()):
        assert torch.equal(reference_param, param), f"param mismatch: {name}"


def assert_states_offloaded(optimizer: CPUOffloadOptimizer) -> None:
    for param in optimizer.state:
        for key, value in optimizer.state[param].items():
            assert value.device.type == "cpu", f"state '{key}' on {value.device}, expected cpu"
            assert value.is_pinned(), f"state '{key}' not pinned"


def assert_state_dicts_equal(saved: dict, restored: dict) -> None:
    assert saved["param_groups"] == restored["param_groups"]
    assert saved["state"].keys() == restored["state"].keys()
    for index in saved["state"]:
        assert saved["state"][index].keys() == restored["state"][index].keys()
        for key, saved_value in saved["state"][index].items():
            restored_value = restored["state"][index][key]
            if isinstance(saved_value, torch.Tensor):
                assert torch.equal(saved_value.cpu(), restored_value.cpu()), f"state[{index}]['{key}'] mismatch"
            else:
                assert saved_value == restored_value, f"state[{index}]['{key}'] mismatch"


def state_data_ptrs(optimizer: CPUOffloadOptimizer) -> dict[tuple[int, str], int]:
    return {
        (id(param), key): value.data_ptr() for param in optimizer.state for key, value in optimizer.state[param].items()
    }


@pytest.fixture
def trained_pair() -> TrainedPair:
    reference_model = make_model(0)
    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=1e-2)
    run_steps(reference_model, reference_optimizer, num_steps=3, seed=1)

    offloaded_model = make_model(0)
    offloaded_optimizer = CPUOffloadOptimizer(torch.optim.AdamW(offloaded_model.parameters(), lr=1e-2))
    run_steps(offloaded_model, offloaded_optimizer, num_steps=3, seed=1)

    return TrainedPair(reference_model, reference_optimizer, offloaded_model, offloaded_optimizer)


def test_offloaded_steps_match_plain_adamw(trained_pair: TrainedPair) -> None:
    assert_params_equal(trained_pair.reference_model, trained_pair.offloaded_model)


def test_states_are_on_cpu_and_pinned_between_steps(trained_pair: TrainedPair) -> None:
    assert_states_offloaded(trained_pair.offloaded_optimizer)


def test_state_dict_round_trip_is_exact_and_training_continues(trained_pair: TrainedPair) -> None:
    optimizer = trained_pair.offloaded_optimizer

    saved = optimizer.state_dict()
    optimizer.load_state_dict(saved)
    restored = optimizer.state_dict()

    assert_state_dicts_equal(saved, restored)
    assert_states_offloaded(optimizer)

    run_steps(trained_pair.reference_model, trained_pair.reference_optimizer, num_steps=1, seed=2)
    run_steps(trained_pair.offloaded_model, optimizer, num_steps=1, seed=2)
    assert_params_equal(trained_pair.reference_model, trained_pair.offloaded_model)


def test_offload_reuses_pinned_buffers_across_steps() -> None:
    model = make_model(0)
    optimizer = CPUOffloadOptimizer(torch.optim.AdamW(model.parameters(), lr=1e-2))
    run_steps(model, optimizer, num_steps=2, seed=1)

    pointers_before = state_data_ptrs(optimizer)

    run_steps(model, optimizer, num_steps=1, seed=2)

    pointers_after = state_data_ptrs(optimizer)
    assert pointers_before.keys() == pointers_after.keys()
    for key, pointer in pointers_before.items():
        assert pointers_after[key] == pointer, f"state buffer reallocated for {key}"


def test_chunked_gpu_state_budget_keeps_selected_fp32_moments_resident() -> None:
    reference_model = make_model(0)
    reference_optimizer = torch.optim.AdamW(reference_model.parameters(), lr=1e-2)
    offloaded_model = make_model(0)
    base_optimizer = torch.optim.AdamW(offloaded_model.parameters(), lr=1e-2)
    first_param = next(iter(base_optimizer.param_groups[0]["params"]))
    budget_bytes = first_param.numel() * first_param.element_size() * 2
    optimizer = CPUOffloadOptimizer(
        base_optimizer,
        pin_memory=False,
        chunked=True,
        gpu_state_budget_bytes=budget_bytes,
        parameter_names={id(param): name for name, param in offloaded_model.named_parameters()},
    )

    assert optimizer.gpu_state_bytes == budget_bytes
    assert optimizer.gpu_state_parameter_names == ["0.weight"]
    run_steps(reference_model, reference_optimizer, num_steps=3, seed=1)
    run_steps(offloaded_model, optimizer, num_steps=3, seed=1)

    assert_params_equal(reference_model, offloaded_model)
    assert optimizer.state[first_param]["exp_avg"].device.type == "cuda"
    assert optimizer.state[first_param]["exp_avg_sq"].device.type == "cuda"
    assert optimizer.state[first_param]["exp_avg"].dtype == torch.float32
    assert optimizer.state[first_param]["exp_avg_sq"].dtype == torch.float32
    assert all(
        value.device.type == "cpu"
        for param, state in optimizer.state.items()
        if param is not first_param
        for value in state.values()
    )
    assert all(param is not first_param for param, _ in optimizer._cpu_buffers)

    mixed_state = optimizer.state_dict()
    state_for_first = mixed_state["state"][0]
    assert state_for_first["exp_avg"].device.type == "cuda"
    assert state_for_first["exp_avg_sq"].device.type == "cuda"


def test_chunked_oversized_cpu_state_matches_adamw_and_resumes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(state_offload_module, "STATE_CHUNK_TARGET_NUMEL", 4)

    def make_parameters() -> list[torch.nn.Parameter]:
        small = torch.nn.Parameter(torch.tensor([0.2, -0.4], device="cuda"))
        large = torch.nn.Parameter(torch.linspace(-1, 1, 13, device="cuda"))
        return [small, large]

    reference_params = make_parameters()
    reference = torch.optim.AdamW(
        reference_params,
        lr=2e-3,
        weight_decay=0.03,
        amsgrad=True,
        foreach=None,
    )
    offloaded_params = make_parameters()
    offloaded = CPUOffloadOptimizer(
        torch.optim.AdamW(
            offloaded_params,
            lr=2e-3,
            weight_decay=0.03,
            amsgrad=True,
            foreach=None,
        ),
        pin_memory=False,
        chunked=True,
    )

    def assign_grads(params: list[torch.nn.Parameter], step: int) -> None:
        params[0].grad = torch.full_like(params[0], 0.1 * step)
        params[1].grad = None if step == 2 else torch.linspace(-0.3, 0.2, 13, device="cuda") * step

    def assert_equal(
        candidate_params: list[torch.nn.Parameter] = offloaded_params,
        candidate_optimizer: CPUOffloadOptimizer = offloaded,
    ) -> None:
        for reference_param, candidate_param in zip(reference_params, candidate_params):
            assert torch.equal(reference_param, candidate_param)
        for reference_param, candidate_param in zip(reference_params, candidate_params):
            reference_state = reference.state[reference_param]
            offloaded_state = candidate_optimizer.state[candidate_param]
            assert reference_state.keys() == offloaded_state.keys()
            for key, reference_value in reference_state.items():
                offloaded_value = offloaded_state[key]
                if isinstance(reference_value, torch.Tensor):
                    assert torch.equal(reference_value.cpu(), offloaded_value.cpu()), key
                else:
                    assert reference_value == offloaded_value

    large_state_tensor = None
    for step in range(1, 4):
        assign_grads(reference_params, step)
        assign_grads(offloaded_params, step)
        reference.step()
        offloaded.step()
        reference.zero_grad(set_to_none=True)
        offloaded.zero_grad(set_to_none=True)
        assert_equal()
        if step == 1:
            large_state_tensor = offloaded.state[offloaded_params[1]]["exp_avg"]
        if step >= 2:
            assert offloaded.state[offloaded_params[1]]["exp_avg"] is large_state_tensor

    saved = offloaded.state_dict()
    resumed_params = [torch.nn.Parameter(param.detach().clone()) for param in offloaded_params]
    resumed = CPUOffloadOptimizer(
        torch.optim.AdamW(
            resumed_params,
            lr=2e-3,
            weight_decay=0.03,
            amsgrad=True,
            foreach=None,
        ),
        pin_memory=False,
        chunked=True,
    )
    resumed.load_state_dict(saved)
    assign_grads(reference_params, 4)
    assign_grads(resumed_params, 4)
    reference.step()
    resumed.step()
    assert_equal(resumed_params, resumed)
