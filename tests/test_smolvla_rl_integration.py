import contextlib
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

tensordict = pytest.importorskip("tensordict")
pytest.importorskip("ray")
pytest.importorskip("hydra")
OmegaConf = pytest.importorskip("omegaconf").OmegaConf
from tensordict import TensorDict

from verl import DataProto
from verl.trainer.main_ppo import RobRewardManager
from verl.utils.smolvla_utils import (
    build_smolvla_batch,
    smolvla_image_resize_size,
    unnormalize_smolvla_actions,
)
from verl.workers.actor.dp_rob import RobDataParallelPPOActor
from verl.workers.rollout.rob_rollout import RobHFRollout, _prepare_libero_action_for_env


def smolvla_trainer_config(*, dense_mode="log_only", reward_coef=5.0):
    return OmegaConf.create(
        {
            "actor_rollout_ref": {
                "model": {
                    "vla": "smolvla",
                    "action_token_len": 1,
                    "action_chunks_len": 8,
                }
            },
            "verifier": {"reward_coef": reward_coef},
            "reward": {
                "subgoal": {
                    "enabled": dense_mode != "disabled",
                    "mode": dense_mode,
                    "log": True,
                }
            },
        }
    )


def actor_config():
    return OmegaConf.create(
        {
            "vla": "smolvla",
            "ppo_mini_batch_size": 1,
            "ppo_micro_batch_size": 1,
            "use_dynamic_bsz": False,
            "ppo_max_token_len_per_gpu": 1024,
            "ulysses_sequence_parallel_size": 1,
            "grad_clip": 10.0,
            "flow_weight_beta": 1.0,
            "flow_weight_max": 20.0,
        }
    )


def smolvla_batch(*, advantages):
    advantages = torch.tensor(advantages, dtype=torch.float32).reshape(1, 2, 1)
    tensors = {
        "responses": torch.zeros(1, 2, 1, dtype=torch.long),
        "advantages": advantages,
        "finish_step": torch.tensor([16], dtype=torch.long),
        "action": torch.zeros(1, 2, 8, 7),
        "observation.state": torch.zeros(1, 2, 8),
        "observation.language.tokens": torch.zeros(1, 2, 4, dtype=torch.long),
        "observation.language.attention_mask": torch.ones(1, 2, 4, dtype=torch.bool),
        "observation.image": torch.zeros(1, 2, 3, 8, 8),
    }
    return DataProto(batch=TensorDict(tensors, batch_size=[1]))


class FakeSmolVLAPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.forward_calls = 0

    def forward(self, batch, reduction="none"):
        self.forward_calls += 1
        assert reduction == "none"
        assert "action" in batch
        per_sample = self.weight.expand(batch["action"].shape[0])
        return per_sample, {"raw_flow_loss": float(self.weight.detach().item())}


@pytest.fixture(autouse=True)
def no_cuda_or_dist(monkeypatch):
    monkeypatch.setattr(TensorDict, "cuda", lambda self: self, raising=False)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None, raising=False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None, raising=False)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None, raising=False)


def build_actor(module):
    actor = RobDataParallelPPOActor.__new__(RobDataParallelPPOActor)
    actor.config = actor_config()
    actor.actor_module = module
    actor.actor_optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    actor.use_remove_padding = False
    actor.ulysses_sequence_parallel_size = 1
    actor.use_ulysses_sp = False
    return actor


def test_smolvla_terminal_reward_is_counted_on_finished_chunk():
    manager = RobRewardManager(num_examine=0, config=smolvla_trainer_config(dense_mode="disabled"))
    data = DataProto.from_dict(
        tensors={
            "responses": torch.zeros(2, 3, 1, dtype=torch.long),
            "finish_step": torch.tensor([8, 16], dtype=torch.long),
            "complete": torch.tensor([True, False]),
            "acc": torch.tensor([1.0, 0.0]),
        }
    )

    reward_tensors, metrics = manager(data)

    assert reward_tensors["gt_scores"].tolist() == [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert reward_tensors["all"].tolist() == [[5.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
    assert metrics["verifier"] == pytest.approx(0.5)


def test_smolvla_dense_reward_is_added_per_chunk_with_terminal_reward():
    manager = RobRewardManager(num_examine=0, config=smolvla_trainer_config(dense_mode="add"))
    data = DataProto.from_dict(
        tensors={
            "responses": torch.zeros(1, 3, 1, dtype=torch.long),
            "finish_step": torch.tensor([24], dtype=torch.long),
            "complete": torch.tensor([True]),
            "acc": torch.tensor([1.0]),
            "reward_total": torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32),
        }
    )

    reward_tensors, metrics = manager(data)

    assert reward_tensors["subgoal_scores"].tolist()[0] == pytest.approx([0.1, 0.2, 0.3])
    assert reward_tensors["all"].tolist()[0] == pytest.approx([0.1, 0.2, 5.3])
    assert metrics["subgoal_dense"] == pytest.approx(0.6)


@pytest.mark.parametrize("label", ["lora_style", "full_model_style"])
def test_smolvla_training_update_starts_for_lora_and_full_model_styles(label):
    module = FakeSmolVLAPolicy()
    actor = build_actor(module)
    before = module.weight.detach().item()

    metrics = actor.update_policy(smolvla_batch(advantages=[0.0, 0.0]))

    assert module.forward_calls == 1
    assert metrics["actor/flow_loss"][0] == pytest.approx(1.0)
    assert metrics["actor/flow_valid_chunks"][0] == pytest.approx(2.0)
    assert module.weight.detach().item() != pytest.approx(before)


def test_smolvla_flow_weighted_loss_changes_with_advantage():
    flat_actor = build_actor(FakeSmolVLAPolicy())
    skewed_actor = build_actor(FakeSmolVLAPolicy())

    flat_metrics = flat_actor.update_policy(smolvla_batch(advantages=[0.0, 0.0]))
    skewed_metrics = skewed_actor.update_policy(smolvla_batch(advantages=[0.0, 2.0]))

    assert flat_metrics["actor/flow_loss"][0] == pytest.approx(1.0)
    expected_skewed = (torch.exp(torch.tensor(-1.0)) + torch.exp(torch.tensor(1.0))) / 2
    assert skewed_metrics["actor/flow_loss"][0] == pytest.approx(float(expected_skewed), rel=1e-5)
    assert skewed_metrics["actor/flow_loss"][0] > flat_metrics["actor/flow_loss"][0]


def test_smolvla_tokenizer_pads_to_max_prompt_length_across_workers():
    class FakeTokenizer:
        def __call__(self, prompts, padding, truncation, max_length, return_tensors):
            assert padding == "max_length"
            assert truncation is True
            assert return_tensors == "pt"
            batch_size = len(prompts)
            return {
                "input_ids": torch.zeros(batch_size, max_length, dtype=torch.long),
                "attention_mask": torch.ones(batch_size, max_length, dtype=torch.long),
            }

    fake_policy = SimpleNamespace(
        config=SimpleNamespace(image_features=["observation.image"]),
        model=SimpleNamespace(
            vlm_with_expert=SimpleNamespace(
                processor=SimpleNamespace(tokenizer=FakeTokenizer()),
            )
        ),
    )

    batch = build_smolvla_batch(
        fake_policy,
        [
            {
                "state": np.zeros(8, dtype=np.float32),
                "full_image": np.zeros((8, 8, 3), dtype=np.uint8),
            }
        ],
        ["short"],
        device=torch.device("cpu"),
        max_prompt_length=17,
    )

    assert batch["observation.language.tokens"].shape == (1, 17)
    assert batch["observation.language.attention_mask"].shape == (1, 17)


def test_fsdp_batch_size_normalization_never_returns_zero():
    from verl.workers.fsdp_workers import _divide_batch_size_for_mesh

    assert _divide_batch_size_for_mesh(1, 2) == 1
    assert _divide_batch_size_for_mesh(4, 2) == 2
    assert _divide_batch_size_for_mesh(64, 8) == 8


def test_smolvla_uses_checkpoint_visual_input_size():
    fake_policy = SimpleNamespace(
        config=SimpleNamespace(
            input_features={
                "observation.images.image": {"type": "VISUAL", "shape": [3, 256, 256]},
                "observation.state": {"type": "STATE", "shape": [8]},
            }
        )
    )

    assert smolvla_image_resize_size(fake_policy) == 256


def test_smolvla_applies_state_and_action_normalization_stats():
    class FakeTokenizer:
        def __call__(self, prompts, padding, truncation, max_length, return_tensors):
            return {
                "input_ids": torch.zeros(len(prompts), max_length, dtype=torch.long),
                "attention_mask": torch.ones(len(prompts), max_length, dtype=torch.long),
            }

    fake_policy = SimpleNamespace(
        config=SimpleNamespace(image_features=["observation.image"]),
        model=SimpleNamespace(
            vlm_with_expert=SimpleNamespace(
                processor=SimpleNamespace(tokenizer=FakeTokenizer()),
            )
        ),
        _verl_smolvla_norm_stats={
            "preprocessor": {
                "observation.state.mean": torch.tensor([1.0, 2.0]),
                "observation.state.std": torch.tensor([2.0, 4.0]),
            },
            "postprocessor": {
                "action.mean": torch.tensor([10.0, 20.0]),
                "action.std": torch.tensor([2.0, 4.0]),
            },
        },
    )

    batch = build_smolvla_batch(
        fake_policy,
        [
            {
                "state": np.array([3.0, 10.0], dtype=np.float32),
                "full_image": np.zeros((8, 8, 3), dtype=np.uint8),
            }
        ],
        ["task"],
        device=torch.device("cpu"),
        max_prompt_length=4,
    )

    assert torch.allclose(batch["observation.state"], torch.tensor([[1.0, 2.0]]))
    actions = unnormalize_smolvla_actions(fake_policy, torch.tensor([[[1.0, -1.0]]]))
    assert torch.allclose(actions, torch.tensor([[[12.0, 16.0]]]))


def test_smolvla_libero_action_keeps_gripper_convention():
    config = SimpleNamespace(model_family="smolvla", vla="smolvla")
    action = np.array([2.0, -2.0, 0.5, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)

    env_action = _prepare_libero_action_for_env(action, config)

    assert env_action.tolist() == pytest.approx([1.0, -1.0, 0.5, 0.0, 0.0, 0.0, -1.0])


def test_smolvla_select_action_mode_returns_single_action(monkeypatch):
    class FakeSelectPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(n_action_steps=1)
            self.select_calls = 0

        def select_action(self, batch):
            self.select_calls += 1
            return torch.tensor(
                [
                    [0.5, -2.0, 0.0, 0.0, 0.0, 0.0, 1.5],
                    [-0.5, 2.0, 0.0, 0.0, 0.0, 0.0, -1.5],
                ],
                dtype=torch.float32,
            )

    rollout = RobHFRollout.__new__(RobHFRollout)
    rollout.module = FakeSelectPolicy()
    rollout.config = SimpleNamespace(vla="smolvla")
    rollout.smolvla_action_mode = "select_action"
    rollout._fsdp_summon_context = lambda: contextlib.nullcontext()

    out = rollout._generate_one_step_smolvla(
        {
            "observation.state": torch.zeros(2, 8),
            "observation.language.tokens": torch.zeros(2, 4, dtype=torch.long),
            "observation.language.attention_mask": torch.ones(2, 4, dtype=torch.bool),
            "observation.image": torch.zeros(2, 3, 8, 8),
        }
    )

    assert rollout.module.select_calls == 1
    assert out["action"].shape == (2, 1, 7)
    assert out["env_action"].shape == (2, 1, 7)
    assert out["env_action"].amin().item() >= -1.0
    assert out["env_action"].amax().item() <= 1.0


def test_smolvla_chunk_mode_can_execute_shorter_prefix():
    class FakeChunkPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.predict_calls = 0

        def predict_action_chunk(self, batch):
            self.predict_calls += 1
            return torch.arange(2 * 6 * 7, dtype=torch.float32).reshape(2, 6, 7)

    rollout = RobHFRollout.__new__(RobHFRollout)
    rollout.module = FakeChunkPolicy()
    rollout.config = SimpleNamespace(vla="smolvla")
    rollout.smolvla_action_mode = "chunk"
    rollout.smolvla_chunk_steps = 3
    rollout._fsdp_summon_context = lambda: contextlib.nullcontext()

    out = rollout._generate_one_step_smolvla(
        {
            "observation.state": torch.zeros(2, 8),
            "observation.language.tokens": torch.zeros(2, 4, dtype=torch.long),
            "observation.language.attention_mask": torch.ones(2, 4, dtype=torch.bool),
            "observation.image": torch.zeros(2, 3, 8, 8),
        }
    )

    expected = torch.arange(2 * 6 * 7, dtype=torch.float32).reshape(2, 6, 7)[:, :3]
    assert rollout.module.predict_calls == 1
    assert out["action"].shape == (2, 3, 7)
    assert torch.equal(out["action"], expected)
    assert out["env_action"].shape == (2, 3, 7)


def test_smolvla_validation_minimal_output_produces_scores():
    pytest.importorskip("ray")
    from verl.trainer.ppo.ray_trainer import RayTrainer

    class FakeActorRolloutWG:
        def generate_sequences(self, prompts):
            batch_size = len(prompts)
            return DataProto.from_dict(
                tensors={
                    "complete": torch.tensor([True, False][:batch_size], dtype=torch.bool),
                    "finish_step": torch.tensor([8, 16][:batch_size], dtype=torch.long),
                }
            )

    trainer = RayTrainer.__new__(RayTrainer)
    trainer.config = OmegaConf.create(
        {
            "data": {"task_suite_name": "libero_spatial"},
            "trainer": {
                "validation": {
                    "target_rollouts": 2,
                    "max_passes": 1,
                    "save_video": False,
                }
            },
        }
    )
    trainer.tokenizer = SimpleNamespace(eos_token_id=0, pad_token_id=0)
    trainer.actor_rollout_wg = FakeActorRolloutWG()
    trainer.val_reward_fn = RobRewardManager(num_examine=0, config=smolvla_trainer_config())
    trainer.validation_call_idx = 0
    trainer.prev_val_task_success_state = {}
    trainer.val_dataloader = [
        {
            "task_id": torch.tensor([[0], [1]], dtype=torch.long),
            "trial_id": torch.tensor([[0], [0]], dtype=torch.long),
            "task_suite_name": np.array(["libero_spatial", "libero_spatial"], dtype=object),
        }
    ]

    metrics = trainer._validate(global_steps=3)

    assert metrics["test_score/all"] == pytest.approx(0.5)
    assert metrics["test_score/num_rollouts"] == pytest.approx(2.0)
    assert "test_score/libero_spatial" in metrics


def test_smolvla_lora_and_full_checkpoints_save(tmp_path, monkeypatch):
    pytest.importorskip("ray")
    pytest.importorskip("transformers")
    original_json_loads = json.loads
    original_json_dumps = json.dumps
    if "tensorflow" not in sys.modules:
        fake_tf = types.ModuleType("tensorflow")

        class FakeTFConfig:
            @staticmethod
            def set_visible_devices(*args, **kwargs):
                return None

        fake_tf.config = FakeTFConfig()
        sys.modules["tensorflow"] = fake_tf
    from verl.workers import fsdp_workers
    from verl.workers.fsdp_workers import RobActorRolloutRefWorker
    json.loads = original_json_loads
    json.dumps = original_json_dumps

    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0, raising=False)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None, raising=False)
    monkeypatch.setattr(fsdp_workers.RobActorRolloutRefWorker, "_validate_saved_lora_adapter", lambda *a, **k: None)

    class FakeTokenizer:
        def save_pretrained(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "tokenizer.marker").write_text("saved", encoding="utf-8")

    class FakePeft(torch.nn.Module):
        def save_pretrained(self, path, state_dict=None, safe_serialization=True):
            Path(path).mkdir(parents=True, exist_ok=True)
            (Path(path) / "adapter.marker").write_text("saved", encoding="utf-8")

    monkeypatch.setattr(fsdp_workers, "PeftModel", FakePeft)

    lora_worker = RobActorRolloutRefWorker.__new__(RobActorRolloutRefWorker)
    lora_worker._is_actor = True
    lora_worker._is_lora = True
    lora_worker._is_offload_param = False
    lora_worker._rank = 0
    lora_worker.actor_module = FakePeft()
    lora_worker.actor_module_fsdp = lora_worker.actor_module
    lora_worker.tokenizer = FakeTokenizer()
    lora_worker.model_local_path = "HuggingFaceVLA/smolvla_libero"
    lora_worker.config = OmegaConf.create(
        {
                "model": {
                    "vla": "smolvla",
                    "path": "HuggingFaceVLA/smolvla_libero",
                    "lora_rank": 4,
                    "lora_alpha": 8,
                    "target_modules": "smolvla-default",
            },
            "rollout": {"use_proprio": True},
        }
    )

    lora_path = tmp_path / "lora"
    lora_worker.save_checkpoint(str(lora_path))
    assert (lora_path / "lora_adapter" / "adapter.marker").exists()
    assert (lora_path / "checkpoint_metadata.json").exists()

    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    monkeypatch.setattr(
        FSDP,
        "state_dict_type",
        classmethod(lambda cls, module, *args, **kwargs: contextlib.nullcontext()),
        raising=False,
    )

    class FakeActorModule(torch.nn.Module):
        def state_dict(self):
            return {"weight": torch.ones(1)}

    class FakeSaveModel:
        def save_pretrained(self, path, state_dict=None):
            Path(path).mkdir(parents=True, exist_ok=True)
            torch.save(state_dict, Path(path) / "model_state.pt")

    full_worker = RobActorRolloutRefWorker.__new__(RobActorRolloutRefWorker)
    full_worker._is_actor = True
    full_worker._is_lora = False
    full_worker._is_offload_param = False
    full_worker._rank = 0
    full_worker.actor = SimpleNamespace(actor_module=FakeActorModule())
    full_worker.actor_module = FakeSaveModel()
    full_worker.actor_module_fsdp = full_worker.actor.actor_module
    full_worker.tokenizer = FakeTokenizer()
    full_worker.config = OmegaConf.create({"model": {"vla": "smolvla"}})

    full_path = tmp_path / "full"
    full_worker.save_checkpoint(str(full_path))
    assert (full_path / "model_state.pt").exists()
    assert (full_path / "tokenizer.marker").exists()
