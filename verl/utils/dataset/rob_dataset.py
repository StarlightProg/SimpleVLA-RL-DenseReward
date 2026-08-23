# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from omegaconf import ListConfig
import os
from typing import List, Union

import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer, PreTrainedTokenizer

from verl import DataProto
from verl.utils.fs import copy_local_path_from_hdfs

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
import json

try:
    from libero.libero import benchmark
except ImportError as e:
    print(f"Warning : can't import libero: {e}")
    
def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class LIBERO_Dataset(Dataset):
    def __init__(self,
                 task_suite_name,
                 num_trials_per_task=50,
                 train_val ="train",
                 ):
        
        self.task_suite_name = task_suite_name  
        self.num_trials_per_task = num_trials_per_task  
        self.train_val = train_val
        self._read_files_and_tokenize()

    def _read_files_and_tokenize(self):
        benchmark_dict = benchmark.get_benchmark_dict()
        task_suite = benchmark_dict[self.task_suite_name]()
        num_tasks_in_suite = task_suite.n_tasks
        dataframes = []
        
        if self.task_suite_name in benchmark_dict:
            for task_id in range(num_tasks_in_suite):
                if self.train_val == "train":
                    trials_range = list(range(0, int(self.num_trials_per_task)))
                elif self.train_val == "valid":
                    trials_range = list(range(0, int(self.num_trials_per_task)))  
                else:
                    raise ValueError
                for i in trials_range:
                    data = {
                        "task_suite_name": self.task_suite_name,
                        "task_id": torch.tensor(task_id, dtype=torch.int64).unsqueeze(0),
                        "trial_id": torch.tensor(i, dtype=torch.int64).unsqueeze(0),
                        "trial_seed": torch.tensor(-1, dtype=torch.int64).unsqueeze(0)
                    }
                    dataframes.append(data)
            self.dataframe = dataframes
            print(f'dataset len: {len(self.dataframe)}')
        else:
            raise ValueError
     

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        return self.dataframe[item]


class TaskBalancedHardBatchSampler(Sampler):
    """Batch sampler with rotating task coverage plus hard-task oversampling.

    Each batch reserves a fraction of slots for round-robin task-balanced
    sampling and uses the remaining slots for probabilistic hard-task sampling.
    Hard-task probabilities are based on ``1 - EMA(success_rate)`` and can be
    updated after validation.
    """

    def __init__(
        self,
        dataset,
        batch_size: int,
        uniform_fraction: float = 0.7,
        min_task_probability: float = 0.05,
        ema_momentum: float = 0.8,
        default_success: float = 0.5,
        seed: int = 0,
        drop_last: bool = True,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.uniform_fraction = float(np.clip(uniform_fraction, 0.0, 1.0))
        self.min_task_probability = float(max(min_task_probability, 0.0))
        self.ema_momentum = float(np.clip(ema_momentum, 0.0, 1.0))
        self.default_success = float(np.clip(default_success, 0.0, 1.0))
        self.drop_last = bool(drop_last)
        self.rng = np.random.default_rng(int(seed))

        self.task_to_indices = self._build_task_index(dataset)
        if not self.task_to_indices:
            raise ValueError("TaskBalancedHardBatchSampler requires samples with task_id")
        self.task_ids = sorted(self.task_to_indices)
        self.task_success_ema = {task_id: self.default_success for task_id in self.task_ids}
        self._task_order = []
        self._task_cursor = 0
        self._index_orders = {}
        self._index_cursors = {}
        self._reset_cycles()

    @staticmethod
    def _sample_task_id(sample) -> int | None:
        if not isinstance(sample, dict) or "task_id" not in sample:
            return None
        value = sample["task_id"]
        try:
            if isinstance(value, torch.Tensor):
                return int(value.reshape(-1)[0].item())
            return int(np.asarray(value).reshape(-1)[0])
        except Exception:
            return None

    @classmethod
    def _build_task_index(cls, dataset) -> dict[int, list[int]]:
        task_to_indices: dict[int, list[int]] = {}
        for index in range(len(dataset)):
            task_id = cls._sample_task_id(dataset[index])
            if task_id is None or task_id < 0:
                continue
            task_to_indices.setdefault(task_id, []).append(index)
        return task_to_indices

    def _reset_cycles(self):
        self._task_order = list(self.task_ids)
        self.rng.shuffle(self._task_order)
        self._task_cursor = 0
        self._index_orders = {}
        self._index_cursors = {}
        for task_id, indices in self.task_to_indices.items():
            order = list(indices)
            self.rng.shuffle(order)
            self._index_orders[task_id] = order
            self._index_cursors[task_id] = 0

    def __len__(self):
        if self.drop_last:
            return max(1, len(self.dataset) // self.batch_size)
        return max(1, int(np.ceil(len(self.dataset) / self.batch_size)))

    def set_task_success(self, task_success: dict[int, float], momentum: float | None = None):
        momentum = self.ema_momentum if momentum is None else float(np.clip(momentum, 0.0, 1.0))
        for task_id, success in task_success.items():
            task_id = int(task_id)
            if task_id not in self.task_success_ema:
                continue
            success = float(np.clip(success, 0.0, 1.0))
            old = self.task_success_ema.get(task_id, self.default_success)
            self.task_success_ema[task_id] = momentum * old + (1.0 - momentum) * success

    def task_probabilities(self) -> dict[int, float]:
        n_tasks = len(self.task_ids)
        uniform = np.full(n_tasks, 1.0 / n_tasks, dtype=np.float64)
        priorities = np.array(
            [max(1.0 - self.task_success_ema.get(task_id, self.default_success), 1e-6) for task_id in self.task_ids],
            dtype=np.float64,
        )
        hard = priorities / priorities.sum()
        mixed = self.uniform_fraction * uniform + (1.0 - self.uniform_fraction) * hard
        mixed = self._apply_probability_floor(mixed, self.min_task_probability)
        return {task_id: float(prob) for task_id, prob in zip(self.task_ids, mixed)}

    @staticmethod
    def _apply_probability_floor(probs: np.ndarray, floor: float) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / probs.sum()
        n_tasks = probs.size
        if floor <= 0:
            return probs
        floor = min(float(floor), 1.0 / n_tasks)
        fixed = np.zeros(n_tasks, dtype=bool)
        while True:
            below = probs < floor
            if not below.any():
                return probs / probs.sum()
            fixed |= below
            fixed_mass = floor * fixed.sum()
            if fixed_mass >= 1.0:
                return np.full(n_tasks, 1.0 / n_tasks, dtype=np.float64)
            remaining = 1.0 - fixed_mass
            free = ~fixed
            probs[fixed] = floor
            if free.any():
                free_sum = probs[free].sum()
                if free_sum <= 0:
                    probs[free] = remaining / free.sum()
                else:
                    probs[free] = probs[free] / free_sum * remaining

    def _next_uniform_tasks(self, count: int) -> list[int]:
        tasks = []
        for _ in range(max(0, int(count))):
            if self._task_cursor >= len(self._task_order):
                self.rng.shuffle(self._task_order)
                self._task_cursor = 0
            tasks.append(self._task_order[self._task_cursor])
            self._task_cursor += 1
        return tasks

    def _next_index_for_task(self, task_id: int) -> int:
        order = self._index_orders[task_id]
        cursor = self._index_cursors[task_id]
        if cursor >= len(order):
            self.rng.shuffle(order)
            cursor = 0
        index = order[cursor]
        self._index_cursors[task_id] = cursor + 1
        return index

    def __iter__(self):
        self._reset_cycles()
        for _ in range(len(self)):
            uniform_slots = int(round(self.batch_size * self.uniform_fraction))
            uniform_slots = min(self.batch_size, max(0, uniform_slots))
            hard_slots = self.batch_size - uniform_slots

            batch_tasks = self._next_uniform_tasks(uniform_slots)
            if hard_slots > 0:
                probs = self.task_probabilities()
                prob_array = np.array([probs[task_id] for task_id in self.task_ids], dtype=np.float64)
                sampled = self.rng.choice(self.task_ids, size=hard_slots, replace=True, p=prob_array)
                batch_tasks.extend(int(task_id) for task_id in sampled.tolist())

            self.rng.shuffle(batch_tasks)
            yield [self._next_index_for_task(task_id) for task_id in batch_tasks]

class Robotwin_Dataset(Dataset):
    def __init__(self, task_name, num_trials_per_task=50,train_val ="train"):
        if "robotwin2" in task_name:
            self.version = "2.0"
        else:
            self.version = "1.0"
        self.task_name = task_name  
        if self.version == "1.0":
            self.all_task_names = ["block_hammer_beat", "block_handover", "blocks_stack_easy", "blocks_stack_hard", "bottle_adjust", "container_place", "diverse_bottles_pick", "dual_bottles_pick_easy", "dual_bottles_pick_hard", "dual_shoes_place", "empty_cup_place", "mug_hanging_easy", "mug_hanging_hard", "pick_apple_messy", "put_apple_cabinet", "shoe_place", "tool_adjust"]
            self.all_task_names = ["robotwin_" + name for name in self.all_task_names]
        elif self.version == "2.0":
            self.all_task_names = [ "handover_mic",
                                    "move_can_pot",
                                    "pick_dual_bottles",
                                    "place_phone_stand",
                                    "click_bell",
                                    "place_a2b_left",
                                    "place_a2b_right",
                                    "lift_pot",
                                    "put_bottles_dustbin",
                                    "stack_blocks_two",
                                    "stack_bowls_two","handover_block","place_empty_cup","shake_bottle","move_stapler_pad","place_container_plate","blocks_ranking_rgb","beat_block_hammer","place_mouse_pad","place_shoe","move_pillbottle_pad"]  
            self.all_task_names = ["robotwin2_" + name for name in self.all_task_names]
        
        self.num_trials_per_task = num_trials_per_task  
        if train_val == "valid":
            self.num_trials_per_task=128
        self.train_val = train_val
        if self.version == "2.0":
            self.load_twin2_success_seed()
        else:
            self.train_start_seed_id = 100000
            self.eval_start_seed_id = 100000000
        self._read_files_and_tokenize()
        
    def load_twin2_success_seed(self):
        self.twin2_success_seeds = {}
        self.twin2_success_seeds_eval = {}

        current_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(current_dir)
        eval_seeds_path = os.path.join(base_dir, "envs/robotwin2/seeds/robotwin2_eval_seeds.json")
        with open(eval_seeds_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for task_name, task_data in data.items():
                assert "robotwin2_" + task_name in self.all_task_names
                self.twin2_success_seeds_eval["robotwin2_" + task_name] = task_data['success_seeds']

        train_seeds_path = os.path.join(base_dir, "envs/robotwin2/seeds/robotwin2_train_seeds.json")
        with open(train_seeds_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for task_name, task_data in data.items():
                assert "robotwin2_" + task_name in self.all_task_names
                self.twin2_success_seeds["robotwin2_" + task_name] = task_data['success_seeds']                 

    def _read_files_and_tokenize(self):
        dataframes = []
        seeds_eval = None
        if self.task_name in self.all_task_names:
            if self.train_val == "train":
                if self.version == "2.0":
                    suc_seeds = self.twin2_success_seeds[self.task_name]
                    seeds = [suc_seeds[k] for k in range(int(self.num_trials_per_task))]
                else:
                    seeds = list(range(0, int(self.num_trials_per_task))) * 5  # repeat 5 time. make sure train dataset has enough batch
                    seeds = [ s+self.train_start_seed_id for s in seeds ]
            else:
                if self.version == "2.0":
                    suc_seeds = self.twin2_success_seeds[self.task_name]
                    seeds = [suc_seeds[k] for k in range(int(self.num_trials_per_task))]
                    suc_seeds_eval = self.twin2_success_seeds_eval[self.task_name]
                    seeds_eval = [suc_seeds_eval[k] for k in range(int(self.num_trials_per_task))]
                else:
                    seeds = list(range(0, int(self.num_trials_per_task)))
                    seeds = [ s+self.train_start_seed_id for s in seeds ]
                    seeds_eval = list(range(0, int(self.num_trials_per_task)))
                    seeds_eval = [ s+self.eval_start_seed_id for s in seeds_eval ]
            
            for i in seeds:
                data = {
                    "task_suite_name": self.task_name,
                    "task_id": torch.tensor(-1, dtype=torch.int64).unsqueeze(0),
                    "trial_id": torch.tensor(i, dtype=torch.int64).unsqueeze(0),
                    "trial_seed": torch.tensor(i, dtype=torch.int64).unsqueeze(0),
                    "data_source": self.task_name + "_train_iid"
                }
                dataframes.append(data)
            
            if seeds_eval is not None:
                for i in seeds_eval:
                    data = {
                        "task_suite_name": self.task_name,
                        "task_id": torch.tensor(-1, dtype=torch.int64).unsqueeze(0),
                        "trial_id": torch.tensor(i, dtype=torch.int64).unsqueeze(0),
                        "trial_seed": torch.tensor(i, dtype=torch.int64).unsqueeze(0),
                        "data_source": self.task_name + "_eval_ood"
                    }
                    dataframes.append(data)
            
            self.dataframe = dataframes
            print(f'dataset len: {len(self.dataframe)}')
        else:
            raise ValueError

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        return self.dataframe[item]


class BufferedDataLoader:
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.batch_size = dataloader.batch_size
        if self.batch_size is None and getattr(dataloader, "batch_sampler", None) is not None:
            self.batch_size = getattr(dataloader.batch_sampler, "batch_size", None)
        self.buffer = []
        self.dataloader_iter = None

    def start_new_epoch(self):
        self.dataloader_iter = iter(self.dataloader)

    def get_next_batch(self):
        try:
            return next(self.dataloader_iter)
        except StopIteration:
            raise StopIteration

    def __len__(self):
        return len(self.dataloader)

    def add_to_buffer(self, samples):
        if len(self.buffer) == 0:
            self.buffer = samples
        else:
            self.buffer = DataProto.concat([self.buffer, samples])

    def get_from_buffer(self, count, dp_size):
        if count > self.buffer_size():
            count = (self.buffer_size() // dp_size) * dp_size
        samples = self.buffer.slice(range(0, count))
        self.buffer = self.buffer.slice(range(count, self.buffer_size()))
        return samples

    def buffer_size(self):
        return len(self.buffer)
