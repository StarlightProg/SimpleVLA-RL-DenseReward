from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterator

import h5py
import numpy as np
import tensorflow_datasets as tfds


class LiberoCubePlaceNoNoops(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {"1.0.0": "Initial cube-place dataset release."}

    def _info(self) -> tfds.core.DatasetInfo:
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict(
                {
                    "steps": tfds.features.Dataset(
                        {
                            "observation": tfds.features.FeaturesDict(
                                {
                                    "image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                    ),
                                    "wrist_image": tfds.features.Image(
                                        shape=(256, 256, 3),
                                        dtype=np.uint8,
                                        encoding_format="jpeg",
                                    ),
                                    "state": tfds.features.Tensor(shape=(8,), dtype=np.float32),
                                    "joint_state": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                                }
                            ),
                            "action": tfds.features.Tensor(shape=(7,), dtype=np.float32),
                            "discount": tfds.features.Scalar(dtype=np.float32),
                            "reward": tfds.features.Scalar(dtype=np.float32),
                            "dense_reward": tfds.features.Scalar(dtype=np.float32),
                            "is_first": tfds.features.Scalar(dtype=np.bool_),
                            "is_last": tfds.features.Scalar(dtype=np.bool_),
                            "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                            "terminal_success": tfds.features.Scalar(dtype=np.bool_),
                            "language_instruction": tfds.features.Text(),
                            "seed": tfds.features.Scalar(dtype=np.int64),
                            "task_id": tfds.features.Scalar(dtype=np.int64),
                            "episode_id": tfds.features.Text(),
                        }
                    ),
                    "episode_metadata": tfds.features.FeaturesDict(
                        {
                            "file_path": tfds.features.Text(),
                            "episode_id": tfds.features.Text(),
                            "seed": tfds.features.Scalar(dtype=np.int64),
                            "task_id": tfds.features.Scalar(dtype=np.int64),
                            "terminal_success": tfds.features.Scalar(dtype=np.bool_),
                            "initial_cube_position": tfds.features.Tensor(
                                shape=(3,), dtype=np.float32
                            ),
                            "initial_bowl_position": tfds.features.Tensor(
                                shape=(3,), dtype=np.float32
                            ),
                            "initial_target_position": tfds.features.Tensor(
                                shape=(3,), dtype=np.float32
                            ),
                            "final_cube_position": tfds.features.Tensor(
                                shape=(3,), dtype=np.float32
                            ),
                            "relation": tfds.features.Text(),
                        }
                    ),
                }
            )
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        del dl_manager
        source = os.environ.get("LIBERO_CUBE_PLACE_HDF5")
        if not source:
            raise ValueError("Set LIBERO_CUBE_PLACE_HDF5 to the generated HDF5 file.")
        return {"train": self._generate_examples(Path(source).expanduser().resolve())}

    def _generate_examples(self, path: Path) -> Iterator[tuple[str, dict[str, Any]]]:
        with h5py.File(path, "r") as handle:
            for episode_id in sorted(handle["data"], key=lambda name: int(name.split("_")[-1])):
                episode = handle["data"][episode_id]
                length = len(episode["actions"])
                instruction = str(episode.attrs["language_instruction"])
                seed = int(episode.attrs["seed"])
                task_id = int(episode.attrs["task_id"])
                success = bool(episode.attrs["terminal_success"])
                steps = []
                for index in range(length):
                    steps.append(
                        {
                            "observation": {
                                "image": episode["obs/agentview_rgb"][index][::-1, ::-1],
                                "wrist_image": episode["obs/eye_in_hand_rgb"][index][::-1, ::-1],
                                "state": episode["robot_states"][index].astype(np.float32),
                                "joint_state": episode["obs/joint_states"][index].astype(np.float32),
                            },
                            "action": episode["actions"][index].astype(np.float32),
                            "discount": np.float32(1.0),
                            "reward": np.float32(episode["rewards"][index]),
                            "dense_reward": np.float32(episode["dense_rewards"][index]),
                            "is_first": index == 0,
                            "is_last": index == length - 1,
                            "is_terminal": bool(success and index == length - 1),
                            "terminal_success": bool(success),
                            "language_instruction": instruction,
                            "seed": np.int64(seed),
                            "task_id": np.int64(task_id),
                            "episode_id": episode_id,
                        }
                    )
                yield episode_id, {
                    "steps": steps,
                    "episode_metadata": {
                        "file_path": str(path),
                        "episode_id": episode_id,
                        "seed": np.int64(seed),
                        "task_id": np.int64(task_id),
                        "terminal_success": bool(success),
                        "initial_cube_position": np.asarray(
                            episode.attrs["initial_cube_position"], dtype=np.float32
                        ),
                        "initial_bowl_position": np.asarray(
                            episode.attrs["initial_bowl_position"], dtype=np.float32
                        ),
                        "initial_target_position": np.asarray(
                            episode.attrs["initial_target_position"], dtype=np.float32
                        ),
                        "final_cube_position": np.asarray(
                            episode.attrs["final_cube_position"], dtype=np.float32
                        ),
                        "relation": str(episode.attrs["relation"]),
                    },
                }
