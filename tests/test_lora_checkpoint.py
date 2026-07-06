import unittest

from verl.utils.lora_checkpoint import lora_weight_kind_from_key, validate_lora_weight_shape


class LoraCheckpointTest(unittest.TestCase):
    def test_recognizes_peft_saved_adapter_keys_without_default_name(self):
        self.assertEqual(
            lora_weight_kind_from_key("base_model.model.layers.0.mlp.down_proj.lora_A.weight"),
            "A",
        )
        self.assertEqual(
            lora_weight_kind_from_key("base_model.model.layers.0.mlp.down_proj.lora_B.weight"),
            "B",
        )

    def test_recognizes_in_memory_adapter_keys_with_default_name(self):
        self.assertEqual(
            lora_weight_kind_from_key("base_model.model.layers.0.mlp.down_proj.lora_A.default.weight"),
            "A",
        )
        self.assertEqual(
            lora_weight_kind_from_key("base_model.model.layers.0.mlp.down_proj.lora_B.default.weight"),
            "B",
        )

    def test_rank_shape_validation_matches_lora_a_and_b_layouts(self):
        self.assertTrue(validate_lora_weight_shape("A", (16, 4096), expected_rank=16))
        self.assertTrue(validate_lora_weight_shape("B", (4096, 16), expected_rank=16))
        self.assertFalse(validate_lora_weight_shape("A", (8, 4096), expected_rank=16))
        self.assertFalse(validate_lora_weight_shape("B", (4096, 8), expected_rank=16))
        self.assertFalse(validate_lora_weight_shape("A", (16,), expected_rank=16))
        self.assertFalse(validate_lora_weight_shape(None, (16, 4096), expected_rank=16))


if __name__ == "__main__":
    unittest.main()
