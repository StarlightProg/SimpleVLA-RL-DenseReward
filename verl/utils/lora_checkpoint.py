def lora_weight_kind_from_key(key: str, adapter_name: str = "default"):
    """Return LoRA weight kind for PEFT keys saved with or without adapter_name."""

    suffixes = {
        "A": (
            f".lora_A.{adapter_name}.weight",
            f".lora_A.{adapter_name}",
            f".lora_embedding_A.{adapter_name}.weight",
            f".lora_embedding_A.{adapter_name}",
            ".lora_A.weight",
            ".lora_embedding_A.weight",
        ),
        "B": (
            f".lora_B.{adapter_name}.weight",
            f".lora_B.{adapter_name}",
            f".lora_embedding_B.{adapter_name}.weight",
            f".lora_embedding_B.{adapter_name}",
            ".lora_B.weight",
            ".lora_embedding_B.weight",
        ),
    }
    for kind, candidates in suffixes.items():
        for suffix in candidates:
            if key.endswith(suffix) or key == suffix[1:]:
                return kind
    return None


def validate_lora_weight_shape(kind: str, shape, expected_rank: int) -> bool:
    if len(shape) < 2:
        return False
    if kind == "A":
        return shape[0] == expected_rank
    if kind == "B":
        return shape[1] == expected_rank
    return False
