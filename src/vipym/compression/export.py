"""Export utilities for serialized compressed checkpoints and runtime configs."""

import json
from pathlib import Path
from typing import Any

from vipym.core.logger import get_logger

logger = get_logger(__name__)


def write_quantization_config(
    output_dir: Path | str,
    quant_method: str = "compressed-tensors",
    format_type: str = "pack-quantized",
    scheme: str | None = None,
    bits: int | None = None,
    group_size: int | None = None,
    symmetric: bool = True,
    extra_config: dict[str, Any] | None = None,
) -> Path:
    """Inject standard vLLM-compatible quantization_config into config.json."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"

    config_dict: dict[str, Any] = {}
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                config_dict = json.load(f)
        except Exception as e:
            logger.warning(f"Could not read existing config.json at {config_path}: {e}")

    # Build compressed-tensors or standard quantization config
    q_config: dict[str, Any] = {
        "quant_method": quant_method,
    }
    if format_type:
        q_config["format"] = format_type

    if quant_method == "compressed-tensors":
        group_weights: dict[str, Any] = {
            "type": "int",
            "symmetric": symmetric,
        }
        if bits is not None:
            group_weights["num_bits"] = bits
        if group_size is not None:
            group_weights["strategy"] = "group"
            group_weights["group_size"] = group_size
        else:
            group_weights["strategy"] = "channel"

        q_config["config_groups"] = {
            "group_0": {
                "weights": group_weights,
            }
        }
    elif quant_method == "fp8":
        q_config.update(
            {
                "activation_scheme": scheme or "dynamic",
                "weight_dtype": "fp8_e4m3",
                "activation_dtype": "fp8_e4m3",
                "quantized_weights": True,
            }
        )

    if extra_config:
        q_config.update(extra_config)

    config_dict["quantization_config"] = q_config

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    return config_path
