"""Unit tests for compressed-tensors and vLLM quantization configuration export utilities."""

import json
import tempfile
from pathlib import Path

from vipym.compression.export import write_quantization_config


def test_write_compressed_tensors_quantization_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "test_model"
        cfg_path = write_quantization_config(
            output_dir=out_dir,
            quant_method="compressed-tensors",
            format_type="pack-quantized",
            bits=4,
            group_size=128,
            symmetric=True,
        )

        assert cfg_path.exists()
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        assert "quantization_config" in data
        qcfg = data["quantization_config"]
        assert qcfg["quant_method"] == "compressed-tensors"
        assert qcfg["format"] == "pack-quantized"
        assert "group_0" in qcfg["config_groups"]
        weights = qcfg["config_groups"]["group_0"]["weights"]
        assert weights["num_bits"] == 4
        assert weights["group_size"] == 128
        assert weights["symmetric"] is True
        assert weights["strategy"] == "group"


def test_write_fp8_quantization_config():
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "test_fp8_model"
        cfg_path = write_quantization_config(
            output_dir=out_dir,
            quant_method="fp8",
            scheme="static",
            extra_config={"activation_dtype": "fp8_e4m3"},
        )

        assert cfg_path.exists()
        data = json.loads(cfg_path.read_text(encoding="utf-8"))
        qcfg = data["quantization_config"]
        assert qcfg["quant_method"] == "fp8"
        assert qcfg["activation_scheme"] == "static"
        assert qcfg["activation_dtype"] == "fp8_e4m3"
