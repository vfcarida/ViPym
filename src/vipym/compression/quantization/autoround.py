"""AutoRound sign-gradient rounding optimization adapter.

References:
- Cheng et al., "Optimize Weight Rounding via Signed Gradient Descent for Large Language Model
  Quantization", 2023.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from vipym.compression.registry import CompressionRegistry
from vipym.core.constants import ComputeArchitecture, SupportedDtype
from vipym.core.logger import get_logger
from vipym.interfaces.compression import CompressionArtifact, CompressionMethod
from vipym.interfaces.model import ModelMetadata, PluginCapability

logger = get_logger(__name__)


class AutoRoundCompressionMethod(CompressionMethod):
    """AutoRound Advanced Rounding Optimization with native PyTorch Sign-SGD fallback."""

    def __init__(
        self,
        bits: int = 4,
        group_size: int = 128,
        iters: int = 50,
        lr: float = 0.05,
        sym: bool = True,
        **kwargs: Any,
    ) -> None:
        self.bits = bits
        self.group_size = group_size
        self.iters = iters
        self.lr = lr
        self.sym = sym
        self.extra_kwargs = kwargs

    @property
    def name(self) -> str:
        return f"autoround_w{self.bits}a16_g{self.group_size}"

    def get_capabilities(self) -> PluginCapability:
        return PluginCapability(
            supported_architectures={
                ComputeArchitecture.DENSE,
                ComputeArchitecture.MOE,
            },
            supported_dtypes={SupportedDtype.INT4, SupportedDtype.FP16, SupportedDtype.BF16},
            supports_moe=True,
            requires_calibration=True,
            supported_runtimes={"vllm", "sglang", "hf"},
        )

    def validate_applicability(self, model_metadata: ModelMetadata) -> None:
        pass

    def _collect_layer_inputs(
        self,
        model: nn.Module,
        linear_layers: list[tuple[str, nn.Module]],
        calibration_data: Any | None,
        tokenizer: Any | None,
        max_tokens: int = 512,
    ) -> dict[str, torch.Tensor]:
        """Collect intermediate activation inputs for each linear layer via PyTorch forward hooks."""
        if calibration_data is None:
            return {}

        inputs_map: dict[str, list[torch.Tensor]] = {name: [] for name, _ in linear_layers}
        hooks = []

        def make_hook(name: str):
            def hook_fn(_mod: nn.Module, inp: tuple[Any, ...], _out: Any) -> None:
                if inp and isinstance(inp[0], torch.Tensor):
                    x = inp[0].detach()
                    # Reshape to (Batch * Seq, In_Features)
                    if x.dim() >= 2:
                        flat_x = x.view(-1, x.shape[-1])
                        # Subsample if large
                        if flat_x.shape[0] > max_tokens:
                            flat_x = flat_x[:max_tokens]
                        inputs_map[name].append(flat_x.cpu())

            return hook_fn

        for name, layer in linear_layers:
            hooks.append(layer.register_forward_hook(make_hook(name)))

        model.eval()
        device = next(model.parameters()).device

        try:
            with torch.no_grad():
                # Process calibration samples
                samples: list[str] = []
                if isinstance(calibration_data, list):
                    for item in calibration_data[:8]:
                        if isinstance(item, str):
                            samples.append(item)
                        elif isinstance(item, dict) and "text" in item:
                            samples.append(item["text"])
                        elif isinstance(item, dict) and "prompt" in item:
                            samples.append(item["prompt"])
                elif hasattr(calibration_data, "texts"):
                    samples = calibration_data.texts[:8]

                if not samples and tokenizer is not None:
                    samples = [
                        "def factorial(n: int) -> int:\n    return 1 if n <= 1 else n * factorial(n - 1)\n",
                        "class Node:\n    def __init__(self, val=0):\n        self.val = val\n",
                    ]

                for sample in samples:
                    if tokenizer is not None:
                        encoded = tokenizer(
                            sample,
                            return_tensors="pt",
                            truncation=True,
                            max_length=max_tokens,
                        )
                        input_ids = encoded["input_ids"].to(device)
                        model(input_ids)
                    elif hasattr(model, "forward"):
                        # Dummy forward with random int IDs if tokenizer absent, or float fallback
                        try:
                            dummy_ids = torch.randint(
                                0, 1000, (1, min(64, max_tokens)), device=device
                            )
                            model(dummy_ids)
                        except Exception:
                            first_layer = linear_layers[0][1] if linear_layers else None
                            in_dim = getattr(first_layer, "in_features", 64) if first_layer else 64
                            dummy_floats = torch.randn(1, in_dim, device=device)
                            model(dummy_floats)
        except Exception as exc:
            logger.warning(
                f"Activation hook collection encountered error ({exc}); continuing with captured activations."
            )
        finally:
            for hook in hooks:
                hook.remove()

        # Concatenate collected inputs
        final_inputs: dict[str, torch.Tensor] = {}
        for name, tensor_list in inputs_map.items():
            if tensor_list:
                cat_t = torch.cat(tensor_list, dim=0)
                if cat_t.shape[0] > max_tokens:
                    cat_t = cat_t[:max_tokens]
                final_inputs[name] = cat_t

        return final_inputs

    def _optimize_layer_sign_sgd(
        self,
        weight: torch.Tensor,
        inputs: torch.Tensor | None,
        bits: int,
        group_size: int,
        iters: int,
        lr: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Perform signed gradient descent rounding optimization on layer weights.

        Minimizes ||W X^T - W_hat(V) X^T||_F^2 using Sign-SGD on continuous perturbation V.
        """
        w_orig = weight.float()
        out_f, in_f = w_orig.shape

        # Calculate group boundaries
        eff_group = group_size if (group_size > 0 and in_f % group_size == 0) else in_f
        w_grouped = w_orig.view(-1, eff_group)

        # Scale and base integer calculation
        w_min = w_grouped.amin(dim=-1, keepdim=True)
        w_max = w_grouped.amax(dim=-1, keepdim=True)
        scale = torch.clamp((w_max - w_min) / float((2**bits) - 1), min=1e-8)
        base = torch.clamp(torch.floor((w_grouped - w_min) / scale), 0, (2**bits) - 2)

        # Fractional residual in [0, 1)
        res = (w_grouped - w_min) / scale - base

        # Baseline Round-to-Nearest (RTN)
        rtn_binary = (res >= 0.5).float()
        w_rtn = (w_min + scale * (base + rtn_binary)).view(out_f, in_f)

        if inputs is None or inputs.numel() == 0 or iters <= 0:
            return w_rtn.to(weight.dtype), {
                "initial_loss": 0.0,
                "final_loss": 0.0,
                "loss_reduction": 0.0,
            }

        # Subsample inputs for fast, stable Sign-SGD
        device = weight.device
        x_calib = inputs[: min(256, inputs.shape[0])].float().to(device)
        w_dev = w_orig.to(device)
        w_min_dev = w_min.to(device)
        scale_dev = scale.to(device)
        base_dev = base.to(device)
        res_dev = res.to(device)

        # Target unquantized layer output
        with torch.no_grad():
            y_target = torch.matmul(x_calib, w_dev.t())
            w_rtn_dev = w_rtn.to(device)
            y_rtn = torch.matmul(x_calib, w_rtn_dev.t())
            initial_loss = torch.mean((y_target - y_rtn) ** 2).item()

        # Track best discrete rounding candidate (guaranteeing monotonic non-increasing error vs RTN)
        best_loss = initial_loss
        best_w = w_rtn_dev.clone()

        # Initialize continuous rounding decision variable V
        v = ((res_dev - 0.5) * 4.0).detach().clone()
        v.requires_grad_(True)

        for _ in range(iters):
            v_prob = torch.sigmoid(v)
            w_candidate = (w_min_dev + scale_dev * (base_dev + v_prob)).view(out_f, in_f)
            y_pred = torch.matmul(x_calib, w_candidate.t())
            loss = torch.mean((y_target - y_pred) ** 2)
            loss.backward()

            with torch.no_grad():
                if v.grad is not None:
                    # Sign-SGD step
                    v.data -= lr * torch.sign(v.grad)
                    v.grad.zero_()

                # Evaluate discrete binarization and update best candidate
                v_disc = (torch.sigmoid(v) >= 0.5).float()
                w_disc = (w_min_dev + scale_dev * (base_dev + v_disc)).view(out_f, in_f)
                y_disc = torch.matmul(x_calib, w_disc.t())
                disc_loss = torch.mean((y_target - y_disc) ** 2).item()
                if disc_loss < best_loss:
                    best_loss = disc_loss
                    best_w = w_disc.clone()

        loss_red = (
            max(0.0, initial_loss - best_loss) / max(initial_loss, 1e-8)
            if initial_loss > 0
            else 0.0
        )

        return best_w.to(weight.dtype), {
            "initial_loss": initial_loss,
            "final_loss": best_loss,
            "loss_reduction": loss_red,
        }

    def compress(
        self,
        model: nn.Module,
        tokenizer: Any,
        calibration_data: Any | None = None,
        output_dir: Path | None = None,
        **kwargs: Any,
    ) -> CompressionArtifact:
        out = Path(output_dir or "./autoround_model")
        out.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Executing AutoRound quantization (bits={self.bits}, group_size={self.group_size}, iters={self.iters})"
        )

        auto_round_success = False
        try:
            from auto_round import AutoRound

            autoround = AutoRound(
                model=model,
                tokenizer=tokenizer,
                bits=self.bits,
                group_size=self.group_size,
                iters=self.iters,
            )
            autoround.quantize()
            autoround.save_quantized(output_dir=str(out), format="auto_round")
            auto_round_success = True
        except (ImportError, Exception) as exc:
            logger.info(
                f"auto_round package execution skipped ({exc}); executing native PyTorch Sign-SGD layer-wise optimization."
            )

        telemetry: dict[str, Any] = {
            "bits": self.bits,
            "group_size": self.group_size,
            "iters": self.iters,
            "native_sign_sgd": not auto_round_success,
            "layers_optimized": 0,
            "mean_loss_reduction": 0.0,
        }

        if not auto_round_success:
            linear_layers: list[tuple[str, nn.Module]] = []
            for name, module in model.named_modules():
                if (
                    (isinstance(module, nn.Linear) or module.__class__.__name__ == "Conv1D")
                    and not any(k in name.lower() for k in ("lm_head", "embed", "wte", "wpe"))
                    and hasattr(module, "weight")
                    and module.weight is not None
                    and len(module.weight.shape) == 2
                ):
                    linear_layers.append((name, module))

            logger.info(
                f"Collecting activation hooks for {len(linear_layers)} candidate layers in AutoRound..."
            )
            inputs_map = self._collect_layer_inputs(
                model=model,
                linear_layers=linear_layers,
                calibration_data=calibration_data,
                tokenizer=tokenizer,
            )

            loss_reductions = []
            for name, module in linear_layers:
                is_conv1d = module.__class__.__name__ == "Conv1D"
                w_tensor = module.weight.data.t() if is_conv1d else module.weight.data
                layer_inputs = inputs_map.get(name)

                opt_weight, metrics = self._optimize_layer_sign_sgd(
                    weight=w_tensor,
                    inputs=layer_inputs,
                    bits=self.bits,
                    group_size=self.group_size,
                    iters=self.iters,
                    lr=self.lr,
                )

                module.weight.data.copy_(opt_weight.t() if is_conv1d else opt_weight)
                loss_reductions.append(metrics["loss_reduction"])

            telemetry["layers_optimized"] = len(linear_layers)
            telemetry["mean_loss_reduction"] = (
                float(sum(loss_reductions) / len(loss_reductions)) if loss_reductions else 0.0
            )

            # Persist model weights / checkpoint
            if hasattr(model, "save_pretrained"):
                model.save_pretrained(out)
            else:
                torch.save(model.state_dict(), out / "model.pt")

            if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
                tokenizer.save_pretrained(out)

            # Save autoround metadata and scales
            (out / "autoround_metadata.json").write_text(
                json.dumps(telemetry, indent=2), encoding="utf-8"
            )

            from vipym.compression.export import write_quantization_config

            write_quantization_config(
                output_dir=out,
                quant_method="compressed-tensors",
                format_type="pack-quantized",
                bits=self.bits,
                group_size=self.group_size,
                symmetric=self.sym,
            )

        # Compute empirical compressed size
        total_bytes = sum(f.stat().st_size for f in out.glob("**/*") if f.is_file())
        if total_bytes == 0:
            total_params = sum(p.numel() for p in model.parameters())
            total_bytes = int(total_params * (self.bits / 8.0))

        return CompressionArtifact(
            output_path=out,
            format="compressed-tensors",
            compressed_size_bytes=total_bytes,
            applied_methods=[self.name],
            metadata=telemetry,
        )


CompressionRegistry.register("autoround", AutoRoundCompressionMethod)
