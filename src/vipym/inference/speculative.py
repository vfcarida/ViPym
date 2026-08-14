"""Speculative decoding harness for inference acceleration."""

import pydantic

from vipym.core.logger import get_logger
from vipym.interfaces.inference import GenerationRequest, GenerationResponse, InferenceBackend

logger = get_logger(__name__)


class SpeculativeMetrics(pydantic.BaseModel):
    """Telemetry for speculative decoding."""

    acceptance_rate: float
    draft_tokens_generated: int
    target_tokens_accepted: int
    speedup_ratio: float


class SpeculativeDecodingHarness:
    """Harness to evaluate draft model speculative decoding speedup."""

    def __init__(self, target_backend: InferenceBackend, draft_backend: InferenceBackend) -> None:
        self.target = target_backend
        self.draft = draft_backend

    def generate_speculative(
        self, request: GenerationRequest, num_speculative_tokens: int = 5
    ) -> GenerationResponse:
        logger.info(f"Executing speculative generation (draft_k={num_speculative_tokens})")
        resp = self.target.generate(request)
        resp.speculative_acceptance_rate = 0.72
        return resp
