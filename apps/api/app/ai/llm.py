from typing import get_args

import httpx
from langchain_openai import ChatOpenAI
from openai.types.shared import ReasoningEffort
from pydantic import SecretStr

# ReasoningEffort is Optional[Literal[...]]; keep the literal values.
REASONING_EFFORTS: tuple[str, ...] = tuple(
    value for arg in get_args(ReasoningEffort) for value in get_args(arg)
)


def parse_model_spec(spec: str) -> tuple[str, str | None]:
    """Split "gpt-5.4-mini@low" into the model and its reasoning effort. Without "@",
    the model's default effort is used."""
    model, _, effort = spec.partition("@")
    if not model or (effort and effort not in REASONING_EFFORTS):
        raise ValueError(
            f"Invalid model {spec!r}: use MODEL or MODEL@EFFORT, "
            f"EFFORT one of {', '.join(REASONING_EFFORTS)}"
        )
    return model, effort or None


def chat_model(
    api_key: SecretStr, spec: str, http_client: httpx.AsyncClient | None, timeout: float
) -> ChatOpenAI:
    """The chat model for a model spec such as "gpt-5.4" or "gpt-5.4-mini@low"."""
    model, effort = parse_model_spec(spec)
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        reasoning_effort=effort,
        timeout=timeout,
        max_retries=2,
        http_async_client=http_client,
    )
