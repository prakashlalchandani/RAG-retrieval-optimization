import json
import os
from dataclasses import dataclass
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen


DEFAULT_HYDE_PROMPT = (
    "You are writing a short hypothetical answer to improve document retrieval. "
    "Given the user query, produce a concise answer-like paragraph that likely "
    "contains key terminology and numeric fields from the target document. "
    "Do not mention uncertainty.\\n\\n"
    "Query: {query}\\n"
    "Hypothetical answer:"
)


@dataclass
class HydeConfig:
    enabled: bool = os.getenv("HYDE_ENABLED", "true").lower() == "true"
    timeout_seconds: float = float(os.getenv("HYDE_TIMEOUT_SECONDS", "4.0"))
    prompt_template: str = os.getenv("HYDE_PROMPT_TEMPLATE", DEFAULT_HYDE_PROMPT)

    # OpenAI-compatible chat completion endpoint.
    endpoint: str = os.getenv(
        "HYDE_LLM_ENDPOINT", "https://api.openai.com/v1/chat/completions"
    )
    model: str = os.getenv("HYDE_LLM_MODEL", "gpt-4o-mini")
    api_key: Optional[str] = os.getenv("HYDE_LLM_API_KEY")


HYDE_CONFIG = HydeConfig()


def _call_openai_compatible_chat(prompt: str, config: HydeConfig) -> str:
    if not config.api_key:
        raise RuntimeError("HYDE_LLM_API_KEY is not set")

    payload = {
        "model": config.model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": "You generate retrieval-focused hypothetical answers."},
            {"role": "user", "content": prompt},
        ],
    }

    request = Request(
        config.endpoint,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=config.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except URLError as exc:
        raise TimeoutError(f"HyDE call failed or timed out: {exc}") from exc

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("HyDE response did not contain choices")

    content = choices[0].get("message", {}).get("content", "").strip()
    if not content:
        raise RuntimeError("HyDE response content was empty")

    return content


def generate_hypothetical_answer(query: str, config: HydeConfig = HYDE_CONFIG) -> str:
    """Generate a HyDE-style hypothetical answer for a query."""
    if not config.enabled:
        return query

    prompt = config.prompt_template.format(query=query)
    return _call_openai_compatible_chat(prompt, config)


def text_for_retrieval(query: str, config: HydeConfig = HYDE_CONFIG) -> tuple[str, bool, Optional[str]]:
    """Return text to embed with fallback to raw query on HyDE failure.

    Returns: (text_to_embed, used_hyde, error_message)
    """
    if not config.enabled:
        return query, False, None

    try:
        hyde_text = generate_hypothetical_answer(query, config=config)
        return hyde_text, True, None
    except Exception as exc:  # fallback behavior is intentional
        return query, False, str(exc)
