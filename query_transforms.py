import json
import os
import re
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


_TERM_VARIANTS = {
    "emi": ["equated monthly instalment", "equated monthly installment"],
    "equated monthly instalment": ["emi", "equated monthly installment"],
    "equated monthly installment": ["emi", "equated monthly instalment"],
    "interest rate": ["rate of interest", "roi"],
    "rate of interest": ["interest rate", "roi"],
    "roi": ["rate of interest", "interest rate"],
    "loan amount": ["principal amount", "sanctioned amount"],
    "sanctioned amount": ["loan amount", "approved loan amount"],
    "installments": ["instalments", "repayment months"],
    "instalments": ["installments", "repayment months"],
    "tenure": ["loan term", "repayment period"],
}


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


def generate_query_variants(query: str, n: int = 4) -> list[str]:
    """Generate diverse lexical variants for retrieval expansion."""
    normalized = " ".join(query.strip().split())
    if not normalized:
        return []

    variants: list[str] = []
    lowered = normalized.lower()

    def _add_variant(text: str) -> None:
        cleaned = " ".join(text.strip().split())
        if cleaned and cleaned.lower() != lowered and cleaned not in variants:
            variants.append(cleaned)

    for term, replacements in _TERM_VARIANTS.items():
        pattern = rf"\b{re.escape(term)}\b"
        if re.search(pattern, lowered):
            for replacement in replacements:
                candidate = re.sub(pattern, replacement, lowered)
                _add_variant(candidate)
                if len(variants) >= n:
                    return variants[:n]

    # Fallback generic rewrites when domain replacements are not enough.
    generic_templates = [
        f"details about {normalized}",
        f"{normalized} in loan agreement",
        f"official value of {normalized}",
        f"{normalized} explained in financial terms",
    ]
    for template in generic_templates:
        _add_variant(template)
        if len(variants) >= n:
            break

    return variants[:n]
