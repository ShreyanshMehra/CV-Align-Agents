"""Provider-agnostic LLM client factory.

Agents depend on LangChain's :class:`BaseChatModel` abstraction rather than any
concrete provider class. Switching providers (e.g. Gemini -> Groq) is therefore
a configuration change (``LLM_PROVIDER`` in ``.env``) with no code changes in the
agents themselves.
"""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from cv_align_agents.settings import Settings, get_settings

# Sensible free-tier defaults per provider, used when LLM_MODEL is not set.
DEFAULT_MODELS: dict[str, str] = {
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.3-70b-versatile",
}


class LLMConfigError(RuntimeError):
    """Raised when the LLM provider is misconfigured (e.g. missing API key)."""


def get_chat_model(
    *,
    temperature: float = 0.0,
    settings: Settings | None = None,
    **kwargs,
) -> BaseChatModel:
    """Build a configured chat model for the provider named in settings.

    Args:
        temperature: Sampling temperature. Defaults to 0.0 for deterministic,
            reproducible extraction and scoring.
        settings: Optional settings override (useful in tests). Falls back to
            the cached application settings.
        **kwargs: Extra provider-specific keyword arguments passed through to the
            underlying LangChain chat model.

    Returns:
        A LangChain ``BaseChatModel`` ready to ``.invoke()`` or wrap with
        ``.with_structured_output()``.

    Raises:
        LLMConfigError: If the provider is unsupported or its API key is missing.
    """
    settings = settings or get_settings()
    provider = settings.llm_provider.strip().lower()

    if provider not in DEFAULT_MODELS:
        raise LLMConfigError(
            f"Unsupported LLM_PROVIDER '{provider}'. "
            f"Supported: {', '.join(DEFAULT_MODELS)}."
        )

    model = settings.llm_model or DEFAULT_MODELS[provider]

    if provider == "gemini":
        return _build_gemini(model, temperature, settings, **kwargs)
    if provider == "groq":
        return _build_groq(model, temperature, settings, **kwargs)

    # Defensive: should be unreachable given the membership check above.
    raise LLMConfigError(
        f"Unsupported LLM_PROVIDER '{provider}'. "
        f"Supported: {', '.join(DEFAULT_MODELS)}."
    )


def _build_gemini(
    model: str, temperature: float, settings: Settings, **kwargs
) -> BaseChatModel:
    if not settings.google_api_key:
        raise LLMConfigError(
            "GOOGLE_API_KEY is not set. Get a free key at "
            "https://aistudio.google.com/apikey and add it to your .env file."
        )
    # Imported lazily so Groq-only deployments don't require this package.
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
        **kwargs,
    )


def _build_groq(
    model: str, temperature: float, settings: Settings, **kwargs
) -> BaseChatModel:
    if not settings.groq_api_key:
        raise LLMConfigError(
            "GROQ_API_KEY is not set. Get a free key at "
            "https://console.groq.com/keys and add it to your .env file."
        )
    # Imported lazily; langchain-groq is an optional dependency.
    try:
        from langchain_groq import ChatGroq
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise LLMConfigError(
            "langchain-groq is not installed. Run "
            "`pip install langchain-groq` to use the Groq provider."
        ) from exc

    return ChatGroq(
        model=model,
        api_key=settings.groq_api_key,
        temperature=temperature,
        **kwargs,
    )
