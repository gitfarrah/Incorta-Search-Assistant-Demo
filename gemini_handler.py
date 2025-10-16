
from __future__ import annotations

import logging
import time
from typing import Optional, Sequence

import google.generativeai as genai
import streamlit as st


logger = logging.getLogger(__name__)


def _configure_genai() -> None:
    """Configure the Google Generative AI client from environment."""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable.")
    genai.configure(api_key=api_key)


def _supported_models() -> list[str]:
    env_model = st.secrets.get("GEMINI_MODEL")
    if env_model:
        return [env_model]

    candidates: list[str] = [
        # Keep this short to avoid long fallback chains
        "gemini-2.5-pro-preview-03-25",
        "gemini-1.5-pro",
    ]

    try:
        # Dynamically discover models that support generateContent
        models: Sequence = genai.list_models()  # type: ignore[assignment]
        for m in models:
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                name = getattr(m, "name", "")
                # Names may be like "models/gemini-1.5-pro"; strip prefix if present
                name = name.split("/")[-1]
                if name and name not in candidates:
                    candidates.append(name)
    except Exception as e:  # noqa: BLE001 - listing may not be available
        logger.debug("Model listing not available: %s", e)

    return candidates


def ask_gemini(context: str, question: str, model_name: str = "gemini-1.5-pro") -> str:
    if not question or not question.strip():
        return "Please provide a non-empty question."

    _configure_genai()

    # If GEMINI_MODEL is set, _supported_models returns only that one
    supported = _supported_models()
    if model_name and model_name not in supported:
        candidate_models = [model_name] + [m for m in supported if m != model_name]
    else:
        candidate_models = supported

    prompt = (
        "You are an assistant answering questions using the provided enterprise context.\n"
        "Cite sources inline like [Slack #channel @user, time] or [Confluence: Page Title].\n"
        "Be concise, accurate, and clearly separate assumptions if context is insufficient.\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Context (Slack and Confluence):\n"
        f"{context}\n\n"
        "Instructions:\n- Use only the provided context unless general knowledge is unequivocally needed.\n"
        "- Prefer bullet points.\n- Summarize conflicting info and note uncertainties.\n"
    )

    last_error: Optional[Exception] = None

    for name in candidate_models:
        try:
            logger.info("Trying Gemini model: %s", name)
            # Some SDKs accept plain name; some expect full path without prefix here
            model = genai.GenerativeModel(name)
            # Minimal retry to reduce latency
            for attempt in range(1):
                try:
                    resp = model.generate_content(prompt)
                    text = getattr(resp, "text", None)
                    if text:
                        return text.strip()
                    if getattr(resp, "candidates", None):
                        parts = resp.candidates[0].content.parts  # type: ignore[attr-defined]
                        joined = "".join(getattr(p, "text", "") for p in parts)
                        if joined:
                            return joined.strip()
                    return "(No response text from Gemini)"
                except Exception as e:  # noqa: BLE001
                    last_error = e
                    logger.warning("Attempt %d failed: %s", attempt + 1, e)
                    # No additional backoff when attempts == 1
        except Exception as e:  # noqa: BLE001
            last_error = e
            logger.debug("Model init failed for %s: %s", name, e)
            continue

    logger.error("All Gemini attempts failed. Last error: %s", last_error)
    return (
        "There was an error generating the answer. Please try again later."
    )


__all__ = ["ask_gemini"]