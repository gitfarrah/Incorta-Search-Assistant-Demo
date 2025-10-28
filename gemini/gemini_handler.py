from __future__ import annotations

import logging
import time
from typing import Optional, Sequence, List, Dict

import google.generativeai as genai
import streamlit as st

from .gemini_config import configure_genai

logger = logging.getLogger(__name__)


def _supported_models() -> list[str]:
    env_model = st.secrets.get("GEMINI_MODEL")
    if env_model:
        return [env_model]

    candidates: list[str] = [
        "gemini-2.0-flash",
        "gemini-2.0-flash-exp",
        "gemini-2.5-pro",
        "gemini-flash-2.5-pro",
        "gemini-2.5-pro-preview-03-25",
        "gemini-1.5-pro",
    ]

    try:
        models: Sequence = genai.list_models()  # type: ignore[assignment]
        for m in models:
            methods = getattr(m, "supported_generation_methods", []) or []
            if "generateContent" in methods:
                name = getattr(m, "name", "")
                name = name.split("/")[-1]
                if name and name not in candidates:
                    candidates.append(name)
    except Exception as e:  # noqa: BLE001
        logger.debug("Model listing not available: %s", e)

    return candidates


def build_enhanced_prompt(
    context: str,
    question: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    query_type: str = "new_search"
) -> str:
    """
    Build a concise prompt that produces short, clear responses.
    """
    
    # Build conversation context
    conv_context = ""
    if conversation_history and len(conversation_history) > 1:
        recent_conv = conversation_history[-4:]  # Last 2 exchanges
        conv_lines = []
        for msg in recent_conv:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:150]  # Truncate long messages
            conv_lines.append(f"{role}: {content}")
        conv_context = "Previous conversation:\n" + "\n".join(conv_lines) + "\n\n"
    
    # Simplified instructions based on query type
    if query_type in ["follow_up", "clarification"]:
        instructions = """Answer concisely based on our previous conversation. No need to re-cite sources."""
    else:
        instructions = """Answer the question directly and concisely using the provided context."""
    
    prompt = f"""{instructions}

{conv_context}Question: {question}

Context from Slack and Confluence:
{context}

CRITICAL RULES:
- Be brief and direct (2-4 short paragraphs maximum)
- Start with the direct answer immediately
- Use simple, clear language
- No elaborate formatting or structure
- Do NOT include inline citations, parenthetical references, or page titles in your answer
- Do NOT add source names like "(Product Initiative: ...)" or "(MCP Server Architecture...)"
- Present information naturally without citing which document it came from
- If information is insufficient, state what's missing briefly
- ALWAYS use exact dates/timestamps from the context - do NOT interpret or convert them
- When citing Slack messages, use the exact date format provided in the context

Answer:"""
    
    return prompt


def ask_gemini(
    context: str,
    question: str,
    model_name: str = "gemini-2.0-flash-exp",
    conversation_history: Optional[List[Dict[str, str]]] = None,
    query_type: str = "new_search"
) -> str:
    """
    Enhanced Gemini query with conversation awareness and query type handling.
    """
    
    if not question or not question.strip():
        return "Please provide a non-empty question."

    configure_genai()

    supported = _supported_models()
    if model_name and model_name not in supported:
        candidate_models = [model_name] + [m for m in supported if m != model_name]
    else:
        candidate_models = supported

    # Build enhanced prompt
    prompt = build_enhanced_prompt(context, question, conversation_history, query_type)
    
    last_error: Optional[Exception] = None

    for name in candidate_models:
        try:
            logger.info(f"Trying Gemini model: {name}")
            model = genai.GenerativeModel(name)
            
            # Configure generation for concise responses
            generation_config = {
                "temperature": 0.7,  # Balance creativity and accuracy
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 800,  # Reduced for concise responses
            }
            
            # For follow-ups, be more conversational but still concise
            if query_type in ["follow_up", "clarification"]:
                generation_config["temperature"] = 0.8
                generation_config["max_output_tokens"] = 600  # Even shorter for follow-ups
            
            for attempt in range(2):  # Allow one retry
                try:
                    resp = model.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                    result_text = (getattr(resp, "text", None) or "").strip()
                    if result_text:
                        return result_text
                except Exception as e:  # noqa: BLE001
                    last_error = e
                    time.sleep(0.2)
            # If both attempts failed for this model, try the next candidate
            continue
        
        except Exception as e:  # noqa: BLE001
            last_error = e
            logger.warning(f"Model {name} failed with error: {e}")
            continue
    
    # If all models fail, provide a graceful fallback
    logger.error(f"All Gemini model attempts failed. Last error: {last_error}")
    return "Sorry, I couldn't generate a response right now. Please try again in a moment."