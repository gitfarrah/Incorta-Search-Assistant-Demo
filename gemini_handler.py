from __future__ import annotations

import logging
import time
from typing import Optional, Sequence, List, Dict

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
    Build an enhanced prompt that helps Gemini understand:
    1. Whether this is a follow-up or new question
    2. How to use conversation history
    3. When to cite sources vs use prior knowledge
    """
    
    # Build conversation context
    conv_context = ""
    if conversation_history and len(conversation_history) > 1:
        recent_conv = conversation_history[-6:]  # Last 3 exchanges
        conv_lines = []
        for msg in recent_conv:
            role = "User" if msg["role"] == "user" else "Assistant"
            content = msg["content"][:200]  # Truncate long messages
            conv_lines.append(f"{role}: {content}")
        conv_context = "=== Recent Conversation ===\n" + "\n".join(conv_lines) + "\n\n"
    
    # Determine instructions based on query type
    if query_type == "follow_up":
        instructions = """
This is a FOLLOW-UP question about information already provided in the conversation.

IMPORTANT INSTRUCTIONS:
- Reference and elaborate on information from your previous responses
- You do NOT need to cite sources for information you already provided
- Be conversational and build upon what you've already explained
- Only use new sources from the context if they add genuinely new information
- If the user asks for clarification, explain more clearly without re-searching
"""
    elif query_type == "clarification":
        instructions = """
This is a CLARIFICATION request about your previous response.

IMPORTANT INSTRUCTIONS:
- Explain the same information in a different way or with more detail
- No need to cite sources again unless providing NEW information
- Be patient and thorough in your explanation
- Focus on making the information clearer and more accessible
"""
    else:  # new_search
        instructions = """
This is a NEW question requiring fresh information from the sources.

IMPORTANT INSTRUCTIONS:
- Analyze the provided context carefully
- Do NOT include inline citations within sentences
- Do NOT include a Sources section at the end
- Synthesize information from multiple sources when relevant
- If context is insufficient, clearly state what's missing
- Distinguish between what the sources say and general knowledge
- Write clean, readable text without citation clutter
"""
    
    prompt = f"""{instructions}

{conv_context}Current Question:
{question}

=== Retrieved Context (Slack and Confluence) ===
{context}

RESPONSE GUIDELINES:
1. Write a comprehensive response with clear structure and formatting
2. Start with a brief introductory paragraph
3. Use "Here's a breakdown based on the provided context:" as a transition
4. Present key points with bold headings and bullet points
5. Do NOT include inline citations within sentences
6. Do NOT include a Sources section at the end
7. Write clean, readable text without citation clutter

Response Structure:
- Brief introductory paragraph (2-3 sentences)
- "Here's a breakdown based on the provided context:"
- **Bold Heading:** Description
- **Bold Heading:** Description
- Continue with additional points as needed
- End with suggestions for follow-up questions
- NO Sources section

Tone:
- Professional, descriptive, and engaging
- Explain the "why" and "how" behind information
- Include competitive advantages and strategic context when relevant
- Acknowledge uncertainty if context is incomplete
- Suggest follow-up questions to get more complete information

If follow-up:
- Build on previous explanation naturally
- Reference previous points without re-citing unless adding new info

Conflicts:
- Present both sides with detailed explanation
- Provide your assessment with reasoning

Now, please answer the question. If there is any retrieved context above, do not say that no information was found. If context is sparse, explain what's missing and suggest a follow-up query or filter.
"""
    
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

    _configure_genai()

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
            
            # Configure generation for better responses
            generation_config = {
                "temperature": 0.7,  # Balance creativity and accuracy
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 2048,
            }
            
            # For follow-ups, be more conversational
            if query_type in ["follow_up", "clarification"]:
                generation_config["temperature"] = 0.8
            
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