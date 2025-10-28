"""Shared Gemini configuration utilities."""
from __future__ import annotations

import google.generativeai as genai
import streamlit as st


def configure_genai() -> None:
    """Configure the Google Generative AI client from environment."""
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY environment variable.")
    genai.configure(api_key=api_key)


__all__ = ["configure_genai"]

