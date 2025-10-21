from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from typing import List
from datetime import datetime

import streamlit as st

from confluence_search import search_confluence, search_confluence_optimized
from gemini_handler import ask_gemini
from slack_search import search_slack, search_slack_recent, search_slack_optimized
from intent_analyzer import analyze_user_intent, validate_intent
from cache_manager import (
    cache_intent_analysis, get_cached_intent_analysis,
    cache_search_results, get_cached_search_results,
    get_cache_manager
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Credentials are provided via st.secrets in Streamlit Cloud/local


def _validate_env() -> List[str]:
    """Return a list of missing required environment variables."""
    missing = []
    if not st.secrets.get("SLACK_USER_TOKEN"):
        missing.append("SLACK_USER_TOKEN")
    if not st.secrets.get("CONFLUENCE_URL"):
        missing.append("CONFLUENCE_URL")
    if not st.secrets.get("CONFLUENCE_EMAIL"):
        missing.append("CONFLUENCE_EMAIL")
    if not st.secrets.get("CONFLUENCE_API_TOKEN"):
        missing.append("CONFLUENCE_API_TOKEN")
    if not st.secrets.get("GEMINI_API_KEY"):
        missing.append("GEMINI_API_KEY")
    return missing


def _format_context(slack_messages: List[dict], confluence_pages: List[dict]) -> str:
    """Create a single context string with clear headers for Gemini."""
    parts: List[str] = []

    if slack_messages:
        parts.append("=== Slack Messages ===")
        for i, m in enumerate(slack_messages, start=1):
            meta = f"[# {m.get('channel','?')} | @{m.get('username','?')} | ts: {m.get('ts','?')}]"
            line = f"{i}. {meta}\n{m.get('text','').strip()}\nSource: {m.get('permalink','')}\n"
            parts.append(line)

    if confluence_pages:
        parts.append("=== Confluence Pages ===")
        for i, p in enumerate(confluence_pages, start=1):
            meta = f"[{p.get('space','?')} | last modified: {p.get('last_modified','?')}]"
            excerpt = (p.get('excerpt') or '').strip()
            line = (
                f"{i}. {p.get('title','Untitled')} {meta}\n"
                f"Excerpt: {excerpt}\nSource: {p.get('url','')}\n"
            )
            parts.append(line)

    return "\n".join(parts)


def _render_sources(slack_messages: List[dict], confluence_pages: List[dict]) -> None:
    """Render expandable raw source sections for Slack and Confluence with improved formatting."""
    
    # Slack Sources
    with st.expander(f"Slack Messages ({len(slack_messages)} found)", expanded=False):
        if not slack_messages:
            st.info("No Slack results found.")
        else:
            for idx, m in enumerate(slack_messages, 1):
                channel = m.get('channel', 'Unknown')
                username = m.get('username', 'Unknown')
                timestamp = m.get('ts', '')
                text = m.get('text', '').strip()
                permalink = m.get('permalink', '')
                
                # Create a clean card-like display for each message
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #4A90E2;">
                    <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                        <span style="font-weight: 600; color: #1f1f1f;">#{channel}</span>
                        <span style="color: #666; font-size: 0.9em;">{timestamp}</span>
                    </div>
                    <div style="color: #4A90E2; font-size: 0.9em; margin-bottom: 10px;">@{username}</div>
                    <div style="color: #1f1f1f; line-height: 1.6; margin-bottom: 10px;">{text}</div>
                    <a href="{permalink}" target="_blank" style="color: #4A90E2; text-decoration: none; font-size: 0.9em;">View in Slack</a>
                </div>
                """, unsafe_allow_html=True)

    # Confluence Sources
    with st.expander(f"Confluence Pages ({len(confluence_pages)} found)", expanded=False):
        if not confluence_pages:
            st.info("No Confluence results found.")
        else:
            for idx, p in enumerate(confluence_pages, 1):
                title = p.get('title', 'Untitled')
                space = p.get('space', 'Unknown')
                last_modified = p.get('last_modified', 'Unknown')
                excerpt = (p.get('excerpt') or '').strip()
                url = p.get('url', '')
                
                # Clean up excerpt by removing HTML highlighting tags and truncating intelligently
                import re
                cleaned_excerpt = re.sub(r'@@@hl@@@(.*?)@@@endhl@@@', r'\1', excerpt)
                cleaned_excerpt = re.sub(r'<[^>]+>', '', cleaned_excerpt)  # Remove any remaining HTML tags
                cleaned_excerpt = cleaned_excerpt.replace('&nbsp;', ' ').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                
                # Truncate to reasonable length and add ellipsis
                if len(cleaned_excerpt) > 200:
                    cleaned_excerpt = cleaned_excerpt[:200].rsplit(' ', 1)[0] + '...'
                
                # Create a clean card-like display for each page
                st.markdown(f"""
                <div style="background-color: #f0f2f6; padding: 15px; border-radius: 8px; margin-bottom: 15px; border-left: 4px solid #6B46C1;">
                    <div style="font-weight: 600; color: #1f1f1f; font-size: 1.1em; margin-bottom: 8px;">{title}</div>
                    <div style="display: flex; gap: 15px; margin-bottom: 10px; font-size: 0.9em;">
                        <span style="color: #6B46C1; font-weight: 500;">{space}</span>
                        <span style="color: #666;">{last_modified}</span>
                    </div>
                    <div style="color: #444; line-height: 1.6; margin-bottom: 10px; font-style: italic;">{cleaned_excerpt}</div>
                    <a href="{url}" target="_blank" style="color: #6B46C1; text-decoration: none; font-size: 0.9em;">Open Page</a>
                </div>
                """, unsafe_allow_html=True)


def main() -> None:
    # Page config and header
    st.set_page_config(page_title="Incorta AI Search Assistant", page_icon="incorta.png", layout="wide")
    st.title("Incorta AI Search Assistant Demo")
    st.caption("Searches Slack and Confluence, then asks Gemini to synthesize an answer with citations.")

    # Env validation
    missing = _validate_env()
    if missing:
        st.warning("Missing environment variables: " + ", ".join(missing) + ". Set them to enable full functionality.")

    # Initialize session state
    if "search_history" not in st.session_state:
        st.session_state["search_history"] = []

    # Sidebar options
    with st.sidebar:
        st.markdown("<h1 style=\"margin-top:0; margin-bottom:0.25rem; font-size:1.6rem;\">Search Options</h1>", unsafe_allow_html=True)
        
        st.divider()
        
        # Filters
        st.subheader("Filters")
        date_from = st.date_input("From date", value=st.session_state.get("date_from"), key="date_from")
        date_to = st.date_input("To date", value=st.session_state.get("date_to"), key="date_to")
        channel_hint = st.text_input("Slack channel", value=st.session_state.get("channel_hint", ""), 
                                     key="channel_hint", placeholder="e.g., general, engineering")
        space_hint = st.text_input("Confluence space", value=st.session_state.get("space_hint", ""), 
                                   key="space_hint", placeholder="e.g., PROJ, DOCS")

        st.divider()
        # Context refresh is always enabled (UI control removed)
        auto_refresh = True
        
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = [
            {"role": "assistant", "content": "Hi! Ask me anything. I'll search Slack and Confluence, then summarize with sources."}
        ]
    
    if "context" not in st.session_state:
        st.session_state["context"] = None
    if "slack_results" not in st.session_state:
        st.session_state["slack_results"] = []
    if "conf_results" not in st.session_state:
        st.session_state["conf_results"] = []
    if "filters" not in st.session_state:
        st.session_state["filters"] = {}

    # Render chat history
    for msg in st.session_state["chat_messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Type your question")

    if user_input:
        # Echo user message
        st.session_state["chat_messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            # Detect filter changes to decide reuse
            current_filters = {
                "channel_hint": (channel_hint or "").strip(),
                "space_hint": (space_hint or "").strip(),
                "date_from": date_from,
                "date_to": date_to,
            }
            filters_changed = current_filters != st.session_state.get("filters", {})

            # Don't reuse context for latest_message intent to ensure fresh data
            intent_type = ""
            if "intent_data" in st.session_state:
                intent_type = st.session_state["intent_data"].get("intent", "")
            
            reuse_context = (not auto_refresh) and (not filters_changed) and bool(st.session_state.get("context")) and (intent_type != "latest_message")

            if reuse_context:
                context = st.session_state.get("context") or ""
                slack_results = st.session_state.get("slack_results") or []
                conf_results = st.session_state.get("conf_results") or []
            else:
                with st.spinner("Analyzing query and retrieving sources..."):
                    # Step 1: Check cache for intent analysis
                    cache_filters = {
                        "channel_hint": (channel_hint or "").strip(),
                        "space_hint": (space_hint or "").strip(),
                        "date_from": date_from,
                        "date_to": date_to,
                    }
                    
                    intent_data = get_cached_intent_analysis(user_input, cache_filters)
                    
                    if not intent_data:
                        # Analyze user intent if not cached
                        intent_data = analyze_user_intent(user_input)
                        intent_data = validate_intent(intent_data)
                        cache_intent_analysis(user_input, cache_filters, intent_data)
                    
                    # Override with UI filters if provided
                    if channel_hint and channel_hint.strip():
                        intent_data["slack_params"]["channels"] = channel_hint.strip()
                    if space_hint and space_hint.strip():
                        intent_data["confluence_params"]["spaces"] = space_hint.strip()
                    if date_from:
                        intent_data["slack_params"]["date_from"] = date_from.strftime("%Y-%m-%d")
                    if date_to:
                        intent_data["slack_params"]["date_to"] = date_to.strftime("%Y-%m-%d")

                    # Normalize Slack channel parameter if AI returned a placeholder or empty
                    try:
                        ch_val = (intent_data.get("slack_params", {}) or {}).get("channels")
                        if not ch_val or str(ch_val).strip().lower() in {"specific_channel_name", "channel_name", "none", ""}:
                            # Try to extract channel from user query (e.g., #engineering or "engineering channel")
                            import re as _re
                            extracted = None
                            m = _re.search(r"#(\w+)", user_input)
                            if m:
                                extracted = m.group(1)
                            else:
                                m2 = _re.search(r"(?:in|on)\s+the\s+(\w+)\s+channel", user_input, _re.IGNORECASE)
                                if m2:
                                    extracted = m2.group(1)
                                else:
                                    m3 = _re.search(r"\b(\w+)\s+channel\b", user_input, _re.IGNORECASE)
                                    if m3:
                                        extracted = m3.group(1)
                            if extracted:
                                intent_data["slack_params"]["channels"] = extracted
                            else:
                                # Fallback to search across all accessible channels
                                intent_data["slack_params"]["channels"] = "all"
                    except Exception as _e:
                        logger.debug(f"Channel normalization skipped: {_e}")
                    
                    logger.info(f"Intent analysis result: {intent_data}")
                    
                    # Step 1.5: Attempt to reuse cached search results using intent signature
                    try:
                        intent_sig_parts = {
                            "slack_channels": intent_data.get("slack_params", {}).get("channels"),
                            "slack_time_range": intent_data.get("slack_params", {}).get("time_range"),
                            "slack_keywords": tuple(sorted((intent_data.get("slack_params", {}).get("keywords") or []))),
                            "slack_priority": tuple(sorted((intent_data.get("slack_params", {}).get("priority_terms") or []))),
                            "conf_spaces": intent_data.get("confluence_params", {}).get("spaces"),
                            "conf_keywords": tuple(sorted((intent_data.get("confluence_params", {}).get("keywords") or []))),
                            "conf_priority": tuple(sorted((intent_data.get("confluence_params", {}).get("priority_terms") or []))),
                            "strategy": intent_data.get("search_strategy"),
                        }
                    except Exception:
                        intent_sig_parts = {"fallback": True}

                    search_cache_filters = {**cache_filters, "intent_sig": str(sorted(intent_sig_parts.items()))}

                    cached_results = get_cached_search_results(user_input, search_cache_filters)
                    if cached_results:
                        slack_results = cached_results.get("slack_results", [])
                        conf_results = cached_results.get("conf_results", [])
                        logger.info("Using cached search results")
                    else:
                        # Step 2: Run optimized searches in parallel
                        slack_results: List[dict] = []
                        conf_results: List[dict] = []
                        
                        data_sources = intent_data.get("data_sources", ["slack", "confluence"])
                        
                        with ThreadPoolExecutor(max_workers=3) as pool:
                            futures = {}
                            
                            # Submit Slack search if needed
                            if "slack" in data_sources:
                                futures["slack"] = pool.submit(
                                    search_slack_optimized,
                                    intent_data,
                                    user_input
                                )
                            
                            # Submit Confluence search if needed
                            if "confluence" in data_sources:
                                futures["confluence"] = pool.submit(
                                    search_confluence_optimized,
                                    intent_data,
                                    user_input
                                )
                            
                            # Collect results with better error handling
                            for source, future in futures.items():
                                try:
                                    if source == "slack":
                                        slack_results = future.result(timeout=30)
                                        logger.info(f"Retrieved {len(slack_results)} Slack messages")
                                    elif source == "confluence":
                                        conf_results = future.result(timeout=30)
                                        logger.info(f"Retrieved {len(conf_results)} Confluence pages")
                                    
                                except Exception as e:
                                    error_msg = f"{source.title()} search failed: {str(e)}"
                                    logger.error(error_msg)
                                    
                                    # Show user-friendly error message
                                    if "timeout" in str(e).lower():
                                        st.warning(f"{source.title()} search timed out. This might be due to high load.")
                                    elif "permission" in str(e).lower() or "unauthorized" in str(e).lower():
                                        st.warning(f"{source.title()} access denied. Check your permissions.")
                                    elif "rate limit" in str(e).lower():
                                        st.warning(f"{source.title()} rate limit exceeded. Please wait a moment.")
                                    else:
                                        st.warning(f"{source.title()} search encountered an issue: {str(e)[:100]}...")
                                    
                                    if source == "slack":
                                        slack_results = []
                                    elif source == "confluence":
                                        conf_results = []

                        # Cache fresh search results
                        try:
                            cache_search_results(user_input, search_cache_filters, {
                                "slack_results": slack_results,
                                "conf_results": conf_results
                            })
                        except Exception as e:
                            logger.debug(f"Skipping cache of search results: {e}")

                    if not slack_results and not conf_results:
                        nores = "No results found in Slack or Confluence. Try adjusting your query or filters."
                        st.write(nores)
                        st.session_state["chat_messages"].append({"role": "assistant", "content": nores})
                        return

                    context = _format_context(slack_results, conf_results)
                    # Persist fresh context/results and filters
                    st.session_state["context"] = context
                    st.session_state["slack_results"] = slack_results
                    st.session_state["conf_results"] = conf_results
                    st.session_state["filters"] = current_filters
                    st.session_state["intent_data"] = intent_data

            # Include short conversation memory for coherence
            history_lines = []
            for m in st.session_state["chat_messages"][-6:]:
                prefix = "User:" if m["role"] == "user" else "Assistant:"
                history_lines.append(f"{prefix} {m['content']}")
            preface = ("Previous conversation context (use for continuity):\n" + "\n".join(history_lines) + "\n\n") if history_lines else ""

            # Stream response with write_stream
            response_placeholder = st.empty()
            full_response = ""
            
            with st.spinner("Thinking..."):
                answer = ask_gemini(context=context, question=preface + user_input)
            
            # Stream the response character by character for better UX
            for char in answer:
                full_response += char
                response_placeholder.markdown(full_response + "▌")
                time.sleep(0.01)  # Small delay for streaming effect
            
            # Final response without cursor
            response_placeholder.markdown(full_response)
            
            # Show expandable sources
            _render_sources(slack_results, conf_results)
            
            # Store response in chat history
            st.session_state["chat_messages"].append({"role": "assistant", "content": full_response})

            # Save to search history (prepend)
            try:
                new_entry = {
                    "question": user_input.strip(),
                    "channel_hint": (channel_hint or "").strip() or None,
                    "space_hint": (space_hint or "").strip() or None,
                    "date_from": date_from,
                    "date_to": date_to,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                }
                if not st.session_state["search_history"] or st.session_state["search_history"][0].get("question") != new_entry["question"]:
                    st.session_state["search_history"].insert(0, new_entry)
                    if len(st.session_state["search_history"]) > 50:
                        st.session_state["search_history"] = st.session_state["search_history"][:50]
            except Exception as e:
                logger.warning("Failed to append to search history: %s", e)


if __name__ == "__main__":
    main()