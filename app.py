"""
Streamlit app for Slack + Confluence AI Search Assistant.

Main application that combines Slack and Confluence search with Gemini AI.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List
from datetime import datetime

import streamlit as st

from confluence_search import search_confluence
from gemini_handler import ask_gemini
from slack_search import search_slack, search_slack_recent


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
    """Render expandable raw source sections for Slack and Confluence."""
    with st.expander("View Slack sources"):
        if not slack_messages:
            st.info("No Slack results.")
        else:
            for m in slack_messages:
                st.markdown(
                    f"- **{m.get('channel','?')}** | @{m.get('username','?')} | ts: {m.get('ts','?')}\n\n"
                    f"{m.get('text','')}\n\n"
                    f"[Permalink]({m.get('permalink','')})"
                )

    with st.expander("View Confluence sources"):
        if not confluence_pages:
            st.info("No Confluence results.")
        else:
            for p in confluence_pages:
                st.markdown(
                    f"- **{p.get('title','Untitled')}** ({p.get('space','?')}) | last modified: {p.get('last_modified','?')}\n\n"
                    f"Excerpt: {p.get('excerpt','')}\n\n"
                    f"[Open Page]({p.get('url','')})"
                )


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
        st.header("⚙️ Search Options")
        
        st.divider()
        
        # Filters
        st.subheader("🔍 Filters")
        date_from = st.date_input("From date", value=st.session_state.get("date_from"), key="date_from")
        date_to = st.date_input("To date", value=st.session_state.get("date_to"), key="date_to")
        channel_hint = st.text_input("Slack channel", value=st.session_state.get("channel_hint", ""), 
                                     key="channel_hint", placeholder="e.g., general")
        space_hint = st.text_input("Confluence space", value=st.session_state.get("space_hint", ""), 
                                   key="space_hint", placeholder="e.g., PROJ")

        st.divider()
        
        # Context refresh behavior
        st.subheader("🔄 Context Settings")
        auto_refresh = st.checkbox(
            "Auto-refresh sources each message",
            value=True,
            help="If off, reuse last retrieved sources unless filters change.",
        )

    # Chat UI - Simple single chat without history management
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
            st.write(msg["content"])

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

            reuse_context = (not auto_refresh) and (not filters_changed) and bool(st.session_state.get("context"))

            if reuse_context:
                context = st.session_state.get("context") or ""
                slack_results = st.session_state.get("slack_results") or []
                conf_results = st.session_state.get("conf_results") or []
            else:
                with st.spinner("Retrieving sources..."):
                    # Run Slack and Confluence searches in parallel
                    slack_results: List[dict] = []
                    conf_results: List[dict]
                    
                    with ThreadPoolExecutor(max_workers=3) as pool:
                        # Always fetch recent messages first (primary source)
                        fut_slack_recent = pool.submit(
                            search_slack_recent, 
                            channel_hint.strip() if channel_hint else "",  # Empty string searches all channels
                            user_input, 
                            20,  # Get more recent messages
                            72   # Look back 72 hours (3 days)
                        )
                        
                        # Submit Confluence search
                        fut_conf = pool.submit(
                            search_confluence, 
                            user_input,
                            10, 
                            space_hint.strip() if space_hint else None
                        )
                        
                        # Collect recent Slack messages (primary source)
                        try:
                            slack_results = fut_slack_recent.result(timeout=60)
                            logger.info(f"Retrieved {len(slack_results)} recent Slack messages")
                        except Exception as e:
                            logger.error("Slack recent messages failed: %s", e)
                            slack_results = []
                        
                        # If we didn't get enough recent messages, try search API as fallback
                        if len(slack_results) < 5:
                            logger.info("Falling back to Slack search API")
                            try:
                                # Build query for search API
                                q = user_input
                                if date_from:
                                    q += f" after:{date_from.strftime('%Y-%m-%d')}"
                                if date_to:
                                    q += f" before:{date_to.strftime('%Y-%m-%d')}"
                                
                                search_results = search_slack(q, 15)
                                
                                # Merge with recent results, avoiding duplicates
                                seen_ts = {msg.get('ts') for msg in slack_results if msg.get('ts')}
                                for msg in search_results:
                                    ts = msg.get('ts')
                                    if ts and ts not in seen_ts:
                                        slack_results.append(msg)
                                        seen_ts.add(ts)
                                
                                logger.info(f"Total Slack results after fallback: {len(slack_results)}")
                            except Exception as e:
                                logger.error("Slack search API fallback failed: %s", e)
                        
                        # Collect Confluence results
                        try:
                            conf_results = fut_conf.result(timeout=60)
                        except Exception as e:
                            logger.error("Confluence search failed: %s", e)
                            conf_results = []

                    if not slack_results and not conf_results:
                        nores = "No results found in Slack or Confluence. Try adjusting filters or query."
                        st.write(nores)
                        st.session_state["chat_messages"].append({"role": "assistant", "content": nores})
                        return

                    context = _format_context(slack_results, conf_results)
                    # Persist fresh context/results and filters
                    st.session_state["context"] = context
                    st.session_state["slack_results"] = slack_results
                    st.session_state["conf_results"] = conf_results
                    st.session_state["filters"] = current_filters

            # Include short conversation memory for coherence
            history_lines = []
            for m in st.session_state["chat_messages"][-6:]:
                prefix = "User:" if m["role"] == "user" else "Assistant:"
                history_lines.append(f"{prefix} {m['content']}")
            preface = ("Previous conversation context (use for continuity):\n" + "\n".join(history_lines) + "\n\n") if history_lines else ""

            with st.spinner("Thinking..."):
                answer = ask_gemini(context=context, question=preface + user_input)
            st.write(answer)
            _render_sources(slack_results, conf_results)
            st.session_state["chat_messages"].append({"role": "assistant", "content": answer})

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