"""
Slack search module.

Responsible for querying Slack's search.messages API and returning a
normalized list of message dictionaries that are convenient to render and
send as context to LLMs.

Environment:
- SLACK_USER_TOKEN: xoxp- token impersonating a user (required)

Notes:
- Uses user token (not bot).
- Caches userId->username lookups to minimize API calls.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Dict, List, Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError


logger = logging.getLogger(__name__)


def _get_slack_client() -> WebClient:
    """
    Construct a Slack WebClient using the SLACK_USER_TOKEN from env.

    Raises:
        RuntimeError: If SLACK_USER_TOKEN is not provided.
    """
    token = os.getenv("SLACK_USER_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing SLACK_USER_TOKEN environment variable (must be a user xoxp- token)."
        )
    return WebClient(token=token)


def _resolve_username(client: WebClient, user_id: Optional[str], cache: Dict[str, str]) -> str:
    """
    Resolve a Slack user ID to a human-readable username, with simple caching.

    If user_id is None (e.g., message posted by an app/bot), returns "Unknown".
    """
    if not user_id:
        return "Unknown"
    if user_id in cache:
        return cache[user_id]
    try:
        resp = client.users_info(user=user_id)
        profile = resp.get("user", {})
        username = profile.get("real_name") or profile.get("name") or user_id
        cache[user_id] = username
        return username
    except SlackApiError as e:
        logger.warning("Failed to resolve username for %s: %s", user_id, e)
        return user_id


def _get_channel_id(client: WebClient, channel_name_or_id: str) -> Optional[str]:
    """
    Resolve a channel name (with or without #) to its ID. If an ID is provided,
    returns it as-is.

    Searches across public and private channels with pagination.
    """
    if not channel_name_or_id:
        return None

    # If looks like an ID already (e.g., C..., G...), accept it directly
    if channel_name_or_id.startswith(("C", "G")) and " " not in channel_name_or_id:
        return channel_name_or_id

    channel_name = channel_name_or_id.lstrip('#')

    try:
        next_cursor: Optional[str] = None
        while True:
            response = client.conversations_list(
                types="public_channel,private_channel",
                limit=1000,
                cursor=next_cursor or None,
            )
            channels = response.get("channels", [])
            for channel in channels:
                if channel.get("name") == channel_name:
                    return channel.get("id")

            next_cursor = (response.get("response_metadata") or {}).get("next_cursor") or ""
            if not next_cursor:
                break

        logger.warning(f"Channel '{channel_name}' not found")
        return None

    except SlackApiError as e:
        logger.error(f"Failed to resolve channel name '{channel_name}': {e}")
        return None


def _get_all_channels(client: WebClient, limit: int = 20) -> List[str]:
    """
    Get a list of all accessible channel IDs (public and private).
    
    Args:
        client: Slack WebClient instance
        limit: Maximum number of channels to return
    
    Returns:
        List of channel IDs
    """
    channel_ids = []
    try:
        next_cursor: Optional[str] = None
        while len(channel_ids) < limit:
            response = client.conversations_list(
                types="public_channel,private_channel",
                exclude_archived=True,
                limit=200,
                cursor=next_cursor or None,
            )
            channels = response.get("channels", [])
            
            # Filter for channels the user is a member of
            for channel in channels:
                if channel.get("is_member", False):
                    channel_ids.append(channel.get("id"))
                    if len(channel_ids) >= limit:
                        break
            
            next_cursor = (response.get("response_metadata") or {}).get("next_cursor") or ""
            if not next_cursor:
                break
        
        logger.info(f"Found {len(channel_ids)} accessible channels")
        return channel_ids[:limit]
        
    except SlackApiError as e:
        logger.error(f"Failed to get channel list: {e}")
        return []


def search_slack_recent(
    channel_name: str, 
    query: str, 
    max_results: int = 10,
    max_age_hours: int = 72
) -> List[dict]:
    """
    Fetch recent messages from Slack channels and filter by relevance.
    
    This bypasses Slack's search indexing delays and gets the latest messages.
    If no channel is specified, searches across all accessible channels.
    
    Args:
        channel_name: Slack channel name (with or without # prefix), or empty string for all channels
        query: Search query to filter messages for relevance
        max_results: Maximum number of relevant messages to return
        max_age_hours: Only consider messages from the last N hours (default 72)
    
    Returns:
        List of dictionaries with keys: text, username, channel, ts, permalink
    """
    client = _get_slack_client()
    user_cache: Dict[str, str] = {}
    channel_name_cache: Dict[str, str] = {}  # Cache channel IDs to names

    try:
        # Determine which channels to search
        channel_ids: List[str] = []
        
        if channel_name and channel_name.strip():
            # Single channel specified
            channel_id = _get_channel_id(client, channel_name.strip())
            if channel_id:
                channel_ids = [channel_id]
                channel_name_cache[channel_id] = channel_name.strip().lstrip('#')
        else:
            # No channel specified - search all accessible channels
            logger.info("No channel specified, searching all accessible channels")
            channel_ids = _get_all_channels(client, limit=20)
            
            # Get channel names for these IDs
            for cid in channel_ids:
                try:
                    info = client.conversations_info(channel=cid)
                    ch_data = info.get("channel", {})
                    channel_name_cache[cid] = ch_data.get("name", cid)
                except Exception as e:
                    logger.warning(f"Failed to get name for channel {cid}: {e}")
                    channel_name_cache[cid] = cid

        if not channel_ids:
            logger.warning("No channels to search")
            return []

        # Time window
        now_ts = time.time()
        oldest = now_ts - (max_age_hours * 3600)

        # Collect messages from all channels
        all_messages: List[dict] = []
        
        for channel_id in channel_ids:
            try:
                next_cursor: Optional[str] = None
                channel_messages: List[dict] = []
                
                while True:
                    response = client.conversations_history(
                        channel=channel_id,
                        limit=100,
                        cursor=next_cursor or None,
                        inclusive=True,
                        oldest=str(oldest)
                    )
                    batch = response.get("messages", [])
                    if not batch:
                        break
                    
                    channel_messages.extend(batch)
                    
                    # Stop if we have enough messages from this channel
                    if len(channel_messages) >= 50:
                        break
                    
                    next_cursor = (response.get("response_metadata") or {}).get("next_cursor") or ""
                    if not next_cursor:
                        break
                
                logger.info(f"Fetched {len(channel_messages)} messages from channel {channel_id}")
                all_messages.extend([{**msg, "_channel_id": channel_id} for msg in channel_messages])
                
            except SlackApiError as e:
                logger.warning(f"Failed to fetch from channel {channel_id}: {e}")
                continue

        logger.info(f"Total messages fetched: {len(all_messages)}")

        if not all_messages:
            return []

        # Filter messages for relevance
        query_terms = query.lower().split()
        relevant_messages: List[dict] = []

        for msg in all_messages:
            text = msg.get("text", "")
            if not text:
                continue
            
            # Skip system messages
            if msg.get("subtype") in ["channel_join", "channel_leave", "channel_topic", "channel_purpose"]:
                continue

            try:
                ts_val = float(msg.get("ts", "0"))
            except Exception:
                ts_val = 0.0
            if ts_val < oldest:
                continue

            # Calculate relevance score
            text_lower = text.lower()
            relevance_score = sum(1 for term in query_terms if term in text_lower) if query_terms else 1

            if relevance_score > 0:
                channel_id = msg.get("_channel_id", "unknown")
                channel_display_name = channel_name_cache.get(channel_id, channel_id)
                username = _resolve_username(client, msg.get("user"), user_cache)
                
                try:
                    permalink_resp = client.chat_getPermalink(channel=channel_id, message_ts=msg.get("ts"))
                    permalink = permalink_resp.get("permalink", "")
                except Exception:
                    permalink = ""

                relevant_messages.append({
                    "text": text,
                    "username": username,
                    "channel": channel_display_name,
                    "ts": msg.get("ts"),
                    "permalink": permalink,
                    "relevance": relevance_score,
                    "_timestamp": ts_val,
                })

        # Sort by relevance (desc) then timestamp (desc) to get most relevant recent messages
        relevant_messages.sort(key=lambda x: (-x.get("relevance", 0), -x.get("_timestamp", 0)))

        # Trim and strip helper fields
        results: List[dict] = []
        for item in relevant_messages[:max_results]:
            item.pop("relevance", None)
            item.pop("_timestamp", None)
            results.append(item)

        logger.info(f"Returning {len(results)} relevant recent messages")
        return results

    except SlackApiError as e:
        logger.error(f"Slack API error fetching recent messages: {e}")
        return []
    except Exception as e:
        logger.exception(f"Unexpected error fetching recent messages: {e}")
        return []


def search_slack(query: str, max_results: int = 20) -> List[dict]:
    """
    Search Slack messages using the search.messages API.

    Args:
        query: Free-text search query.
        max_results: Maximum number of messages to return (default 20).

    Returns:
        List of dictionaries with keys: text, username, channel, ts, permalink.

    Raises:
        RuntimeError: For missing credentials.
    """
    if not query or not query.strip():
        return []

    client = _get_slack_client()
    collected: List[dict] = []
    page = 1
    per_page = 20
    user_cache: Dict[str, str] = {}

    try:
        while len(collected) < max_results:
            resp = client.search_messages(query=query, count=per_page, page=page)
            messages = resp.get("messages", {})
            matches = messages.get("matches", [])
            if not matches:
                break

            for item in matches:
                text = item.get("text", "")
                channel_name = (item.get("channel", {}) or {}).get("name", "unknown")
                ts = item.get("ts") or (item.get("message", {}) or {}).get("ts")
                permalink = item.get("permalink")

                username = item.get("username")
                if not username:
                    username = _resolve_username(client, item.get("user"), user_cache)

                collected.append(
                    {
                        "text": text,
                        "username": username,
                        "channel": channel_name,
                        "ts": ts,
                        "permalink": permalink,
                    }
                )

                if len(collected) >= max_results:
                    break

            paging = messages.get("paging", {})
            total_pages = paging.get("pages") or 1
            page += 1
            if page > total_pages:
                break

    except SlackApiError as e:
        logger.error("Slack API error during search: %s", e)
    except Exception as e:
        logger.exception("Unexpected error during Slack search: %s", e)

    return collected[:max_results]


__all__ = ["search_slack", "search_slack_recent"]