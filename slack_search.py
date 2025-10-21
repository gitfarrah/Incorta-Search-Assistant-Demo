from __future__ import annotations

import logging
import time
import re
from typing import Dict, List, Optional

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import streamlit as st


logger = logging.getLogger(__name__)


def _get_slack_client() -> WebClient:
    token = st.secrets.get("SLACK_USER_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing SLACK_USER_TOKEN environment variable (must be a user xoxp- token)."
        )
    return WebClient(token=token)


def _resolve_username(client: WebClient, user_id: Optional[str], cache: Dict[str, str]) -> str:
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
    if not channel_name_or_id:
        return None

    if channel_name_or_id.startswith(("C", "G")) and " " not in channel_name_or_id:
        return channel_name_or_id

    channel_name = channel_name_or_id.lstrip('#')
    # Early return for placeholder values to avoid wasted API calls
    if channel_name.lower() in {"specific_channel_name", "channel_name", "none", ""}:
        return None

    logger.info(f"Looking for channel: '{channel_name}'")
    
    try:
        next_cursor: Optional[str] = None
        found_channels = []
        
        while True:
            response = client.conversations_list(
                types="public_channel,private_channel",
                limit=1000,
                cursor=next_cursor or None,
            )
            channels = response.get("channels", [])
            
            for channel in channels:
                ch_name = channel.get("name", "")
                ch_id = channel.get("id", "")
                found_channels.append(f"{ch_name} ({ch_id})")
                
                if ch_name == channel_name:
                    logger.info(f"Found channel '{channel_name}' with ID: {ch_id}")
                    return ch_id

            next_cursor = (response.get("response_metadata") or {}).get("next_cursor") or ""
            if not next_cursor:
                break

        logger.warning(f"Channel '{channel_name}' not found. Available channels: {found_channels[:10]}...")
        return None

    except SlackApiError as e:
        logger.error(f"Failed to resolve channel name '{channel_name}': {e}")
        return None


def _get_all_channels(client: WebClient, limit: int = 200) -> List[str]:  # Increased limit significantly
    """Get all accessible channels, excluding DMs."""
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
            
            for channel in channels:
                channel_id = channel.get("id")
                is_member = channel.get("is_member", False)
                is_private = channel.get("is_private", False)
                is_im = channel.get("is_im", False)
                is_mpim = channel.get("is_mpim", False)
                
                if not is_im and not is_mpim and (not is_private or is_member):
                    channel_ids.append(channel_id)
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


def calculate_message_relevance(
    text: str,
    keywords: List[str],
    priority_terms: List[str],
    search_strategy: str = "fuzzy_match"
) -> float:
    """
    Calculate relevance score for a message based on keywords and strategy.
    
    Returns a score between 0 and 100.
    """
    if not text:
        return 0.0
    
    text_lower = text.lower()
    score = 0.0
    
    # Priority terms MUST match for exact_match strategy
    if search_strategy == "exact_match" and priority_terms:
        priority_matches = sum(1 for term in priority_terms if term.lower() in text_lower)
        if priority_matches == 0:
            return 0.0  # No match on priority terms = not relevant
        score += priority_matches * 20  # High weight for priority terms
    elif priority_terms:
        # For other strategies, priority terms still matter but aren't blockers
        priority_matches = sum(1 for term in priority_terms if term.lower() in text_lower)
        score += priority_matches * 15
    
    # Keyword matching
    if keywords:
        keyword_matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        score += keyword_matches * 5
        
        # Bonus for keyword density
        if keyword_matches > 0:
            keyword_density = keyword_matches / len(keywords)
            score += keyword_density * 10
    
    # Exact phrase matching bonus
    for kw in keywords + priority_terms:
        if len(kw) > 3 and re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower):
            score += 3  # Bonus for word boundary matches
    
    # Technical pattern bonuses (version numbers, IDs, etc.)
    technical_patterns = [
        r'\bv?\d+\.\d+(?:\.\d+)?(?:\.\d+)?\b',  # Version numbers
        r'\b[A-Z]+-\d+\b',  # JIRA-style IDs
        r'\b[A-Z]{2,}\d+\b',  # Product codes
    ]
    for pattern in technical_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score += 5
    
    # Semantic context bonuses - look for related concepts and synonyms
    semantic_bonuses = {
        # Architecture and technical concepts
        'architecture': ['design', 'structure', 'components', 'system', 'framework', 'infrastructure'],
        'server': ['service', 'backend', 'api', 'endpoint', 'host'],
        'integration': ['connect', 'link', 'bridge', 'interface', 'combine'],
        'authentication': ['auth', 'login', 'security', 'access', 'credentials'],
        'data': ['information', 'analytics', 'insights', 'metrics', 'database'],
        
        # Development and deployment
        'development': ['dev', 'build', 'create', 'develop', 'coding'],
        'deployment': ['deploy', 'launch', 'release', 'publish', 'go-live'],
        'testing': ['test', 'qa', 'validate', 'verify', 'check'],
        
        # Business and project terms
        'project': ['initiative', 'effort', 'work', 'task', 'undertaking'],
        'team': ['group', 'crew', 'staff', 'members', 'people'],
        'meeting': ['call', 'session', 'discussion', 'conversation', 'chat'],
    }
    
    # Apply semantic bonuses based on keyword context
    for primary_term, synonyms in semantic_bonuses.items():
        if any(primary_term in kw.lower() for kw in keywords + priority_terms):
            for synonym in synonyms:
                if synonym in text_lower:
                    score += 3  # Moderate bonus for semantic matches
    
    # Length penalty for very short messages (likely noise)
    if len(text.strip()) < 20:
        score *= 0.5
    
    # Cap score at 100
    return min(score, 100.0)


def search_slack_recent(
    channel_name: str, 
    query: str, 
    max_results: int = 10,
    max_age_hours: int = 72,
    keywords: Optional[List[str]] = None,
    priority_terms: Optional[List[str]] = None,
    search_strategy: str = "fuzzy_match"
) -> List[dict]:
    """Enhanced recent message search with smart relevance scoring."""
    
    client = _get_slack_client()
    user_cache: Dict[str, str] = {}
    channel_name_cache: Dict[str, str] = {}
    
    keywords = keywords or []
    priority_terms = priority_terms or []

    try:
        # Determine channels to search
        channel_ids: List[str] = []
        
        if channel_name and channel_name.strip():
            channel_id = _get_channel_id(client, channel_name.strip())
            if channel_id:
                channel_ids = [channel_id]
                channel_name_cache[channel_id] = channel_name.strip().lstrip('#')
        else:
            logger.info("Searching all accessible channels")
            channel_ids = _get_all_channels(client, limit=200)  # Search ALL accessible channels
            
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

        # Time window - only apply if not searching all history
        now_ts = time.time()
        if max_age_hours > 0:  # 0 means search all history
            oldest = now_ts - (max_age_hours * 3600)
        else:
            oldest = 0  # Search from beginning of time

        # Collect messages
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
                    
                    if len(channel_messages) >= 50:
                        break
                    
                    next_cursor = (response.get("response_metadata") or {}).get("next_cursor") or ""
                    if not next_cursor:
                        break
                
                logger.info(f"Fetched {len(channel_messages)} messages from {channel_id}")
                all_messages.extend([{**msg, "_channel_id": channel_id} for msg in channel_messages])
                
            except SlackApiError as e:
                logger.warning(f"Failed to fetch from channel {channel_id}: {e}")
                continue

        logger.info(f"Total messages fetched: {len(all_messages)}")

        if not all_messages:
            return []

        # Score and filter messages
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
            relevance_score = calculate_message_relevance(
                text, keywords, priority_terms, search_strategy
            )

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

        # Sort by relevance then timestamp
        relevant_messages.sort(key=lambda x: (-x.get("relevance", 0), -x.get("_timestamp", 0)))

        # Clean and return top results
        results: List[dict] = []
        for item in relevant_messages[:max_results]:
            item.pop("relevance", None)
            item.pop("_timestamp", None)
            results.append(item)

        logger.info(f"Returning {len(results)} relevant messages")
        return results

    except SlackApiError as e:
        logger.error(f"Slack API error: {e}")
        return []
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return []


def search_slack(query: str, max_results: int = 20) -> List[dict]:
    """Basic Slack search using search API."""
    
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

                collected.append({
                    "text": text,
                    "username": username,
                    "channel": channel_name,
                    "ts": ts,
                    "permalink": permalink,
                })

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


def search_slack_optimized(
    intent_data: dict,
    user_query: str
) -> List[dict]:
    """Optimized Slack search using enhanced intent analysis."""
    
    client = _get_slack_client()
    user_cache: Dict[str, str] = {}
    channel_name_cache: Dict[str, str] = {}
    
    slack_params = intent_data.get("slack_params", {})
    channels = slack_params.get("channels", "all")
    time_range = slack_params.get("time_range", "30d")
    keywords = slack_params.get("keywords", [])
    priority_terms = slack_params.get("priority_terms", [])
    limit = slack_params.get("limit", 10)
    sort = slack_params.get("sort", "relevance")
    search_strategy = intent_data.get("search_strategy", "fuzzy_match")
    
    try:
        # Determine channels
        channel_ids: List[str] = []
        
        if channels == "all":
            channel_ids = _get_all_channels(client, limit=200)  # Search ALL accessible channels
        elif isinstance(channels, str):
            channel_id = _get_channel_id(client, channels)
            if channel_id:
                channel_ids = [channel_id]
                channel_name_cache[channel_id] = channels.lstrip('#')
        elif isinstance(channels, list):
            for channel_name in channels:
                channel_id = _get_channel_id(client, channel_name)
                if channel_id:
                    channel_ids.append(channel_id)
                    channel_name_cache[channel_id] = channel_name.lstrip('#')
        
        if not channel_ids:
            logger.warning("No accessible channels found")
            return []
        
        # Handle latest_message intent or specific channel queries
        intent_type = (intent_data.get("intent") or "").lower()
        user_query_lower = user_query.lower()
        
        # Check if user is asking for a specific channel
        is_specific_channel_query = any([
            "channel" in user_query_lower and any(ch in user_query_lower for ch in ["incorta-kudos", "incorta_kudos", "kudos"]),
            "in #" in user_query_lower,
            "from #" in user_query_lower,
            intent_type == "latest_message"
        ])
        
        if is_specific_channel_query:
            # For specific channel queries, search each channel individually
            latest_results: List[dict] = []
            
            # If user mentioned a specific channel, prioritize that channel
            target_channel = None
            if "incorta-kudos" in user_query_lower or "incorta_kudos" in user_query_lower:
                target_channel = "incorta-kudos"
            elif "kudos" in user_query_lower:
                target_channel = "kudos"
            
            # Search channels in order of priority
            search_channels = []
            if target_channel:
                # Find the target channel first
                for cid in channel_ids:
                    try:
                        info = client.conversations_info(channel=cid)
                        ch_name = (info.get("channel", {}) or {}).get("name", "")
                        if target_channel in ch_name.lower():
                            search_channels.insert(0, cid)  # Prioritize target channel
                            logger.info(f"Found target channel: {ch_name} ({cid})")
                        else:
                            search_channels.append(cid)
                    except Exception:
                        search_channels.append(cid)
            else:
                search_channels = channel_ids
            
            for channel_id in search_channels:
                try:
                    # Get more messages for better results
                    resp = client.conversations_history(channel=channel_id, limit=10, inclusive=True)
                    msgs = resp.get("messages", [])
                    if not msgs:
                        continue
                    
                    # Get channel name
                    channel_name = channel_name_cache.get(channel_id)
                    if not channel_name:
                        try:
                            info = client.conversations_info(channel=channel_id)
                            channel_name = (info.get("channel", {}) or {}).get("name", channel_id)
                        except Exception:
                            channel_name = channel_id
                    
                    # Process messages
                    for msg in msgs:
                        text = msg.get("text", "")
                        if not text:
                            continue
                        
                        # Calculate relevance for this specific query
                        relevance_score = calculate_message_relevance(
                            text, keywords, priority_terms, search_strategy
                        )
                        
                        if relevance_score > 0:
                            username = _resolve_username(client, msg.get("user"), user_cache)
                            try:
                                permalink_resp = client.chat_getPermalink(channel=channel_id, message_ts=msg.get("ts"))
                                permalink = permalink_resp.get("permalink", "")
                            except Exception:
                                permalink = ""
                            
                            latest_results.append({
                                "text": text,
                                "username": username,
                                "channel": channel_name,
                                "ts": msg.get("ts"),
                                "permalink": permalink,
                                "relevance_score": relevance_score,
                            })
                    
                    # If we found results in the target channel, prioritize them
                    if target_channel and any(r["channel"].lower() == target_channel for r in latest_results):
                        break
                        
                except SlackApiError as e:
                    logger.warning(f"Failed to fetch from {channel_id}: {e}")
                    continue
            
            # Sort by relevance and timestamp
            latest_results.sort(key=lambda x: (-x.get("relevance_score", 0), -float(x.get("ts", "0"))))
            
            # Clean up and return
            for result in latest_results:
                result.pop("relevance_score", None)
            
            return latest_results[:max(1, int(limit or 1))]

        # Build optimized search query
        search_terms = []
        
        # Prioritize priority_terms for exact matching
        if priority_terms:
            if search_strategy == "exact_match":
                # Use quotes for exact phrase matching
                search_terms.extend([f'"{term}"' for term in priority_terms])
            else:
                search_terms.extend(priority_terms)
        
        # Add keywords
        if keywords:
            # Remove duplicates already in priority_terms
            unique_keywords = [kw for kw in keywords if kw not in priority_terms]
            search_terms.extend(unique_keywords[:5])  # Limit keywords to avoid query overload
        
        # If no terms, use original query
        if not search_terms:
            search_query = user_query
        else:
            search_query = " ".join(search_terms)
        
        # Add time constraints (only if not "all")
        if time_range == "recent":
            search_query += " after:1d"
        elif time_range == "7d":
            search_query += " after:7d"
        elif time_range == "30d":
            search_query += " after:30d"
        elif time_range == "90d":
            search_query += " after:90d"
        # For "all" time_range, don't add any time constraints
        
        logger.info(f"Optimized Slack query: {search_query}")
        logger.info(f"Search strategy: {search_strategy}")
        logger.info(f"Searching {len(channel_ids)} channels")
        
        # Log channel names for debugging
        channel_names = []
        for cid in channel_ids[:10]:  # Log first 10 channels
            try:
                info = client.conversations_info(channel=cid)
                ch_data = info.get("channel", {})
                channel_names.append(ch_data.get("name", cid))
            except Exception:
                channel_names.append(cid)
        logger.info(f"Sample channels being searched: {channel_names}")
        
        # Try search API first
        all_results = []
        
        try:
            resp = client.search_messages(query=search_query, count=limit * 3)
            messages = resp.get("messages", {})
            matches = messages.get("matches", [])
            
            logger.info(f"Search API found {len(matches)} total matches for query: {search_query}")
            
            for item in matches:
                channel_info = item.get("channel", {})
                channel_id = channel_info.get("id", "")
                channel_name = channel_info.get("name", "unknown")
                
                # Log all matches for debugging
                logger.debug(f"Found match in channel {channel_name} ({channel_id}): {item.get('text', '')[:100]}...")
                
                # Filter to accessible channels
                if channel_id in channel_ids:
                    text = item.get("text", "")
                    
                    # Calculate relevance with our enhanced scoring
                    relevance_score = calculate_message_relevance(
                        text, keywords, priority_terms, search_strategy
                    )
                    
                    logger.debug(f"Relevance score for '{text[:50]}...': {relevance_score}")
                    
                    # Skip low relevance results for exact_match strategy
                    if search_strategy == "exact_match" and relevance_score < 20:
                        logger.debug(f"Skipping low relevance result: {relevance_score}")
                        continue
                    
                    ts = item.get("ts") or (item.get("message", {}) or {}).get("ts")
                    permalink = item.get("permalink")
                    
                    username = item.get("username")
                    if not username:
                        username = _resolve_username(client, item.get("user"), user_cache)
                    
                    all_results.append({
                        "text": text,
                        "username": username,
                        "channel": channel_name,
                        "ts": ts,
                        "permalink": permalink,
                        "relevance_score": relevance_score
                    })
                else:
                    logger.debug(f"Channel {channel_name} ({channel_id}) not in accessible channels list")
            
            logger.info(f"Search API returned {len(all_results)} relevant results after filtering")
            
        except SlackApiError as e:
            logger.warning(f"Search API failed, using recent messages: {e}")
            # Fallback to recent messages
            # Determine max_age_hours based on time_range
            if time_range == "all":
                max_age_hours = 0  # Search all history
            elif time_range == "recent":
                max_age_hours = 24  # 1 day
            elif time_range == "7d":
                max_age_hours = 168  # 7 days
            elif time_range == "30d":
                max_age_hours = 720  # 30 days
            elif time_range == "90d":
                max_age_hours = 2160  # 90 days
            else:
                max_age_hours = 168  # Default to 7 days
            
            # Use all accessible channels for fallback search
            fallback_channels = "all" if channels == "all" else (channels if isinstance(channels, str) else "")
            all_results = search_slack_recent(
                fallback_channels,
                user_query,
                limit * 3,  # Increased multiplier
                max_age_hours,
                keywords,
                priority_terms,
                search_strategy
            )
            logger.info(f"Fallback search returned {len(all_results)} results")
        
        # Sort by relevance and recency
        if all_results:
            all_results.sort(
                key=lambda x: (
                    -x.get("relevance_score", 0),
                    -float(x.get("ts", "0"))
                )
            )
        
        # Clean up and return
        for result in all_results:
            result.pop("relevance_score", None)
        
        final_results = all_results[:limit]
        logger.info(f"Returning {len(final_results)} Slack results")
        return final_results
        
    except Exception as e:
        logger.error(f"Optimized Slack search failed: {e}")
        return search_slack(user_query, limit)


__all__ = ["search_slack", "search_slack_recent", "search_slack_optimized"]