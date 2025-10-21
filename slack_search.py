from __future__ import annotations

import logging
import time
import re
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from functools import lru_cache
import threading

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import streamlit as st


logger = logging.getLogger(__name__)

# Thread-safe caches
_user_cache_lock = threading.Lock()
_channel_cache_lock = threading.Lock()
_user_cache: Dict[str, str] = {}
_channel_info_cache: Dict[str, Dict] = {}


def _get_slack_client() -> WebClient:
    token = st.secrets.get("SLACK_USER_TOKEN")
    if not token:
        raise RuntimeError(
            "Missing SLACK_USER_TOKEN environment variable (must be a user xoxp- token)."
        )
    return WebClient(token=token)


def _resolve_username(client: WebClient, user_id: Optional[str]) -> str:
    """Thread-safe username resolution with caching."""
    if not user_id:
        return "Unknown"
    
    with _user_cache_lock:
        if user_id in _user_cache:
            return _user_cache[user_id]
    
    try:
        resp = client.users_info(user=user_id)
        profile = resp.get("user", {})
        username = profile.get("real_name") or profile.get("name") or user_id
        
        with _user_cache_lock:
            _user_cache[user_id] = username
        
        return username
    except SlackApiError as e:
        logger.warning("Failed to resolve username for %s: %s", user_id, e)
        return user_id


def _get_channel_info(client: WebClient, channel_id: str) -> Dict:
    """Thread-safe channel info retrieval with caching."""
    with _channel_cache_lock:
        if channel_id in _channel_info_cache:
            return _channel_info_cache[channel_id]
    
    try:
        info = client.conversations_info(channel=channel_id)
        channel_data = info.get("channel", {})
        
        with _channel_cache_lock:
            _channel_info_cache[channel_id] = channel_data
        
        return channel_data
    except Exception as e:
        logger.warning(f"Failed to get info for channel {channel_id}: {e}")
        return {"id": channel_id, "name": channel_id}


def _get_channel_id(client: WebClient, channel_name_or_id: str) -> Optional[str]:
    """Resolve channel name to ID with caching."""
    if not channel_name_or_id:
        return None

    # Already an ID
    if channel_name_or_id.startswith(("C", "G")) and " " not in channel_name_or_id:
        return channel_name_or_id

    channel_name = channel_name_or_id.lstrip('#')
    
    # Skip placeholder values
    if channel_name.lower() in {"specific_channel_name", "channel_name", "none", ""}:
        return None

    logger.info(f"Looking for channel: '{channel_name}'")
    
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
                ch_name = channel.get("name", "")
                ch_id = channel.get("id", "")
                
                if ch_name == channel_name:
                    logger.info(f"Found channel '{channel_name}' with ID: {ch_id}")
                    return ch_id

            next_cursor = (response.get("response_metadata") or {}).get("next_cursor") or ""
            if not next_cursor:
                break

        logger.warning(f"Channel '{channel_name}' not found")
        return None

    except SlackApiError as e:
        logger.error(f"Failed to resolve channel name '{channel_name}': {e}")
        return None


def _get_all_channels_impl(client: WebClient, limit: int = 200) -> List[str]:
    """Internal implementation of channel listing."""
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


def _get_all_channels(limit: int = 200) -> List[str]:
    """Get all accessible channels (cached per session)."""
    # Use Streamlit session state for caching instead of lru_cache
    if "slack_channel_list" not in st.session_state:
        client = _get_slack_client()
        st.session_state["slack_channel_list"] = _get_all_channels_impl(client, limit)
    return st.session_state["slack_channel_list"]


def _quick_relevance_check(text: str, keywords: List[str], priority_terms: List[str]) -> bool:
    """Fast preliminary check before detailed scoring."""
    if not text:
        return False
    
    text_lower = text.lower()
    
    # Must match at least one priority term if they exist
    if priority_terms:
        if not any(term.lower() in text_lower for term in priority_terms):
            return False
    
    # Must match at least one keyword if they exist
    if keywords:
        if not any(kw.lower() in text_lower for kw in keywords):
            return False
    
    return True


def calculate_message_relevance(
    text: str,
    keywords: List[str],
    priority_terms: List[str],
    search_strategy: str = "fuzzy_match"
) -> float:
    """Calculate relevance score for a message (0-100)."""
    if not text:
        return 0.0
    
    text_lower = text.lower()
    score = 0.0
    
    # Priority terms scoring
    if priority_terms:
        priority_matches = sum(1 for term in priority_terms if term.lower() in text_lower)
        if search_strategy == "exact_match" and priority_matches == 0:
            return 0.0
        score += priority_matches * 20
    
    # Keyword matching
    if keywords:
        keyword_matches = sum(1 for kw in keywords if kw.lower() in text_lower)
        score += keyword_matches * 5
        
        if keyword_matches > 0:
            keyword_density = keyword_matches / len(keywords)
            score += keyword_density * 10
    
    # Exact phrase matching bonus
    for kw in keywords + priority_terms:
        if len(kw) > 3 and re.search(r'\b' + re.escape(kw.lower()) + r'\b', text_lower):
            score += 3
    
    # Technical patterns
    technical_patterns = [
        r'\bv?\d+\.\d+(?:\.\d+)?(?:\.\d+)?\b',
        r'\b[A-Z]+-\d+\b',
        r'\b[A-Z]{2,}\d+\b',
    ]
    for pattern in technical_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            score += 5
    
    # Semantic context bonuses
    semantic_bonuses = {
        'architecture': ['design', 'structure', 'components', 'system', 'framework'],
        'server': ['service', 'backend', 'api', 'endpoint', 'host'],
        'integration': ['connect', 'link', 'bridge', 'interface'],
        'authentication': ['auth', 'login', 'security', 'access'],
        'data': ['information', 'analytics', 'insights', 'metrics'],
        'development': ['dev', 'build', 'create', 'develop'],
        'deployment': ['deploy', 'launch', 'release', 'publish'],
        'testing': ['test', 'qa', 'validate', 'verify'],
    }
    
    for primary_term, synonyms in semantic_bonuses.items():
        if any(primary_term in kw.lower() for kw in keywords + priority_terms):
            for synonym in synonyms:
                if synonym in text_lower:
                    score += 2
    
    # Length penalty for very short messages
    if len(text.strip()) < 20:
        score *= 0.5
    
    return min(score, 100.0)


def _search_channel_messages(
    client: WebClient,
    channel_id: str,
    oldest: float,
    keywords: List[str],
    priority_terms: List[str],
    search_strategy: str,
    max_messages: int = 100,
    target_results: int = 5
) -> List[Dict]:
    """Search a single channel for relevant messages."""
    results = []
    
    try:
        next_cursor: Optional[str] = None
        total_fetched = 0
        
        while total_fetched < max_messages:
            response = client.conversations_history(
                channel=channel_id,
                limit=min(100, max_messages - total_fetched),
                cursor=next_cursor or None,
                inclusive=True,
                oldest=str(oldest) if oldest > 0 else None
            )
            
            messages = response.get("messages", [])
            if not messages:
                break
            
            total_fetched += len(messages)
            
            for msg in messages:
                text = msg.get("text", "")
                if not text:
                    continue
                
                # Skip system messages
                if msg.get("subtype") in ["channel_join", "channel_leave", "channel_topic", "channel_purpose"]:
                    continue
                
                # Quick relevance check
                if not _quick_relevance_check(text, keywords, priority_terms):
                    continue
                
                # Detailed scoring
                relevance_score = calculate_message_relevance(
                    text, keywords, priority_terms, search_strategy
                )
                
                if relevance_score > 0:
                    try:
                        ts_val = float(msg.get("ts", "0"))
                    except Exception:
                        ts_val = 0.0
                    
                    results.append({
                        "text": text,
                        "user_id": msg.get("user"),
                        "channel_id": channel_id,
                        "ts": msg.get("ts"),
                        "relevance": relevance_score,
                        "_timestamp": ts_val,
                    })
                    
                    # Early exit if we have enough highly relevant results
                    if len(results) >= target_results * 2:
                        high_quality_recent = [r for r in results[-5:] if r["relevance"] > 30]
                        if len(high_quality_recent) >= 3:
                            logger.info(f"Early exit from {channel_id} with {len(results)} results")
                            return results
            
            next_cursor = (response.get("response_metadata") or {}).get("next_cursor") or ""
            if not next_cursor:
                break
        
        logger.info(f"Channel {channel_id}: Found {len(results)} relevant messages from {total_fetched} fetched")
        
    except SlackApiError as e:
        logger.warning(f"Failed to fetch from channel {channel_id}: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error in channel {channel_id}: {e}")
    
    return results


def search_slack_recent(
    channel_name: str, 
    query: str, 
    max_results: int = 10,
    max_age_hours: int = 72,
    keywords: Optional[List[str]] = None,
    priority_terms: Optional[List[str]] = None,
    search_strategy: str = "fuzzy_match"
) -> List[dict]:
    """Enhanced recent message search with parallel processing."""
    
    client = _get_slack_client()
    keywords = keywords or []
    priority_terms = priority_terms or []

    try:
        # Determine channels to search
        channel_ids: List[str] = []
        
        if channel_name and channel_name.strip():
            channel_id = _get_channel_id(client, channel_name.strip())
            if channel_id:
                channel_ids = [channel_id]
        else:
            logger.info("Searching all accessible channels")
            channel_ids = _get_all_channels(limit=200)

        if not channel_ids:
            logger.warning("No channels to search")
            return []

        # Calculate time window
        now_ts = time.time()
        oldest = now_ts - (max_age_hours * 3600) if max_age_hours > 0 else 0

        # Parallel search across channels
        all_messages: List[Dict] = []
        
        # Use thread pool for parallel channel searching
        max_workers = min(20, len(channel_ids))  # Increased max_workers for faster processing
        
        # Use context manager properly
        executor = ThreadPoolExecutor(max_workers=max_workers)
        try:
            future_to_channel = {
                executor.submit(
                    _search_channel_messages,
                    client,
                    channel_id,
                    oldest,
                    keywords,
                    priority_terms,
                    search_strategy,
                    100,
                    max_results
                ): channel_id
                for channel_id in channel_ids
            }
            
            # Wait for all futures with timeout=25s
            done, not_done = wait(future_to_channel.keys(), timeout=25)
            
            # Cancel any remaining futures
            for future in not_done:
                future.cancel()
            
            # Collect results from completed futures
            for future in done:
                channel_id = future_to_channel[future]
                try:
                    channel_results = future.result(timeout=1)
                    all_messages.extend(channel_results)
                    
                    # Early exit if we have enough good results
                    if len(all_messages) >= max_results * 3:
                        high_quality = [m for m in all_messages if m["relevance"] > 40]
                        if len(high_quality) >= max_results:
                            logger.info(f"Early exit: Found {len(high_quality)} high-quality results")
                            # Cancel remaining futures
                            for f in future_to_channel.keys():
                                if not f.done():
                                    f.cancel()
                            break
                except Exception as e:
                    logger.warning(f"Channel {channel_id} search failed: {e}")
        
        finally:
            # Properly shutdown executor
            executor.shutdown(wait=False)

        logger.info(f"Total messages found: {len(all_messages)}")

        if not all_messages:
            return []

        # Sort by relevance then timestamp
        all_messages.sort(key=lambda x: (-x.get("relevance", 0), -x.get("_timestamp", 0)))

        # Enrich and format results
        results: List[dict] = []
        
        for item in all_messages[:max_results]:
            channel_id = item["channel_id"]
            channel_info = _get_channel_info(client, channel_id)
            channel_name = channel_info.get("name", channel_id)
            username = _resolve_username(client, item.get("user_id"))
            
            try:
                permalink_resp = client.chat_getPermalink(channel=channel_id, message_ts=item.get("ts"))
                permalink = permalink_resp.get("permalink", "")
            except Exception:
                permalink = ""

            results.append({
                "text": item["text"],
                "username": username,
                "channel": channel_name,
                "ts": item["ts"],
                "permalink": permalink,
            })

        logger.info(f"Returning {len(results)} relevant messages")
        return results

    except Exception as e:
        logger.exception(f"Unexpected error in search_slack_recent: {e}")
        return []


def search_slack(query: str, max_results: int = 20) -> List[dict]:
    """Basic Slack search using search API."""
    
    if not query or not query.strip():
        return []

    client = _get_slack_client()
    collected: List[dict] = []
    page = 1
    per_page = 20

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
                username = item.get("username") or _resolve_username(client, item.get("user"))

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
    """Optimized Slack search with adaptive strategies and improved fallback."""
    
    client = _get_slack_client()
    
    slack_params = intent_data.get("slack_params", {})
    channels = slack_params.get("channels", "all")
    time_range = slack_params.get("time_range", "30d")
    keywords = slack_params.get("keywords", [])
    priority_terms = slack_params.get("priority_terms", [])
    limit = slack_params.get("limit", 10)
    search_strategy = intent_data.get("search_strategy", "fuzzy_match")
    min_results_threshold = 3  # Minimum results to avoid fallback
    
    try:
        # Determine channels
        channel_ids: List[str] = []
        
        if channels == "all":
            channel_ids = _get_all_channels(limit=200)
        elif isinstance(channels, str):
            channel_id = _get_channel_id(client, channels)
            if channel_id:
                channel_ids = [channel_id]
        elif isinstance(channels, list):
            for channel_name in channels:
                channel_id = _get_channel_id(client, channel_name)
                if channel_id:
                    channel_ids.append(channel_id)
        
        if not channel_ids:
            logger.warning("No accessible channels found")
            return []
        
        # Handle latest_message intent
        intent_type = (intent_data.get("intent") or "").lower()
        user_query_lower = user_query.lower()
        
        is_latest_query = (
            intent_type == "latest_message" or
            "latest" in user_query_lower or
            "recent" in user_query_lower
        )
        
        if is_latest_query:
            # For latest messages, use adaptive time window starting recent
            max_age_hours = 24 if time_range == "recent" else 168
            results = search_slack_recent(
                "",
                user_query,
                limit,
                max_age_hours,
                keywords,
                priority_terms,
                search_strategy
            )
            
            # If insufficient results, extend time window
            if len(results) < min_results_threshold:
                logger.info(f"Extending time window for latest query: {len(results)} < {min_results_threshold}")
                extended_results = search_slack_recent(
                    "",
                    user_query,
                    limit,
                    max_age_hours * 2,  # Double the time window
                    keywords,
                    priority_terms,
                    search_strategy
                )
                results = extended_results[:limit]  # Take up to limit
            
            return results
        
        # Build optimized search query
        search_terms = []
        
        if priority_terms:
            if search_strategy == "exact_match":
                search_terms.extend([f'"{term}"' for term in priority_terms])
            else:
                search_terms.extend(priority_terms)
        
        if keywords:
            unique_keywords = [kw for kw in keywords if kw not in priority_terms]
            search_terms.extend(unique_keywords[:5])
        
        search_query = " ".join(search_terms) if search_terms else user_query
        
        # Add time constraints if not 'all'
        time_constraint = ""
        if time_range != "all":
            if time_range == "recent":
                time_constraint = " after:1d"
            elif time_range == "7d":
                time_constraint = " after:7d"
            elif time_range == "30d":
                time_constraint = " after:30d"
            elif time_range == "90d":
                time_constraint = " after:90d"
        
        search_query_with_time = search_query + time_constraint
        
        logger.info(f"Optimized Slack query with time: {search_query_with_time}")
        logger.info(f"Searching {len(channel_ids)} channels with strategy: {search_strategy}")
        
        # Try search API first with time constraint if applicable
        all_results = []
        
        try:
            resp = client.search_messages(query=search_query_with_time, count=limit * 3)
            messages = resp.get("messages", {})
            matches = messages.get("matches", [])
            
            logger.info(f"Search API (with time) found {len(matches)} matches")
            
            for item in matches:
                channel_info = item.get("channel", {})
                channel_id = channel_info.get("id", "")
                
                if channel_id in channel_ids:
                    text = item.get("text", "")
                    
                    relevance_score = calculate_message_relevance(
                        text, keywords, priority_terms, search_strategy
                    )
                    
                    if search_strategy == "exact_match" and relevance_score < 20:
                        continue
                    
                    channel_name = channel_info.get("name", "unknown")
                    ts = item.get("ts") or (item.get("message", {}) or {}).get("ts")
                    permalink = item.get("permalink")
                    username = item.get("username") or _resolve_username(client, item.get("user"))
                    
                    all_results.append({
                        "text": text,
                        "username": username,
                        "channel": channel_name,
                        "ts": ts,
                        "permalink": permalink,
                        "relevance_score": relevance_score
                    })
            
            logger.info(f"Search API (with time) returned {len(all_results)} relevant results")
            
        except SlackApiError as e:
            logger.warning(f"Search API (with time) failed: {e}")
        
        # If insufficient results and time constraint was applied, try without time constraint
        if len(all_results) < min_results_threshold and time_constraint:
            logger.info("Insufficient results with time constraint, trying without")
            try:
                resp = client.search_messages(query=search_query, count=limit * 3)
                messages = resp.get("messages", {})
                matches = messages.get("matches", [])
                
                logger.info(f"Search API (without time) found {len(matches)} matches")
                
                for item in matches:
                    channel_info = item.get("channel", {})
                    channel_id = channel_info.get("id", "")
                    
                    if channel_id in channel_ids:
                        text = item.get("text", "")
                        
                        relevance_score = calculate_message_relevance(
                            text, keywords, priority_terms, search_strategy
                        )
                        
                        if search_strategy == "exact_match" and relevance_score < 20:
                            continue
                        
                        channel_name = channel_info.get("name", "unknown")
                        ts = item.get("ts") or (item.get("message", {}) or {}).get("ts")
                        permalink = item.get("permalink")
                        username = item.get("username") or _resolve_username(client, item.get("user"))
                        
                        all_results.append({
                            "text": text,
                            "username": username,
                            "channel": channel_name,
                            "ts": ts,
                            "permalink": permalink,
                            "relevance_score": relevance_score
                        })
                
                logger.info(f"Search API (without time) returned {len(all_results)} relevant results")
                
            except SlackApiError as e:
                logger.warning(f"Search API (without time) failed: {e}")
        
        # Final fallback: If still no results, do manual recent search
        if len(all_results) == 0:
            logger.info("No results from search API, falling back to manual recent search")
            max_age_hours_map = {
                "all": 0,
                "recent": 24,
                "7d": 168,
                "30d": 720,
                "90d": 2160
            }
            max_age_hours = max_age_hours_map.get(time_range, 168)
            
            manual_results = search_slack_recent(
                "",
                user_query,
                limit * 3,
                max_age_hours,
                keywords,
                priority_terms,
                search_strategy
            )
            
            # Convert manual results to same format
            for m in manual_results:
                m["relevance_score"] = calculate_message_relevance(
                    m["text"], keywords, priority_terms, search_strategy
                )
            all_results.extend(manual_results)
        
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