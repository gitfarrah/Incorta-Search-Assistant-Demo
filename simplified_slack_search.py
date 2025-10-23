from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import streamlit as st

from channel_intelligence import get_channel_intelligence

logger = logging.getLogger(__name__)


def _get_slack_client() -> WebClient:
    """Get Slack client from session state or create new one."""
    if "slack_client" not in st.session_state:
        token = st.secrets.get("SLACK_USER_TOKEN")
        if not token:
            raise RuntimeError("Missing SLACK_USER_TOKEN environment variable.")
        st.session_state["slack_client"] = WebClient(token=token)
    return st.session_state["slack_client"]


def _resolve_username(client: WebClient, user_id: Optional[str]) -> str:
    """Resolve user ID to username with caching."""
    if not user_id:
        return "Unknown User"
    
    # Use session state for caching
    cache_key = f"user_cache_{user_id}"
    if cache_key in st.session_state:
        return st.session_state[cache_key]
    
    try:
        resp = client.users_info(user=user_id)
        if resp.get("ok"):
            profile = resp.get("user", {})
            username = (
                profile.get("real_name") or 
                profile.get("display_name") or 
                profile.get("name") or 
                profile.get("profile", {}).get("display_name") or
                profile.get("profile", {}).get("real_name") or
                f"User {user_id}"
            )
        else:
            username = f"User {user_id}"
        
        st.session_state[cache_key] = username
        return username
        
    except Exception as e:
        logger.warning(f"Failed to resolve username for {user_id}: {e}")
        fallback = f"User {user_id}"
        st.session_state[cache_key] = fallback
        return fallback


def _format_slack_timestamp(ts: str) -> str:
    """Convert Slack timestamp to readable date format."""
    try:
        if not ts:
            return "Unknown date"
        timestamp = float(ts)
        dt = datetime.fromtimestamp(timestamp)
        return dt.strftime("%Y-%m-%d %H:%M:%S")
    except (ValueError, TypeError):
        return "Unknown date"


def strategy_1_native_slack_search(user_query: str) -> List[Dict[str, Any]]:
    """
    Strategy 1: Native Slack Search (Unfiltered)
    
    Goal: Mimic typing in the Slack search bar to get natural results.
    Action: Call search.messages with the user's query directly.
    Constraints: NO artificial limits, channel filters, or time restrictions.
    Return ALL results exactly as Slack ranks them.
    """
    logger.info(f"Strategy 1: Native Slack search for: {user_query}")
    
    client = _get_slack_client()
    results = []
    
    try:
        # Pass query directly to Slack - no modifications, no filters, no time limits
        response = client.search_messages(
            query=user_query,
            count=1000,  # Maximum allowed by Slack API
            sort="score"  # Let Slack rank by relevance
        )
        
        messages = response.get("messages", {})
        matches = messages.get("matches", [])
        
        logger.info(f"Native Slack search returned {len(matches)} matches")
        
        # Process results - filter out private DMs but keep all other results
        for match in matches:
            # Extract message data
            text = match.get("text", "")
            if not text:
                continue
            
            # Get channel info
            channel_info = match.get("channel", {})
            channel_name = channel_info.get("name", "unknown")
            channel_id = channel_info.get("id", "")
            
            # BLOCK PRIVATE DMs - Skip direct messages and group DMs
            if channel_info.get("is_im") or channel_info.get("is_mpim"):
                logger.debug(f"Skipping private DM: {channel_name}")
                continue
            
            # BLOCK SPECIFIC CHANNELS - Skip inhousezendesk channel
            if channel_name.lower() == "inhousezendesk":
                logger.debug(f"Skipping excluded channel: {channel_name}")
                continue
            
            # Get user info
            user_id = match.get("user")
            username = _resolve_username(client, user_id)
            
            # Get timestamp and permalink
            ts = match.get("ts")
            permalink = match.get("permalink", "")
            
            results.append({
                "text": text,
                "username": username,
                "channel": channel_name,
                "channel_id": channel_id,
                "ts": ts,
                "date": _format_slack_timestamp(ts),
                "permalink": permalink,
                "strategy": "native_slack",
                "slack_score": match.get("score", 0.0)  # Store Slack's relevance score
            })
        
        logger.info(f"Strategy 1: Returning {len(results)} results (no filtering applied)")
        return results
        
    except SlackApiError as e:
        logger.error(f"Strategy 1: Slack API error: {e}")
        return []
    except Exception as e:
        logger.error(f"Strategy 1: Unexpected error: {e}")
        return []


def strategy_2_smart_targeted_search(
    user_query: str,
    intent_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Strategy 2: Smart Targeted Search (Channel/Time Intelligence)
    
    Goal: Use intelligence to efficiently target relevant content across all history.
    Action: Use sophisticated Channel/Time Intelligence engine to:
    - Channel Selection: Identify relevant channels to INCLUDE and irrelevant ones to EXCLUDE
    - Time Intent Analysis: Analyze query for temporal specificity (used in ranking, NOT filtering)
    - Search: Search ALL history in selected channels using conversations.history
    - Scoring: Apply semantic relevance score to all retrieved messages
    """
    logger.info(f"Strategy 2: Smart targeted search for: {user_query}")
    
    client = _get_slack_client()
    intelligence = get_channel_intelligence()
    
    # Extract parameters
    slack_params = intent_data.get("slack_params", {})
    keywords = slack_params.get("keywords", [])
    priority_terms = slack_params.get("priority_terms", [])
    
    # Get relevant channels using channel intelligence
    relevant_channel_ids = intelligence.get_relevant_channels(
        query=user_query,
        top_k=10,  # More channels for better coverage
        keywords=keywords,
        priority_terms=priority_terms
    )
    
    if not relevant_channel_ids:
        logger.info("Strategy 2: No relevant channels found")
        return []
    
    logger.info(f"Strategy 2: Searching {len(relevant_channel_ids)} relevant channels")
    
    # Search ALL history in relevant channels (no time restrictions)
    all_results = []
    
    # Use thread pool for parallel channel searching
    max_workers = min(5, len(relevant_channel_ids))
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_channel = {
            executor.submit(
                _search_channel_all_history,
                client,
                channel_id,
                user_query,
                keywords,
                priority_terms
            ): channel_id
            for channel_id in relevant_channel_ids
        }
        
        # Collect results with timeout
        try:
            for future in as_completed(future_to_channel, timeout=30):
                channel_id = future_to_channel[future]
                try:
                    channel_results = future.result(timeout=10)
                    all_results.extend(channel_results)
                except Exception as e:
                    logger.warning(f"Strategy 2: Failed to search channel {channel_id}: {e}")
        except TimeoutError:
            logger.warning("Strategy 2: Timeout waiting for channel search results")
            # Cancel remaining futures
            for future in future_to_channel:
                future.cancel()
    
    logger.info(f"Strategy 2: Found {len(all_results)} results from targeted channels")
    return all_results


def _search_channel_all_history(
    client: WebClient,
    channel_id: str,
    user_query: str,
    keywords: List[str],
    priority_terms: List[str]
) -> List[Dict[str, Any]]:
    """Search a single channel for ALL history (no time restrictions)."""
    results = []
    
    try:
        # Get channel info
        channel_info = client.conversations_info(channel=channel_id)
        channel_data = channel_info.get("channel", {})
        channel_name = channel_data.get("name", f"Channel {channel_id}")
        
        # BLOCK PRIVATE DMs - Skip direct messages and group DMs
        if channel_data.get("is_im") or channel_data.get("is_mpim"):
            logger.debug(f"Skipping private DM channel: {channel_name}")
            return []
        
        # BLOCK SPECIFIC CHANNELS - Skip inhousezendesk channel
        if channel_name.lower() == "inhousezendesk":
            logger.debug(f"Skipping excluded channel: {channel_name}")
            return []
        
        # Search ALL history - no time restrictions
        next_cursor: Optional[str] = None
        total_fetched = 0
        max_messages = 100  # Increased limit for better coverage
        
        while total_fetched < max_messages:
            response = client.conversations_history(
                channel=channel_id,
                limit=min(100, max_messages - total_fetched),
                cursor=next_cursor or None,
                inclusive=True
                # NO oldest parameter - search ALL history
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
                
                # Calculate semantic relevance score
                relevance_score = _calculate_semantic_relevance(text, user_query, keywords, priority_terms)
                
                # Include ALL messages with any relevance (let ranking handle it)
                if relevance_score > 0:
                    # Get user info
                    user_id = msg.get("user")
                    username = _resolve_username(client, user_id)
                    
                    # Get timestamp and permalink
                    ts = msg.get("ts")
                    try:
                        permalink_resp = client.chat_getPermalink(channel=channel_id, message_ts=ts)
                        permalink = permalink_resp.get("permalink", "")
                    except Exception:
                        permalink = ""
                    
                    results.append({
                        "text": text,
                        "username": username,
                        "channel": channel_name,
                        "channel_id": channel_id,
                        "ts": ts,
                        "date": _format_slack_timestamp(ts),
                        "permalink": permalink,
                        "semantic_score": relevance_score,
                        "strategy": "smart_targeted"
                    })
            
            next_cursor = (response.get("response_metadata") or {}).get("next_cursor") or ""
            if not next_cursor:
                break
        
        logger.info(f"Channel {channel_name}: Found {len(results)} relevant messages from {total_fetched} total")
        
    except SlackApiError as e:
        logger.warning(f"Failed to search channel {channel_id}: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error searching channel {channel_id}: {e}")
    
    return results


def _calculate_semantic_relevance(
    text: str,
    user_query: str,
    keywords: List[str],
    priority_terms: List[str]
) -> float:
    """Calculate intelligent semantic relevance score for a message."""
    import re
    
    if not text:
        return 0.0
    
    text_lower = text.lower()
    query_lower = user_query.lower()
    score = 0.0
    
    # 1. EXACT MATCHES - Highest priority
    if query_lower in text_lower:
        score += 20.0
    
    # 2. PRIORITY TERMS - High weight for important terms
    for term in priority_terms:
        term_lower = term.lower()
        if term_lower in text_lower:
            # Bonus for exact word boundaries
            if re.search(r'\b' + re.escape(term_lower) + r'\b', text_lower):
                score += 15.0
            else:
                score += 10.0
    
    # 3. KEYWORD MATCHING - Medium weight
    for keyword in keywords:
        keyword_lower = keyword.lower()
        if keyword_lower in text_lower:
            # Bonus for exact word boundaries
            if re.search(r'\b' + re.escape(keyword_lower) + r'\b', text_lower):
                score += 8.0
            else:
                score += 5.0
    
    # 4. INTELLIGENT QUERY TERM MATCHING
    query_terms = [term for term in query_lower.split() if len(term) > 2]
    matched_terms = 0
    for term in query_terms:
        if re.search(r'\b' + re.escape(term) + r'\b', text_lower):
            score += 2.0
            matched_terms += 1
    
    # Bonus for matching multiple query terms
    if matched_terms > 0:
        coverage_bonus = (matched_terms / len(query_terms)) * 10.0
        score += coverage_bonus
    
    # 5. SEMANTIC PATTERN RECOGNITION
    semantic_patterns = _generate_semantic_patterns(query_lower)
    for pattern in semantic_patterns:
        if re.search(pattern, text_lower):
            score += 5.0
    
    # 6. TECHNICAL ENTITY MATCHING
    technical_entities = _extract_technical_entities(query_lower)
    text_entities = _extract_technical_entities(text_lower)
    
    # Exact technical entity matches
    for entity in technical_entities:
        if entity in text_entities:
            score += 12.0
        else:
            # Fuzzy matching for related entities
            similarity = _calculate_entity_similarity(entity, text_entities)
            if similarity > 0.7:
                score += similarity * 8.0
    
    # 7. CONTEXTUAL RELEVANCE
    context_score = _calculate_contextual_relevance(text_lower, query_lower)
    score += context_score
    
    return score


def _generate_semantic_patterns(query: str) -> List[str]:
    """Generate semantic patterns based on query content."""
    patterns = []
    
    # Extract key concepts from query
    if 'release' in query:
        patterns.extend([
            r'\brelease\b.*\b(?:date|schedule|announcement|available|ga)\b',
            r'\b(?:date|schedule|announcement|available|ga)\b.*\brelease\b',
            r'\bgo\s+ga\b',
            r'\bgeneral\s+availability\b'
        ])
    
    if 'delivery' in query:
        patterns.extend([
            r'\bdelivery\b.*\b(?:date|schedule|timeline)\b',
            r'\b(?:date|schedule|timeline)\b.*\bdelivery\b'
        ])
    
    if any(word in query for word in ['version', 'v', 'update']):
        patterns.extend([
            r'\bversion\b.*\d+\.\d+',
            r'\bv\d+\.\d+',
            r'\bupdate\b.*\b(?:available|released)\b'
        ])
    
    if 'date' in query:
        patterns.extend([
            r'\b(?:date|schedule|timeline)\b.*\b(?:release|delivery|available)\b',
            r'\b(?:release|delivery|available)\b.*\b(?:date|schedule|timeline)\b'
        ])
    
    return patterns


def _extract_technical_entities(text: str) -> List[str]:
    """Extract technical entities like version numbers, codes, etc."""
    import re
    entities = []
    
    # Version numbers (various formats)
    version_patterns = [
        r'\b\d{4}\.\d+\.\d+\b',  # 2024.7.5
        r'\b\d+\.\d+\.\d+\b',    # 25.7.2
        r'\bv\d+\.\d+(?:\.\d+)?\b',  # v25.7.2
        r'\bversion\s+\d+\.\d+(?:\.\d+)?\b'  # version 25.7.2
    ]
    
    for pattern in version_patterns:
        entities.extend(re.findall(pattern, text, re.IGNORECASE))
    
    # Ticket numbers, codes
    code_patterns = [
        r'\b[A-Z]+-\d+\b',  # INC-123
        r'\b[A-Z]{2,}\d+\b'  # ABC123
    ]
    
    for pattern in code_patterns:
        entities.extend(re.findall(pattern, text, re.IGNORECASE))
    
    return entities


def _calculate_entity_similarity(entity1: str, entities2: List[str]) -> float:
    """Calculate similarity between technical entities."""
    if not entities2:
        return 0.0
    
    # Extract version components
    def extract_version_parts(version_str):
        import re
        parts = re.findall(r'\d+', version_str)
        return [int(p) for p in parts] if parts else []
    
    parts1 = extract_version_parts(entity1)
    if not parts1:
        return 0.0
    
    max_similarity = 0.0
    for entity2 in entities2:
        parts2 = extract_version_parts(entity2)
        if not parts2:
            continue
        
        # Calculate similarity based on version structure
        if len(parts1) >= 2 and len(parts2) >= 2:
            # Same major.minor version
            if parts1[0] == parts2[0] and parts1[1] == parts2[1]:
                similarity = 0.9
            # Related versions (e.g., 25.7.2 vs 2024.7.5 - same minor.patch)
            elif len(parts1) >= 3 and len(parts2) >= 3 and parts1[1] == parts2[1] and parts1[2] == parts2[2]:
                similarity = 0.8
            # Same minor version
            elif parts1[1] == parts2[1]:
                similarity = 0.6
            # Same patch version
            elif len(parts1) >= 3 and len(parts2) >= 3 and parts1[2] == parts2[2]:
                similarity = 0.4
            else:
                similarity = 0.0
        else:
            similarity = 0.0
        
        max_similarity = max(max_similarity, similarity)
    
    return max_similarity


def _calculate_contextual_relevance(text: str, query: str) -> float:
    """Calculate contextual relevance based on surrounding context."""
    score = 0.0
    
    # Question-answer context
    if '?' in text and any(word in query for word in ['what', 'when', 'where', 'how', 'why']):
        score += 3.0
    
    # Time-related context
    if any(word in query for word in ['date', 'when', 'schedule', 'timeline']):
        if any(word in text for word in ['date', 'schedule', 'timeline', 'when', 'available']):
            score += 4.0
    
    # Status-related context
    if any(word in query for word in ['status', 'available', 'released', 'ga']):
        if any(word in text for word in ['available', 'released', 'ga', 'status', 'ready']):
            score += 4.0
    
    return score


def _analyze_time_intent(user_query: str) -> Dict[str, Any]:
    """
    Analyze query for temporal specificity.
    Returns intent data for use in adaptive recency boost calculation.
    """
    query_lower = user_query.lower()
    
    # High temporal specificity patterns
    high_temporal_patterns = [
        r'\b(?:version|v)\s*\d{4}\.\d+\.\d+\b',  # version 2025.7.2
        r'\b(?:latest|current|newest|most recent)\b',
        r'\b(?:today|yesterday|this week|this month)\b',
        r'\b(?:just|recently|recent)\b'
    ]
    
    # Check for high temporal specificity
    import re
    for pattern in high_temporal_patterns:
        if re.search(pattern, query_lower):
            return {
                "temporal_specificity": "high",
                "boost_factor": 5.0
            }
    
    # Medium temporal specificity
    medium_temporal_patterns = [
        r'\b(?:when|date|schedule|timeline)\b',
        r'\b(?:release|announcement|update)\b'
    ]
    
    for pattern in medium_temporal_patterns:
        if re.search(pattern, query_lower):
            return {
                "temporal_specificity": "medium",
                "boost_factor": 2.0
            }
    
    # Low temporal specificity (conceptual queries)
    return {
        "temporal_specificity": "low",
        "boost_factor": 0.5
    }


def _calculate_adaptive_recency_boost(
    ts: str,
    user_query: str,
    intent_data: Dict[str, Any]
) -> float:
    """
    Calculate adaptive recency boost based on query intent.
    
    Formula: Adaptive Boost = boost_factor * recency_decay
    
    Where:
    - boost_factor depends on temporal specificity of the query
    - recency_decay decreases with age of the message
    
    This is a MULTIPLIER, not a filter - old valuable content can still rank #1
    """
    try:
        timestamp = float(ts)
        days_ago = (time.time() - timestamp) / (24 * 3600)
    except (ValueError, TypeError):
        return 1.0  # No boost if timestamp is invalid
    
    # Analyze time intent
    time_intent = _analyze_time_intent(user_query)
    boost_factor = time_intent["boost_factor"]
    
    # Calculate recency decay
    if time_intent["temporal_specificity"] == "high":
        # High temporal specificity: strong recency preference
        if days_ago < 1:  # Last 24 hours
            recency_decay = 1.0
        elif days_ago < 7:  # Last week
            recency_decay = 0.8
        elif days_ago < 30:  # Last month
            recency_decay = 0.6
        elif days_ago < 90:  # Last 3 months
            recency_decay = 0.4
        else:
            recency_decay = 0.2
    
    elif time_intent["temporal_specificity"] == "medium":
        # Medium temporal specificity: moderate recency preference
        if days_ago < 7:  # Last week
            recency_decay = 1.0
        elif days_ago < 30:  # Last month
            recency_decay = 0.8
        elif days_ago < 90:  # Last 3 months
            recency_decay = 0.6
        else:
            recency_decay = 0.4
    
    else:  # low temporal specificity
        # Low temporal specificity: minimal recency preference (concepts are timeless)
        if days_ago < 30:  # Last month
            recency_decay = 1.0
        elif days_ago < 90:  # Last 3 months
            recency_decay = 0.9
        else:
            recency_decay = 0.8  # Even old content gets high boost
    
    # Calculate final adaptive boost
    adaptive_boost = boost_factor * recency_decay
    
    return adaptive_boost


def _deduplicate_results(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate results based on message_timestamp + channel_id."""
    seen_hashes: Set[int] = set()
    unique_results = []
    
    for result in results:
        # Create a hash based on timestamp and channel_id (as specified)
        ts = result.get("ts", "")
        channel_id = result.get("channel_id", "")
        result_hash = hash(f"{ts}_{channel_id}")
        
        if result_hash not in seen_hashes:
            seen_hashes.add(result_hash)
            unique_results.append(result)
    
    return unique_results


def _calculate_final_score(
    result: Dict[str, Any],
    user_query: str,
    intent_data: Dict[str, Any]
) -> float:
    """
    Calculate final score using the adaptive ranking formula:
    
    Final Score(M) = (α · Semantic Score(M) + β · Slack Score(M)) × Adaptive Recency Boost(Intent)
    
    Where:
    - α = 0.7 (primary factor - semantic relevance)
    - β = 0.3 (secondary factor - Slack's native ranking)
    """
    # Extract scores
    semantic_score = result.get("semantic_score", 0.0)
    slack_score = result.get("slack_score", 0.0)
    
    # If no semantic score (from strategy 1), calculate one
    if semantic_score == 0.0:
        slack_params = intent_data.get("slack_params", {})
        keywords = slack_params.get("keywords", [])
        priority_terms = slack_params.get("priority_terms", [])
        semantic_score = _calculate_semantic_relevance(
            result.get("text", ""),
            user_query,
            keywords,
            priority_terms
        )
    
    # Calculate adaptive recency boost
    adaptive_boost = _calculate_adaptive_recency_boost(
        result.get("ts", "0"),
        user_query,
        intent_data
    )
    
    # Apply the formula: (α · Semantic Score + β · Slack Score) × Adaptive Boost
    alpha = 0.7  # Primary factor
    beta = 0.3   # Secondary factor
    
    base_score = (alpha * semantic_score) + (beta * slack_score)
    final_score = base_score * adaptive_boost
    
    return final_score


def _rank_results(
    results: List[Dict[str, Any]],
    user_query: str,
    intent_data: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """
    Rank results using the adaptive ranking algorithm.
    
    Step A: Combine & Deduplicate (already done)
    Step B: Apply final ranking algorithm
    Step C: Sort by Final Score (descending)
    """
    logger.info(f"Ranking {len(results)} results using adaptive ranking algorithm")
    
    # Calculate final scores for all results
    scored_results = []
    for result in results:
        final_score = _calculate_final_score(result, user_query, intent_data)
        scored_results.append((final_score, result))
    
    # Sort by final score (descending)
    scored_results.sort(key=lambda x: x[0], reverse=True)
    
    # Remove internal scoring fields from final results
    final_results = []
    for _, result in scored_results:
        # Clean up internal fields
        result.pop("semantic_score", None)
        result.pop("slack_score", None)
        final_results.append(result)
    
    logger.info(f"Ranked {len(final_results)} results")
    return final_results


def search_slack_simplified(
    user_query: str,
    intent_data: Dict[str, Any],
    max_total_results: int = 15  # TOP 15 limit for UI
) -> List[Dict[str, Any]]:
    """
    NEW SIMPLIFIED APPROACH: Max-retrieval, adaptive-ranking model
    
    Execute TWO parallel, unfiltered search strategies:
    1. Strategy 1: Native Slack Search (Unfiltered)
    2. Strategy 2: Smart Targeted Search (Channel/Time Intelligence)
    
    Then apply Adaptive Smart Ranking and return TOP 15 messages.
    """
    logger.info(f"Starting max-retrieval, adaptive-ranking search for: {user_query}")
    
    # Run both strategies in parallel
    with ThreadPoolExecutor(max_workers=2) as executor:
        # Submit both strategies
        future_strategy_1 = executor.submit(strategy_1_native_slack_search, user_query)
        future_strategy_2 = executor.submit(strategy_2_smart_targeted_search, user_query, intent_data)
        
        # Collect results with error handling
        strategy_1_results = []
        strategy_2_results = []
        
        try:
            strategy_1_results = future_strategy_1.result(timeout=30)
            logger.info(f"Strategy 1: Retrieved {len(strategy_1_results)} results")
        except Exception as e:
            logger.error(f"Strategy 1 failed: {e}")
        
        try:
            strategy_2_results = future_strategy_2.result(timeout=30)
            logger.info(f"Strategy 2: Retrieved {len(strategy_2_results)} results")
        except Exception as e:
            logger.error(f"Strategy 2 failed: {e}")
            # Strategy 2 is optional - if it fails, we still have Strategy 1 results
    
    # Step A: Combine & Deduplicate
    all_results = strategy_1_results + strategy_2_results
    logger.info(f"Combined: {len(all_results)} total results")
    
    if not all_results:
        logger.warning("No results from either strategy")
        return []
    
    # Deduplicate by message_timestamp + channel_id
    unique_results = _deduplicate_results(all_results)
    logger.info(f"After deduplication: {len(unique_results)} unique results")
    
    # Step B: Apply Adaptive Smart Ranking
    ranked_results = _rank_results(unique_results, user_query, intent_data)
    
    # Step C: Final Output - TOP 15 messages for UI
    if len(ranked_results) > max_total_results:
        logger.info(f"Limiting to TOP {max_total_results} messages from {len(ranked_results)} total")
        ranked_results = ranked_results[:max_total_results]
    
    logger.info(f"Final results: {len(ranked_results)} messages sent to UI")
    return ranked_results


__all__ = ["search_slack_simplified"]