from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass
import threading

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError
import streamlit as st

logger = logging.getLogger(__name__)


@dataclass
class ChannelInfo:
    """Information about a Slack channel."""
    id: str
    name: str
    purpose: str
    topic: str
    is_private: bool
    is_member: bool
    member_count: int
    keywords: Set[str]
    patterns: Set[str]
    category: Optional[str] = None


@dataclass
class ChannelPattern:
    """Detected pattern in channel naming."""
    pattern: str  # e.g., "data-*", "team-*"
    regex: re.Pattern
    channels: Set[str]
    category: str
    confidence: float


class ChannelIntelligence:
    """
    Intelligent channel analysis and filtering for Slack workspaces.
    
    This class analyzes all accessible channels to:
    1. Discover channel patterns and categories
    2. Extract keywords from names, purposes, and topics
    3. Build an index for fast relevant channel lookup
    4. Cache results in session state for performance
    """
    
    def __init__(self, client: WebClient):
        self.client = client
        self._lock = threading.Lock()
        self._channels: Dict[str, ChannelInfo] = {}
        self._patterns: List[ChannelPattern] = []
        self._keyword_index: Dict[str, Set[str]] = defaultdict(set)
        self._category_index: Dict[str, Set[str]] = defaultdict(set)
        self._initialized = False
    
    def _get_slack_client(self) -> WebClient:
        """Get Slack client from session state or create new one."""
        if "slack_client" not in st.session_state:
            token = st.secrets.get("SLACK_USER_TOKEN")
            if not token:
                raise RuntimeError("Missing SLACK_USER_TOKEN environment variable.")
            st.session_state["slack_client"] = WebClient(token=token)
        return st.session_state["slack_client"]
    
    def initialize(self, force_refresh: bool = False) -> None:
        """
        Initialize channel intelligence by analyzing all accessible channels.
        
        Args:
            force_refresh: If True, re-analyze even if cached data exists
        """
        with self._lock:
            if self._initialized and not force_refresh:
                return
            
            # Check session state cache first
            cache_key = "channel_intelligence_cache"
            if not force_refresh and cache_key in st.session_state:
                cached_data = st.session_state[cache_key]
                self._channels = cached_data.get("channels", {})
                self._patterns = cached_data.get("patterns", [])
                self._keyword_index = defaultdict(set, cached_data.get("keyword_index", {}))
                self._category_index = defaultdict(set, cached_data.get("category_index", {}))
                self._initialized = True
                logger.info(f"Loaded cached channel intelligence for {len(self._channels)} channels")
                return
            
            logger.info("Initializing channel intelligence...")
            
            # Discover all channels
            self._discover_channels()
            
            # Analyze patterns
            self._detect_patterns()
            
            # Build keyword index
            self._build_keyword_index()
            
            # Cache in session state
            st.session_state[cache_key] = {
                "channels": self._channels,
                "patterns": self._patterns,
                "keyword_index": dict(self._keyword_index),
                "category_index": dict(self._category_index)
            }
            
            self._initialized = True
            logger.info(f"Channel intelligence initialized: {len(self._channels)} channels, {len(self._patterns)} patterns")
    
    def _discover_channels(self) -> None:
        """Discover all accessible channels and extract basic information."""
        logger.info("Discovering accessible channels...")
        
        try:
            next_cursor: Optional[str] = None
            total_channels = 0
            
            while True:
                response = self.client.conversations_list(
                    types="public_channel,private_channel",
                    exclude_archived=True,
                    limit=1000,
                    cursor=next_cursor or None,
                )
                
                channels = response.get("channels", [])
                if not channels:
                    break
                
                for channel in channels:
                    channel_id = channel.get("id")
                    if not channel_id:
                        continue
                    
                    # Skip DMs and group DMs
                    if channel.get("is_im") or channel.get("is_mpim"):
                        continue
                    
                    # Skip excluded channels
                    channel_name = channel.get("name", "").lower()
                    if channel_name == "inhousezendesk":
                        continue
                    
                    # Only include channels user is member of (for private) or all public
                    is_private = channel.get("is_private", False)
                    is_member = channel.get("is_member", False)
                    
                    if is_private and not is_member:
                        continue
                    
                    # Extract channel information
                    name = channel.get("name", "")
                    purpose = channel.get("purpose", {}).get("value", "")
                    topic = channel.get("topic", {}).get("value", "")
                    member_count = channel.get("num_members", 0)
                    
                    # Extract keywords from name, purpose, and topic
                    keywords = self._extract_keywords_from_text(f"{name} {purpose} {topic}")
                    
                    channel_info = ChannelInfo(
                        id=channel_id,
                        name=name,
                        purpose=purpose,
                        topic=topic,
                        is_private=is_private,
                        is_member=is_member,
                        member_count=member_count,
                        keywords=keywords,
                        patterns=set()
                    )
                    
                    self._channels[channel_id] = channel_info
                    total_channels += 1
                
                next_cursor = (response.get("response_metadata") or {}).get("next_cursor") or ""
                if not next_cursor:
                    break
            
            logger.info(f"Discovered {total_channels} accessible channels")
            
        except SlackApiError as e:
            logger.error(f"Failed to discover channels: {e}")
            raise
    
    def _extract_keywords_from_text(self, text: str) -> Set[str]:
        """Extract meaningful keywords from text."""
        if not text:
            return set()
        
        # Common stopwords to filter out
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will", "would",
            "could", "should", "may", "might", "can", "about", "what", "when",
            "where", "why", "how", "who", "which", "this", "that", "these", "those",
            "channel", "chat", "discussion", "team", "group", "project"
        }
        
        # Extract words (alphanumeric + hyphens)
        words = re.findall(r'\b[a-zA-Z0-9-]+\b', text.lower())
        
        # Filter and clean
        keywords = set()
        for word in words:
            # Skip stopwords and very short words
            if word in stopwords or len(word) < 2:
                continue
            
            # Skip pure numbers
            if word.isdigit():
                continue
            
            # Clean up the word
            clean_word = word.strip('-')
            if len(clean_word) >= 2:
                keywords.add(clean_word)
        
        return keywords
    
    def _detect_patterns(self) -> None:
        """Detect naming patterns in channels and categorize them."""
        logger.info("Detecting channel patterns...")
        
        # Group channels by common prefixes
        prefix_groups = defaultdict(list)
        suffix_groups = defaultdict(list)
        
        for channel_id, channel_info in self._channels.items():
            name = channel_info.name.lower()
            
            # Extract prefixes (e.g., "data-", "team-", "eng-")
            for i in range(2, len(name)):
                if name[i] == '-':
                    prefix = name[:i+1]
                    prefix_groups[prefix].append(channel_id)
                    break
            
            # Extract suffixes (e.g., "-team", "-dev", "-prod")
            for i in range(len(name)-2, 0, -1):
                if name[i] == '-':
                    suffix = name[i:]
                    suffix_groups[suffix].append(channel_id)
                    break
        
        # Create patterns from significant groups
        self._patterns = []
        
        # Process prefixes
        for prefix, channel_ids in prefix_groups.items():
            if len(channel_ids) >= 2:  # At least 2 channels with this pattern
                category = self._infer_category_from_pattern(prefix, channel_ids)
                confidence = min(1.0, len(channel_ids) / 10.0)  # Higher confidence for more channels
                
                pattern = ChannelPattern(
                    pattern=prefix + "*",
                    regex=re.compile(f"^{re.escape(prefix[:-1])}-", re.IGNORECASE),
                    channels=set(channel_ids),
                    category=category,
                    confidence=confidence
                )
                self._patterns.append(pattern)
                
                # Update channel info
                for channel_id in channel_ids:
                    self._channels[channel_id].patterns.add(pattern.pattern)
                    self._channels[channel_id].category = category
        
        # Process suffixes
        for suffix, channel_ids in suffix_groups.items():
            if len(channel_ids) >= 2:
                category = self._infer_category_from_pattern(suffix, channel_ids)
                confidence = min(1.0, len(channel_ids) / 10.0)
                
                pattern = ChannelPattern(
                    pattern="*" + suffix,
                    regex=re.compile(f"{re.escape(suffix[1:])}$", re.IGNORECASE),
                    channels=set(channel_ids),
                    category=category,
                    confidence=confidence
                )
                self._patterns.append(pattern)
                
                # Update channel info
                for channel_id in channel_ids:
                    self._channels[channel_id].patterns.add(pattern.pattern)
                    if not self._channels[channel_id].category:
                        self._channels[channel_id].category = category
        
        # Sort patterns by confidence
        self._patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        logger.info(f"Detected {len(self._patterns)} channel patterns")
    
    def _infer_category_from_pattern(self, pattern: str, channel_ids: List[str]) -> str:
        """Infer category from pattern and channel names."""
        # Get sample channel names for this pattern
        sample_names = [self._channels[cid].name.lower() for cid in channel_ids[:5]]
        sample_text = " ".join(sample_names)
        
        # Category mapping based on common patterns
        category_keywords = {
            "engineering": ["eng", "dev", "development", "backend", "frontend", "api", "code", "tech"],
            "data": ["data", "analytics", "metrics", "dashboard", "viz", "visualization", "reporting"],
            "product": ["product", "feature", "roadmap", "planning", "strategy"],
            "design": ["design", "ui", "ux", "mockup", "prototype", "wireframe"],
            "marketing": ["marketing", "campaign", "content", "social", "brand"],
            "sales": ["sales", "revenue", "customer", "lead", "prospect"],
            "support": ["support", "help", "customer", "ticket", "issue"],
            "operations": ["ops", "operations", "infrastructure", "deployment", "monitoring"],
            "security": ["security", "auth", "compliance", "audit", "privacy"],
            "qa": ["qa", "test", "testing", "quality", "validation"],
            "project": ["project", "initiative", "milestone", "delivery"],
            "team": ["team", "squad", "group", "department"],
            "release": ["release", "announcement", "announce", "version", "update", "deploy", "deployment", "launch"]
        }
        
        # Find best matching category
        best_category = "general"
        best_score = 0
        
        for category, keywords in category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in sample_text)
            if score > best_score:
                best_score = score
                best_category = category
        
        return best_category
    
    def _build_keyword_index(self) -> None:
        """Build keyword index for fast channel lookup."""
        logger.info("Building keyword index...")
        
        for channel_id, channel_info in self._channels.items():
            # Index all keywords from the channel
            for keyword in channel_info.keywords:
                self._keyword_index[keyword].add(channel_id)
            
            # Index category
            if channel_info.category:
                self._category_index[channel_info.category].add(channel_id)
            
            # Index channel name words
            name_words = self._extract_keywords_from_text(channel_info.name)
            for word in name_words:
                self._keyword_index[word].add(channel_id)
        
        logger.info(f"Built keyword index with {len(self._keyword_index)} keywords")
    
    def get_relevant_channels(
        self, 
        query: str, 
        top_k: int = 20,
        keywords: Optional[List[str]] = None,
        priority_terms: Optional[List[str]] = None
    ) -> List[str]:
        """
        Get the most relevant channels for a query.
        
        Args:
            query: Search query
            top_k: Maximum number of channels to return
            keywords: Additional keywords to consider
            priority_terms: High-priority terms that must match
            
        Returns:
            List of channel IDs ordered by relevance
        """
        if not self._initialized:
            self.initialize()
        
        if not query and not keywords and not priority_terms:
            # Return most active channels if no specific query
            return self._get_most_active_channels(top_k)
        
        # Extract keywords from query if not provided
        if not keywords:
            keywords = list(self._extract_keywords_from_text(query))
        
        if priority_terms:
            keywords.extend(priority_terms)
        
        # Score channels based on keyword matches
        channel_scores = defaultdict(float)
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            
            # Direct keyword matches (highest priority)
            if keyword_lower in self._keyword_index:
                for channel_id in self._keyword_index[keyword_lower]:
                    channel_scores[channel_id] += 2.0  # Increased weight
            
            # Pattern matches
            for pattern in self._patterns:
                if pattern.regex.search(keyword_lower):
                    for channel_id in pattern.channels:
                        channel_scores[channel_id] += pattern.confidence * 1.0  # Increased weight
            
            # Fuzzy matches (partial keyword matching)
            for indexed_keyword, channel_ids in self._keyword_index.items():
                if keyword_lower in indexed_keyword or indexed_keyword in keyword_lower:
                    for channel_id in channel_ids:
                        channel_scores[channel_id] += 0.5  # Increased weight
            
            # Channel name matching (very important for specific channels)
            for channel_id, channel_info in self._channels.items():
                channel_name = channel_info.name.lower()
                if keyword_lower in channel_name:
                    channel_scores[channel_id] += 3.0  # High weight for channel name matches
                elif any(word in channel_name for word in keyword_lower.split()):
                    channel_scores[channel_id] += 1.5  # Medium weight for partial matches
        
        # Boost scores for priority terms
        if priority_terms:
            for term in priority_terms:
                term_lower = term.lower()
                if term_lower in self._keyword_index:
                    for channel_id in self._keyword_index[term_lower]:
                        channel_scores[channel_id] += 2.0  # Higher boost for priority terms
        
        # Sort by score and return top channels
        sorted_channels = sorted(
            channel_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        # Return channel IDs
        relevant_channels = [channel_id for channel_id, score in sorted_channels[:top_k]]
        
        logger.info(f"Found {len(relevant_channels)} relevant channels for query: {query}")
        return relevant_channels
    
    def _get_most_active_channels(self, top_k: int) -> List[str]:
        """Get the most active channels (by member count)."""
        sorted_channels = sorted(
            self._channels.items(),
            key=lambda x: x[1].member_count,
            reverse=True
        )
        return [channel_id for channel_id, _ in sorted_channels[:top_k]]
    
    def get_channel_info(self, channel_id: str) -> Optional[ChannelInfo]:
        """Get information about a specific channel."""
        if not self._initialized:
            self.initialize()
        return self._channels.get(channel_id)
    
    def get_channel_id_by_name(self, channel_name: str) -> Optional[str]:
        """
        Find a channel ID by channel name (case-insensitive).
        
        Args:
            channel_name: The name of the channel to find (with or without #)
        
        Returns:
            The channel ID if found, None otherwise
        """
        if not self._initialized:
            self.initialize()
        
        # Remove # prefix if present
        search_name = channel_name.lstrip('#').lower().strip()
        
        # Search for exact match first
        for channel_id, channel_info in self._channels.items():
            if channel_info.name.lower() == search_name:
                return channel_id
        
        # If no exact match, try partial match
        for channel_id, channel_info in self._channels.items():
            if search_name in channel_info.name.lower():
                return channel_id
        
        return None
    
    def find_similar_channels(self, channel_name: str, max_suggestions: int = 5) -> List[str]:
        """
        Find channels with similar names to the given channel name.
        
        Args:
            channel_name: The channel name to search for (with or without #)
            max_suggestions: Maximum number of suggestions to return
        
        Returns:
            List of similar channel names (without # prefix)
        """
        if not self._initialized:
            self.initialize()
        
        search_name = channel_name.lstrip('#').lower().strip()
        suggestions = []
        
        # Find channels containing any word from the search term
        search_words = search_name.split('_')
        
        for channel_id, channel_info in self._channels.items():
            channel_lower = channel_info.name.lower()
            
            # Calculate similarity score
            score = 0
            
            # Exact match (shouldn't happen if this is called, but just in case)
            if channel_lower == search_name:
                score = 100
            # Partial match
            elif search_name in channel_lower:
                score = 80
            # Check if any search words are in the channel name
            else:
                for word in search_words:
                    if word in channel_lower:
                        score += 20
            
            if score > 0:
                suggestions.append((channel_info.name, score))
        
        # Sort by score (highest first) and return top N
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return [name for name, _ in suggestions[:max_suggestions]]
    
    def get_patterns(self) -> List[ChannelPattern]:
        """Get all detected channel patterns."""
        if not self._initialized:
            self.initialize()
        return self._patterns
    
    def get_categories(self) -> Dict[str, int]:
        """Get channel categories and their counts."""
        if not self._initialized:
            self.initialize()
        
        category_counts = defaultdict(int)
        for channel_info in self._channels.values():
            if channel_info.category:
                category_counts[channel_info.category] += 1
        
        return dict(category_counts)
    
    def search_channels_by_name(self, name_pattern: str) -> List[str]:
        """Search channels by name pattern."""
        if not self._initialized:
            self.initialize()
        
        pattern = re.compile(name_pattern, re.IGNORECASE)
        matching_channels = []
        
        for channel_id, channel_info in self._channels.items():
            if pattern.search(channel_info.name):
                matching_channels.append(channel_id)
        
        return matching_channels


def get_channel_intelligence() -> ChannelIntelligence:
    """Get or create ChannelIntelligence instance with caching."""
    if "channel_intelligence" not in st.session_state:
        token = st.secrets.get("SLACK_USER_TOKEN")
        if not token:
            raise RuntimeError("Missing SLACK_USER_TOKEN environment variable.")
        client = WebClient(token=token)
        st.session_state["channel_intelligence"] = ChannelIntelligence(client)
    
    intelligence = st.session_state["channel_intelligence"]
    intelligence.initialize()
    return intelligence


__all__ = ["ChannelIntelligence", "ChannelInfo", "ChannelPattern", "get_channel_intelligence"]
