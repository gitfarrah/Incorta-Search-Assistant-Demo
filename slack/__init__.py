"""Slack search and channel intelligence module."""

from .slack_search import search_slack_simplified
from .channel_intelligence import get_channel_intelligence

__all__ = ["search_slack_simplified", "get_channel_intelligence"]

