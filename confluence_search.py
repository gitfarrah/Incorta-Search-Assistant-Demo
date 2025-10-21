from __future__ import annotations

import logging
from typing import List, Optional

from atlassian import Confluence
import streamlit as st


logger = logging.getLogger(__name__)


def _get_confluence_client() -> Confluence:
    
    url = st.secrets.get("CONFLUENCE_URL")
    email = st.secrets.get("CONFLUENCE_EMAIL")
    api_token = st.secrets.get("CONFLUENCE_API_TOKEN")
    
    if not (url and email and api_token):
        raise RuntimeError(
            "Missing Confluence credentials. Ensure CONFLUENCE_URL, "
            "CONFLUENCE_EMAIL, and CONFLUENCE_API_TOKEN are set."
        )
    
    return Confluence(url=url, username=email, password=api_token, cloud=True)


def search_confluence(query: str, max_results: int = 10, space_key: Optional[str] = None) -> List[dict]:
    
    if not query or not query.strip():
        logger.warning("Empty query provided to search_confluence")
        return []

    client = _get_confluence_client()
    results: List[dict] = []

    try:
        logger.info(f"Searching Confluence for: '{query}'")
        
        # Try multiple CQL approaches with different search strategies
        # Split query into individual terms for more flexible matching
        query_terms = query.split()
        
        cql_queries = [
            # Approach 1: siteSearch (most comprehensive)
            f'siteSearch ~ "{query}"',
            # Approach 2: text search with type filter
            f'text ~ "{query}" AND type = page',
            # Approach 3: title or text search
            f'(title ~ "{query}" OR text ~ "{query}") AND type = page',
            # Approach 4: Search for individual terms (more flexible)
            f'text ~ "{query_terms[0]}" AND type = page' if query_terms else f'text ~ "{query}" AND type = page',
            # Approach 5: Title search only
            f'title ~ "{query}" AND type = page',
        ]
        
        # Add space filter if provided and not 'all'
        if space_key and str(space_key).strip().lower() not in {"all", "none", ""}:
            cql_queries = [f'{cql} AND space = "{str(space_key).strip()}"' for cql in cql_queries]
        
        # Try each CQL query until one works
        for i, cql in enumerate(cql_queries):
            try:
                logger.info(f"Trying CQL approach {i+1}: {cql}")
                
                resp = client.cql(cql=cql, limit=max_results)
                
                total = resp.get("totalSize", 0)
                logger.info(f"CQL approach {i+1} found {total} total results")
                
                items = resp.get("results", [])
                
                if items:
                    logger.info(f"Processing {len(items)} Confluence results")
                    
                    for item in items[:max_results]:
                        # Handle different response structures
                        content = item.get("content") or item
                        
                        # Extract title
                        title = content.get("title", "Untitled")
                        
                        # Extract space
                        space_data = content.get("space", {})
                        space_name = space_data.get("name", "Unknown") if isinstance(space_data, dict) else "Unknown"
                        
                        # Extract URL
                        base_url = (st.secrets.get("CONFLUENCE_URL", "") or "").rstrip("/")
                        links = content.get("_links", {})
                        webui = links.get("webui", "")
                        url = f"{base_url}{webui}" if base_url and webui else ""
                        
                        # Extract excerpt
                        excerpt = item.get("excerpt", "") or content.get("excerpt", "")
                        
                        # Extract last modified (simplified)
                        version = content.get("version", {})
                        last_modified = version.get("when", "Recent")
                        
                        logger.debug(f"Found page: {title}")
                        
                        results.append({
                            "title": title,
                            "excerpt": excerpt[:300] if excerpt else "",
                            "url": url,
                            "space": space_name,
                            "last_modified": last_modified
                        })
                    
                    # If we got results, break out of the loop
                    break
                else:
                    logger.warning(f"CQL approach {i+1} returned no results")
                    
            except Exception as e:
                logger.warning(f"CQL approach {i+1} failed: {e}")
                continue
        
        # If no CQL approach worked, try alternative method
        if not results:
            logger.info("Trying alternative search method...")
            results = _alternative_confluence_search(client, query, max_results)
        
        logger.info(f"Returning {len(results)} Confluence results")

    except Exception as e:
        logger.error(f"Confluence search error: {e}", exc_info=True)

    return results


def _alternative_confluence_search(client: Confluence, query: str, max_results: int) -> List[dict]:
    
    results: List[dict] = []
    
    try:
        # Get all spaces
        spaces_resp = client.get_all_spaces(limit=10)
        spaces = spaces_resp.get("results", [])
        
        logger.info(f"Searching across {len(spaces)} spaces")
        
        for space in spaces:
            if len(results) >= max_results:
                break
                
            space_key = space.get("key")
            space_name = space.get("name", "Unknown")
            
            try:
                # Get pages from this space
                pages = client.get_all_pages_from_space(
                    space_key,
                    start=0,
                    limit=20,
                    expand="body.view"
                )
                
                # Filter pages that match the query
                for page in pages:
                    if len(results) >= max_results:
                        break
                    
                    title = page.get("title", "")
                    body = page.get("body", {}).get("view", {}).get("value", "")
                    
                    # Enhanced text matching - check for individual terms
                    query_lower = query.lower()
                    title_lower = title.lower()
                    body_lower = body.lower()
                    
                    # Check for full query match first
                    if query_lower in title_lower or query_lower in body_lower:
                        match_score = 3  # High score for full match
                    else:
                        # Check for individual term matches
                        query_terms = query_lower.split()
                        term_matches = sum(1 for term in query_terms if term in title_lower or term in body_lower)
                        if term_matches > 0:
                            match_score = term_matches  # Score based on number of matching terms
                        else:
                            continue  # No match, skip this page
                    
                    base_url = (st.secrets.get("CONFLUENCE_URL", "") or "").rstrip("/")
                    webui = page.get("_links", {}).get("webui", "")
                    url = f"{base_url}{webui}" if base_url and webui else ""
                    
                    # Get excerpt from body
                    excerpt = body[:300] if body else ""
                    
                    results.append({
                        "title": title,
                        "excerpt": excerpt,
                        "url": url,
                        "space": space_name,
                        "last_modified": "Recent",
                        "match_score": match_score
                    })
                        
            except Exception as e:
                logger.warning(f"Failed to search space {space_key}: {e}")
                continue
        
        # Sort results by match score (highest first)
        results.sort(key=lambda x: x.get("match_score", 0), reverse=True)
        
        # Remove match_score from final results
        for result in results:
            result.pop("match_score", None)
        
        logger.info(f"Alternative search found {len(results)} results")
        
    except Exception as e:
        logger.error(f"Alternative search failed: {e}")
    
    return results


def search_confluence_optimized(intent_data: dict, user_query: str) -> List[dict]:
    """
    Optimized Confluence search using intent analysis results.
    """
    confluence_params = intent_data.get("confluence_params", {})
    keywords = confluence_params.get("keywords", [])
    spaces = confluence_params.get("spaces", None)
    limit = confluence_params.get("limit", 5)
    
    # Build optimized search query - avoid duplication
    # Extract distinct words from user query for better relevance
    query_words = set(user_query.lower().split())
    # Remove common stop words that don't add meaning
    stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were", "be", "been", "have", "has", "had", "do", "does", "did", "will", "would", "could", "should", "may", "might", "can", "what", "when", "where", "why", "how", "who", "which", "updates", "about", "on"}
    distinct_words = [word for word in query_words if word not in stop_words and len(word) > 2]
    
    # Build search query prioritizing distinct words
    if distinct_words:
        search_query = " ".join(distinct_words)
    else:
        search_query = user_query
    
    # Add keywords if they're not already in the query
    if keywords:
        existing_words = set(search_query.lower().split())
        new_keywords = [kw for kw in keywords if kw.lower() not in existing_words]
        if new_keywords:
            search_query = f"{search_query} {' '.join(new_keywords)}"
    
    logger.info(f"Optimized Confluence search query: {search_query}")
    logger.info(f"Spaces filter: {spaces}")
    
    return search_confluence(search_query, limit, spaces)


__all__ = ["search_confluence", "search_confluence_optimized"]