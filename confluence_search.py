"""
Confluence search module.

Uses Atlassian Confluence REST API to perform searches and return normalized results.

Environment:
- CONFLUENCE_URL
- CONFLUENCE_EMAIL
- CONFLUENCE_API_TOKEN
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

from atlassian import Confluence


logger = logging.getLogger(__name__)


def _get_confluence_client() -> Confluence:
    """
    Construct a Confluence client using env credentials.

    Raises:
        RuntimeError: If required env vars are missing.
    """
    url = os.getenv("CONFLUENCE_URL")
    email = os.getenv("CONFLUENCE_EMAIL")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")
    
    if not (url and email and api_token):
        raise RuntimeError(
            "Missing Confluence credentials. Ensure CONFLUENCE_URL, "
            "CONFLUENCE_EMAIL, and CONFLUENCE_API_TOKEN are set."
        )
    
    return Confluence(url=url, username=email, password=api_token, cloud=True)


def search_confluence(query: str, max_results: int = 10, space_key: Optional[str] = None) -> List[dict]:
    """
    Search Confluence using CQL or alternative methods.

    Args:
        query: Free-text search query.
        max_results: Max number of pages to return (default 10).
        space_key: Optional Confluence space key to restrict results.

    Returns:
        List of dictionaries with keys: title, excerpt, url, space, last_modified.
    """
    if not query or not query.strip():
        logger.warning("Empty query provided to search_confluence")
        return []

    client = _get_confluence_client()
    results: List[dict] = []

    try:
        logger.info(f"Searching Confluence for: '{query}'")
        
        # Try multiple CQL approaches
        cql_queries = [
            # Approach 1: siteSearch (most comprehensive)
            f'siteSearch ~ "{query}"',
            # Approach 2: text search with type filter
            f'text ~ "{query}" AND type = page',
            # Approach 3: title or text search
            f'(title ~ "{query}" OR text ~ "{query}") AND type = page',
        ]
        
        # Add space filter if provided
        if space_key and space_key.strip():
            cql_queries = [f'{cql} AND space = "{space_key.strip()}"' for cql in cql_queries]
        
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
                        base_url = os.getenv("CONFLUENCE_URL", "").rstrip("/")
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
    """
    Alternative search using get_all_pages_from_space.
    
    This is a fallback when CQL search doesn't work.
    """
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
                    
                    # Simple text matching
                    if query.lower() in title.lower() or query.lower() in body.lower():
                        base_url = os.getenv("CONFLUENCE_URL", "").rstrip("/")
                        webui = page.get("_links", {}).get("webui", "")
                        url = f"{base_url}{webui}" if base_url and webui else ""
                        
                        # Get excerpt from body
                        excerpt = body[:300] if body else ""
                        
                        results.append({
                            "title": title,
                            "excerpt": excerpt,
                            "url": url,
                            "space": space_name,
                            "last_modified": "Recent"
                        })
                        
            except Exception as e:
                logger.warning(f"Failed to search space {space_key}: {e}")
                continue
        
        logger.info(f"Alternative search found {len(results)} results")
        
    except Exception as e:
        logger.error(f"Alternative search failed: {e}")
    
    return results


__all__ = ["search_confluence"]