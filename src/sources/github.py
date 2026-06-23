import os
import time
import requests
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class GitHubSource:
    """
    A source plugin for discovering repositories on GitHub using the REST API.
    Handles pagination, rate limiting, and filtering.
    """
    
    BASE_URL = "https://api.github.com"
    
    def __init__(self, token: Optional[str] = None):
        """
        Initialize the GitHub source.
        Args:
            token: GitHub Personal Access Token. If not provided, it will try to read GITHUB_TOKEN from env.
        """
        self.token = token or os.environ.get("GITHUB_TOKEN")
        self.session = requests.Session()
        
        if self.token:
            self.session.headers.update({
                "Authorization": f"token {self.token}",
                "Accept": "application/vnd.github.v3+json"
            })
        else:
            logger.warning("No GITHUB_TOKEN provided. API rate limits will be strictly limited (60 requests/hr).")

    def _handle_rate_limit(self, response: requests.Response):
        """Check rate limits and sleep if necessary."""
        if response.status_code == 403 and "rate limit exceeded" in response.text.lower():
            reset_time = int(response.headers.get("X-RateLimit-Reset", 0))
            sleep_duration = max(0, reset_time - int(time.time())) + 5
            logger.warning(f"Rate limit exceeded. Sleeping for {sleep_duration} seconds.")
            time.sleep(sleep_duration)
            return True
        return False

    def search_repositories(self, query: str, max_results: int = 100, sort: str = "stars", order: str = "desc") -> List[Dict[str, Any]]:
        """
        Search for repositories matching the query.
        
        Args:
            query: GitHub search syntax (e.g., 'language:go stars:>1000')
            max_results: Maximum number of repositories to return
            sort: Field to sort by (stars, forks, updated)
            order: Sort order (asc, desc)
            
        Returns:
            List of repository metadata dictionaries.
        """
        endpoint = f"{self.BASE_URL}/search/repositories"
        results = []
        page = 1
        per_page = min(100, max_results)
        
        logger.info(f"Searching GitHub for: '{query}' (max {max_results} results)")
        
        while len(results) < max_results:
            params = {
                "q": query,
                "sort": sort,
                "order": order,
                "per_page": per_page,
                "page": page
            }
            
            response = self.session.get(endpoint, params=params)
            
            if self._handle_rate_limit(response):
                continue
                
            response.raise_for_status()
            data = response.json()
            
            items = data.get("items", [])
            if not items:
                break
                
            results.extend(items)
            
            if len(items) < per_page or len(results) >= max_results:
                break
                
            page += 1
            # Respect search API rate limit (30 requests per minute)
            time.sleep(2)
            
        # Trim to exact max_results
        results = results[:max_results]
        
        # Standardize output
        standardized = []
        for repo in results:
            standardized.append({
                "source": "github",
                "id": repo["full_name"],
                "url": repo["html_url"],
                "clone_url": repo["clone_url"],
                "default_branch": repo["default_branch"],
                "stars": repo["stargazers_count"],
                "language": repo["language"],
                "updated_at": repo["updated_at"],
                "license": repo["license"]["spdx_id"] if repo.get("license") else None,
                "metadata": repo
            })
            
        logger.info(f"Discovered {len(standardized)} repositories.")
        return standardized
