import logging
from typing import List, Dict, Any, Optional
from src.sources.github import GitHubSource

logger = logging.getLogger(__name__)

class Discoverer:
    """
    Core engine for discovering data sources (e.g. repositories, documents) across different platforms.
    Currently supports GitHub, but designed to be extensible to Arxiv, PubMed, etc.
    """
    
    def __init__(self):
        self.sources = {
            "github": GitHubSource()
            # Future sources:
            # "arxiv": ArxivSource()
            # "pubmed": PubMedSource()
        }

    def discover(self, source_name: str, query: str, max_results: int = 100, **kwargs) -> List[Dict[str, Any]]:
        """
        Discover items from a specific source.
        
        Args:
            source_name: Name of the source (e.g., 'github')
            query: The search query specific to the source
            max_results: Maximum items to return
            **kwargs: Additional source-specific arguments
            
        Returns:
            List of standardized metadata dictionaries representing the discovered items.
        """
        if source_name not in self.sources:
            raise ValueError(f"Unknown source: {source_name}. Available sources: {list(self.sources.keys())}")
            
        source = self.sources[source_name]
        logger.info(f"Running discovery on source '{source_name}' with query: '{query}'")
        
        if source_name == "github":
            return source.search_repositories(query, max_results, **kwargs)
            
        return []

    def discover_code_guild_targets(self, language: str = "go", min_stars: int = 1000, max_results: int = 100) -> List[Dict[str, Any]]:
        """
        Helper method specifically for the Code Guild.
        Finds high-quality repositories for a given language.
        """
        # Exclude common noisy or tutorial repositories to maintain quality
        query = f"language:{language} stars:>{min_stars} NOT awesome NOT tutorial NOT 'learn {language}'"
        return self.discover("github", query=query, max_results=max_results)

# Example usage for testing:
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    discoverer = Discoverer()
    # Find top 5 Go repositories
    repos = discoverer.discover_code_guild_targets(language="go", min_stars=5000, max_results=5)
    
    for repo in repos:
        print(f"{repo['id']} (⭐ {repo['stars']}) - {repo['clone_url']}")
