import os
import logging
import subprocess
import concurrent.futures
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class Downloader:
    """
    Clones Git repositories concurrently to a local directory.
    """
    
    def __init__(self, output_dir: str = "data/raw"):
        """
        Args:
            output_dir: Base directory where repositories will be cloned.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def _clone_repo(self, repo_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Worker function to clone a single repository.
        Returns the updated repo_info with local_path added.
        """
        clone_url = repo_info["clone_url"]
        # Convert user/repo string to a valid path: data/raw/user_repo
        safe_name = repo_info["id"].replace("/", "_")
        target_path = os.path.join(self.output_dir, safe_name)
        
        repo_info["local_path"] = target_path
        
        if os.path.exists(target_path):
            logger.debug(f"Repo {repo_info['id']} already exists at {target_path}. Skipping clone.")
            repo_info["status"] = "cached"
            return repo_info
            
        logger.info(f"Cloning {repo_info['id']} to {target_path}...")
        try:
            # Shallow clone to save bandwidth and time (--depth 1)
            cmd = ["git", "clone", "--depth", "1", clone_url, target_path]
            subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True
            )
            repo_info["status"] = "success"
            logger.info(f"Successfully cloned {repo_info['id']}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone {repo_info['id']}: {e.stderr}")
            repo_info["status"] = "failed"
            repo_info["error"] = e.stderr
            
        return repo_info

    def download_all(self, repositories: List[Dict[str, Any]], max_workers: int = 4) -> List[Dict[str, Any]]:
        """
        Concurrently clones all provided repositories.
        """
        logger.info(f"Starting concurrent download of {len(repositories)} repositories (max_workers={max_workers}).")
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_repo = {executor.submit(self._clone_repo, repo): repo for repo in repositories}
            for future in concurrent.futures.as_completed(future_to_repo):
                try:
                    res = future.result()
                    results.append(res)
                except Exception as exc:
                    repo = future_to_repo[future]
                    logger.error(f"Repository {repo['id']} generated an exception during clone: {exc}")
                    repo["status"] = "exception"
                    repo["error"] = str(exc)
                    results.append(repo)
                    
        success_count = sum(1 for r in results if r.get("status") in ("success", "cached"))
        logger.info(f"Download complete: {success_count}/{len(repositories)} successful.")
        
        return results

# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    dl = Downloader(output_dir="data/raw_test")
    test_repos = [
        {"id": "go-chi/chi", "clone_url": "https://github.com/go-chi/chi.git"},
        {"id": "sirupsen/logrus", "clone_url": "https://github.com/sirupsen/logrus.git"}
    ]
    dl.download_all(test_repos)
