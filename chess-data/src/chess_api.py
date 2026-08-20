import requests
import time
from typing import List, Optional

class ChessAPI:
    def __init__(self, username: str):
        self.username = username
        self.base_url = "https://api.chess.com/pub"
        # Chess.com requires a valid User-Agent with contact info
        self.headers = {
            "User-Agent": f"ChessDataDownloader/1.0 (Contact: {username})"
        }
        self.session = requests.Session()
        self.session.headers.update(self.headers)

    def get_available_archives(self) -> List[str]:
        """
        Fetches a list of URLs for monthly archives available for the user.
        """
        url = f"{self.base_url}/player/{self.username}/games/archives"
        try:
            response = self.session.get(url)
            if response.status_code == 404:
                raise ValueError(f"User '{self.username}' not found on Chess.com")
            
            response.raise_for_status()
            data = response.json()
            return data.get("archives", [])
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Network error while fetching archives: {e}")
            
    def fetch_monthly_archive(self, archive_url: str) -> Optional[dict]:
        """
        Fetches the games for a given monthly archive URL.
        """
        # Sleep slightly to avoid hitting rate limits too hard if downloading many months
        time.sleep(0.5) 
        
        try:
            response = self.session.get(archive_url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching archive {archive_url}: {e}")
            return None
