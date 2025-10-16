import base64
import httpx
import asyncio
import logging
from typing import Dict, Optional

GITHUB_API_BASE = "https://api.github.com"
logger = logging.getLogger(__name__)

class GitHubClient:
    def __init__(self, token: str, concurrency_limit: int = 4, timeout: float = 30.0):
        self.token = token
        self.timeout = timeout
        self._sem = asyncio.Semaphore(concurrency_limit)
        self._headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "CodeGenMicroservice/1.0"
        }

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        async with self._sem:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.request(method, url, headers=self._headers, **kwargs)
                if resp.status_code == 403 and resp.headers.get("X-RateLimit-Remaining") == "0":
                    reset = int(resp.headers.get("X-RateLimit-Reset", "0"))
                    to_wait = max(reset - int(httpx._models.datetime.datetime.now().timestamp()), 1)
                    logger.warning("GitHub rate limit reached, sleeping %ds", to_wait)
                    await asyncio.sleep(to_wait)
                    resp = await client.request(method, url, headers=self._headers, **kwargs)
                resp.raise_for_status()
                return resp

    async def create_repository(self, name: str, private: bool = True, description: str = "") -> Dict:
        url = f"{GITHUB_API_BASE}/user/repos"
        payload = {"name": name, "private": private, "description": description}
        resp = await self._request("POST", url, json=payload)
        return resp.json()

    async def put_file(self, owner: str, repo: str, path: str, content: str, message: str, branch: Optional[str] = None) -> Dict:
        url = f"{GITHUB_API_BASE}/repos/{owner}/{repo}/contents/{path}"
        content_b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")
        payload = {"message": message, "content": content_b64}
        if branch:
            payload["branch"] = branch
        resp = await self._request("PUT", url, json=payload)
        return resp.json()
