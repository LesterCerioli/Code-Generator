import httpx
import logging
from typing import List, Dict
from app.services.utils.retry import retry_async

logger = logging.getLogger(__name__)

class DeepSeekClient:
    def __init__(self, api_url: str, timeout: float = 60.0):
        self.api_url = api_url
        self.timeout = timeout

    async def _post_requirements(self, requirements: str) -> Dict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self.api_url, json={"requirements": requirements})
            resp.raise_for_status()
            return resp.json()

    async def generate_code(self, requirements: str) -> List[Dict[str, str]]:
        
        def is_transient_http(exc: Exception) -> bool:
            import httpx
            if isinstance(exc, httpx.TransportError):
                return True
            if isinstance(exc, httpx.HTTPStatusError):
                code = exc.response.status_code
                return 500 <= code < 600
            return False

        payload = await retry_async(
            self._post_requirements,
            requirements,
            retries=4,
            initial_delay=0.5,
            backoff_factor=2.0,
            retry_exceptions=(httpx.TransportError, httpx.HTTPStatusError),
            retry_on_predicate=is_transient_http,
        )

        if isinstance(payload, dict) and "files" in payload and isinstance(payload["files"], list):
            return payload["files"]
        if isinstance(payload, dict) and "code" in payload:
            return [{"path": "generated_code.py", "content": payload["code"]}]
        raise RuntimeError("DeepSeek returned unexpected payload structure")
