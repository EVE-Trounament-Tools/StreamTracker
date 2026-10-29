"""Send each finished match to the observed-import endpoint, one match object per POST.

The endpoint URL and token come from the environment (OBSERVED_IMPORT_URL,
OBSERVED_IMPORT_TOKEN) and are never stored in the repo. Posting runs on a
background thread so a slow or failing endpoint never stalls tracking, and
the local database is written either way.
"""

import json
import logging
import os
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

log = logging.getLogger(__name__)

URL_ENV = "OBSERVED_IMPORT_URL"
TOKEN_ENV = "OBSERVED_IMPORT_TOKEN"
TIMEOUT_S = 30
RETRIES = 3  # after the first try, for 5xx and network errors; reposting a match is safe
RETRY_DELAY_S = 5.0


class ImportClient:
    def __init__(self, url: str, token: str, retry_delay_s: float = RETRY_DELAY_S):
        self.url = url
        self.token = token
        self.retry_delay_s = retry_delay_s
        self._pool = ThreadPoolExecutor(max_workers=1)

    @classmethod
    def from_env(cls) -> "ImportClient":
        url, token = os.environ.get(URL_ENV), os.environ.get(TOKEN_ENV)
        missing = [name for name, value in ((URL_ENV, url), (TOKEN_ENV, token)) if not value]
        if missing:
            raise SystemExit(f"--post needs {' and '.join(missing)} set in the environment")
        return cls(url, token)

    def submit(self, match: dict) -> None:
        self._pool.submit(self._post, match)

    def close(self) -> None:
        self._pool.shutdown(wait=True)

    def _post(self, match: dict) -> dict | None:
        """POST one match. Returns the 200 body, or None; never raises."""
        key = f"{match.get('run_id')}:{match.get('match_id')}"
        body = json.dumps(match).encode()
        for attempt in range(RETRIES + 1):
            if attempt:
                time.sleep(self.retry_delay_s * attempt)
            req = urllib.request.Request(self.url, data=body, method="POST", headers={
                "x-import-token": self.token,
                "content-type": "application/json",
            })
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT_S) as resp:
                    result = json.loads(resp.read() or b"{}")
            except urllib.error.HTTPError as e:
                detail = e.read().decode(errors="replace")[:500]
                if e.code < 500:
                    log.error("import %s refused (%d): %s", key, e.code, detail)
                    return None
                log.warning("import %s failed (%d, try %d): %s", key, e.code, attempt + 1, detail)
                continue
            except Exception as e:  # network errors, timeouts, bad JSON
                log.warning("import %s failed (try %d): %s", key, attempt + 1, e)
                continue
            log.info("import %s: %s, game %s", key, result.get("action"), result.get("game_no"))
            if result.get("reasons"):
                log.warning("import %s held for review: %s", key, result["reasons"])
            if result.get("warnings"):
                log.warning("import %s warnings (post again to repair): %s", key, result["warnings"])
            return result
        log.error("import %s gave up after %d tries", key, RETRIES + 1)
        return None
