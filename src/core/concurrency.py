"""ConcurrencyManager — bounded executor, semaphore, and job counter."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor


class ConcurrencyManager:
    """Encapsulates GPU-job concurrency. Instantiate once in the caller; inject as needed.

    No module-level singleton — the caller (server.py) creates the instance.
    """

    def __init__(self, concurrent_jobs: int) -> None:
        self.semaphore = asyncio.Semaphore(concurrent_jobs)
        self._executor = ThreadPoolExecutor(max_workers=max(2, concurrent_jobs + 1))
        self._jobs = 0

    async def run_in_thread(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    def inc_jobs(self) -> None:
        self._jobs += 1

    def dec_jobs(self) -> None:
        self._jobs -= 1

    def job_depth(self) -> int:
        return self._jobs
