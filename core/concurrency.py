"""Process-wide GPU-job concurrency: a bounded executor + semaphore + run-counter.

The semaphore gates concurrent GPU work and MUST be acquired before dispatching to the
executor. The counter (queued + running) backs ``GET /status``.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from config import settings


class ConcurrencyManager:
    """Encapsulates the process-wide executor, semaphore, and job counter.

    All mutable state lives inside this instance — no module-level globals.
    The module-level ``manager`` singleton is the single source of truth.
    """

    def __init__(self, concurrent_jobs: int) -> None:
        # Gate concurrent GPU jobs; acquire BEFORE dispatching to the executor.
        self.semaphore = asyncio.Semaphore(concurrent_jobs)
        self._executor = ThreadPoolExecutor(max_workers=max(2, concurrent_jobs + 1))
        self._jobs = 0  # queued + running, for /status

    async def run_in_thread(self, fn, *args):
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, fn, *args)

    def inc_jobs(self) -> None:
        self._jobs += 1

    def dec_jobs(self) -> None:
        self._jobs -= 1

    def job_depth(self) -> int:
        return self._jobs


# Module-level singleton — all callers import this module and use its attributes.
manager = ConcurrencyManager(settings.CONCURRENT_JOBS)

# Module-level re-exports for backward-compatible call sites:
#   concurrency.semaphore  → manager.semaphore
#   concurrency.run_in_thread(...)  → manager.run_in_thread(...)
#   concurrency.inc_jobs()  → manager.inc_jobs()
#   concurrency.dec_jobs()  → manager.dec_jobs()
#   concurrency.job_depth()  → manager.job_depth()
semaphore = manager.semaphore
run_in_thread = manager.run_in_thread
inc_jobs = manager.inc_jobs
dec_jobs = manager.dec_jobs
job_depth = manager.job_depth
