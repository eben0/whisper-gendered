"""Process-wide GPU-job concurrency: a bounded executor + semaphore + run-counter.

The semaphore gates concurrent GPU work and MUST be acquired before dispatching to the
executor. The counter (queued + running) backs ``GET /status``.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

from config import settings

# Gate concurrent GPU jobs; acquire BEFORE dispatching to the executor.
semaphore = asyncio.Semaphore(settings.CONCURRENT_JOBS)
executor = ThreadPoolExecutor(max_workers=max(2, settings.CONCURRENT_JOBS + 1))

_jobs_in_system = 0  # queued + running, for /status


async def run_in_thread(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(executor, fn, *args)


def inc_jobs() -> None:
    global _jobs_in_system
    _jobs_in_system += 1


def dec_jobs() -> None:
    global _jobs_in_system
    _jobs_in_system -= 1


def job_depth() -> int:
    return _jobs_in_system
