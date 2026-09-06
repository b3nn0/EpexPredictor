import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

import pandas as pd

from predictor.model.priceregion import PriceRegionName

log = logging.getLogger(__name__)


@dataclass
class _EvalPlotEntry:
    data: pd.DataFrame
    created_at: datetime
    task: asyncio.Task | None = None  # in-flight training/refresh, if any


class EvalPlotCache:
    """
    Caches trained eval-plot data (predicted vs. actual prices) per (region, range) for 30 minutes.

    Training the plot is expensive (a full 2-stage model stack per historical
    window), and this endpoint is often called repeatedly with the same
    parameters. Fresh entries are served as-is. Once the TTL is reached the
    cached version is still returned, but an async refresh is triggered -
    deduplicated so a burst of requests only ever runs one refresh per key.
    A cold entry blocks its first caller until the initial training is done;
    concurrent callers queue on the same task.
    """
    TTL: timedelta = timedelta(minutes=30)

    _cache: dict[tuple, _EvalPlotEntry]
    _lock: asyncio.Lock
    _trainer: Callable[..., Awaitable[pd.DataFrame]]

    def __init__(self, trainer: Callable[..., Awaitable[pd.DataFrame]]):
        self._cache = {}
        self._lock = asyncio.Lock()
        # trainer(name: PriceRegionName, region: PriceRegion, start_ts, end_ts) -> pd.DataFrame
        self._trainer = trainer

    @staticmethod
    def _key(region: PriceRegionName, start_ts: datetime, end_ts: datetime) -> tuple:
        # Normalize to whole minutes so trivial query-param jitter still hits the cache
        return (
            region,
            start_ts.astimezone(timezone.utc).replace(second=0, microsecond=0),
            end_ts.astimezone(timezone.utc).replace(second=0, microsecond=0),
        )

    async def get(self, region: PriceRegionName, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
        key = self._key(region, start_ts, end_ts)
        stale = False
        async with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                entry = _EvalPlotEntry(data=pd.DataFrame(), created_at=datetime(1970, 1, 1, tzinfo=timezone.utc))
                self._cache[key] = entry
            elif datetime.now(timezone.utc) - entry.created_at < self.TTL:
                return entry.data
            else:
                stale = True

            # Start training if nothing is in flight yet (cold entry, or stale entry due for refresh)
            if entry.task is None:
                entry.task = asyncio.create_task(self._run(key, entry, region, start_ts, end_ts))
            task = entry.task

        if stale:
            # Serve the cached version now; the refresh runs in the background
            return entry.data

        # Cold entry: wait for the initial training to finish
        await task
        return entry.data

    async def _run(self, key: tuple, entry: _EvalPlotEntry, region: PriceRegionName,
                   start_ts: datetime, end_ts: datetime):
        try:
            entry.data = await self._trainer(region, region.to_region(), start_ts, end_ts)
            entry.created_at = datetime.now(timezone.utc)
        except Exception:
            # Leave no broken entry behind, so the next request retries
            async with self._lock:
                if self._cache.get(key) is entry:
                    del self._cache[key]
            raise
        finally:
            entry.task = None
