from __future__ import annotations
import numpy as np, pandas as pd
from typing import Iterator, Tuple
class PurgedGroupTimeSeriesSplit:
    def __init__(self, n_splits: int = 5, purge_hours: int = 24):
        self.n_splits = n_splits; self.purge = pd.Timedelta(hours=purge_hours)
    def split(self, X: pd.DataFrame) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        df = X[['time','group']].copy().sort_values('time')
        times = df['time'].sort_values().unique()
        windows = np.array_split(times, self.n_splits)
        for i in range(self.n_splits):
            test_times = pd.Index(windows[i])
            if len(test_times)==0: continue
            test_start = test_times[0]; purge_start = test_start - self.purge
            test_idx = df.index[df['time'].isin(test_times)].values
            test_groups = set(df.loc[test_idx, 'group'])
            train_idx = df.index[(df['time'] < purge_start) & (~df['group'].isin(test_groups))].values
            yield train_idx, test_idx


class PurgedTimeSeriesSplit:
    """Time-ordered CV with a purge window; does NOT exclude groups."""
    def __init__(self, n_splits: int = 4, purge_hours: int = 12):
        self.n_splits = n_splits
        self.purge = pd.Timedelta(hours=purge_hours)

    def split(self, X: pd.DataFrame):
        df = X[['time']].copy().sort_values('time')
        times = df['time'].sort_values().unique()
        windows = np.array_split(times, self.n_splits)
        for i in range(1, self.n_splits):  # start at 1 to ensure some history exists
            test_times = pd.Index(windows[i])
            if len(test_times)==0:
                continue
            test_start = test_times[0]
            purge_start = test_start - self.purge
            train_idx = df.index[df['time'] < purge_start].values
            test_idx = df.index[df['time'].isin(test_times)].values
            yield train_idx, test_idx
