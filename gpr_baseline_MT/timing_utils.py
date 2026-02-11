"""
Timing utilities for tracking GPR training and inference performance.

Usage:
    from timing_utils import Timer, TimingLogger

    # Simple timing
    with Timer("training"):
        model.fit(X, y)

    # With logging to CSV
    logger = TimingLogger("experiment_timings.csv")

    with logger.time("training", n_train=1000, n_features=16):
        model.fit(X, y)

    with logger.time("inference", n_test=200, n_features=16):
        predictions = model.predict(X_test)

    # View logged times
    logger.summary()
"""

import time
import csv
import os
from datetime import datetime
from contextlib import contextmanager
import pandas as pd


class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, name="", verbose=True):
        self.name = name
        self.verbose = verbose
        self.elapsed = None

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        if self.verbose:
            print(f"[{self.name}] {self.elapsed:.3f}s")


class TimingLogger:
    """
    Logger that records timing information to CSV with metadata.

    Parameters
    ----------
    filepath : str
        Path to CSV file for logging. Created if doesn't exist.
    experiment_name : str, optional
        Name for this experiment/run.

    Example
    -------
    logger = TimingLogger("timings.csv", experiment_name="baseline_gpr")

    with logger.time("training", n_train=1000, n_features=16, kernel="RBF"):
        train_model()

    with logger.time("inference", n_test=200):
        predict()
    """

    def __init__(self, filepath="timings.csv", experiment_name=None):
        self.filepath = filepath
        self.experiment_name = experiment_name or "default"
        self.records = []

        # Load existing records if file exists
        if os.path.exists(filepath):
            try:
                self.records = pd.read_csv(filepath).to_dict('records')
            except:
                self.records = []

    @contextmanager
    def time(self, operation, verbose=True, **metadata):
        """
        Time a code block and log with metadata.

        Parameters
        ----------
        operation : str
            Name of operation (e.g., "training", "inference", "hyperopt")
        verbose : bool
            Print timing when block completes
        **metadata : dict
            Any additional metadata to log (n_train, n_features, kernel, etc.)

        Yields
        ------
        dict
            Record dict that will be populated with timing info
        """
        record = {
            'timestamp': datetime.now().isoformat(),
            'experiment': self.experiment_name,
            'operation': operation,
            **metadata
        }

        start = time.perf_counter()
        try:
            yield record
        finally:
            elapsed = time.perf_counter() - start
            record['elapsed_seconds'] = elapsed

            if verbose:
                meta_str = ", ".join(f"{k}={v}" for k, v in metadata.items() if v is not None)
                print(f"[{operation}] {elapsed:.3f}s ({meta_str})")

            self.records.append(record)
            self._save()

    def _save(self):
        """Save records to CSV."""
        if not self.records:
            return

        df = pd.DataFrame(self.records)
        df.to_csv(self.filepath, index=False)

    def log(self, operation, elapsed_seconds, **metadata):
        """Manually log a timing record."""
        record = {
            'timestamp': datetime.now().isoformat(),
            'experiment': self.experiment_name,
            'operation': operation,
            'elapsed_seconds': elapsed_seconds,
            **metadata
        }
        self.records.append(record)
        self._save()

    def summary(self, operation=None):
        """
        Print summary of logged timings.

        Parameters
        ----------
        operation : str, optional
            Filter to specific operation
        """
        if not self.records:
            print("No timing records.")
            return

        df = pd.DataFrame(self.records)

        if operation:
            df = df[df['operation'] == operation]

        print(f"\n{'='*60}")
        print(f"Timing Summary: {self.filepath}")
        print(f"{'='*60}")

        # Group by operation
        for op in df['operation'].unique():
            op_df = df[df['operation'] == op]
            times = op_df['elapsed_seconds']
            print(f"\n{op}:")
            print(f"  Count: {len(times)}")
            print(f"  Mean:  {times.mean():.3f}s")
            print(f"  Std:   {times.std():.3f}s")
            print(f"  Min:   {times.min():.3f}s")
            print(f"  Max:   {times.max():.3f}s")

        return df

    def get_dataframe(self):
        """Return records as DataFrame for analysis."""
        return pd.DataFrame(self.records)


def format_time(seconds):
    """Format seconds into human-readable string."""
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{int(hours)}h {int(minutes)}m"


# Convenience function for quick timing
@contextmanager
def timed(name="", verbose=True):
    """Simple timing context manager."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    if verbose:
        print(f"[{name}] {format_time(elapsed)}")
