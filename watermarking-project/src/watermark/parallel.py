"""Run one function over many images on several CPU cores.

Windows notes (this is why the code looks the way it does):
  * Windows starts worker processes with "spawn": each worker re-imports the
    package, so the function you pass must live in a module (watermark.jobs),
    not inside a script, and every script must keep `if __name__ == "__main__":`.
  * OpenCV and scikit-learn each start their own threads. With N worker
    processes that multiplies, so every worker is limited to one thread.
We use "spawn" on every OS so a run that works here works on Windows too.
"""
import multiprocessing as mp
import os
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm


def default_workers():
    """All cores but one, so the laptop stays usable."""
    return max(1, (os.cpu_count() or 2) - 1)


def _init_worker():
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = "1"
    import cv2
    cv2.setNumThreads(1)


def _safe_call(fn, item):
    try:
        return item, fn(item), None
    except Exception:  # one bad image must not kill a two-hour run
        return item, None, traceback.format_exc(limit=3)


def pmap(fn, items, workers=None, desc=None):
    """Yield (item, result, error) as each item finishes (not in input order).

    workers=1 runs in this process (easiest for debugging with print/pdb).
    error is None on success, otherwise a short traceback string.
    """
    items = list(items)
    workers = default_workers() if workers is None else max(1, int(workers))
    if workers == 1 or len(items) <= 1:
        for it in tqdm(items, desc=desc):
            yield _safe_call(fn, it)
        return
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker) as ex:
        futures = [ex.submit(_safe_call, fn, it) for it in items]
        for f in tqdm(as_completed(futures), total=len(futures), desc=desc):
            yield f.result()
