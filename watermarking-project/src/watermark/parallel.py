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


MAX_DEFAULT_WORKERS = 4   # each worker loads its own copy of the model (~100-200 MB)


def default_workers():
    """All cores but one, capped at 4 so a laptop does not run out of memory."""
    return max(1, min(MAX_DEFAULT_WORKERS, (os.cpu_count() or 2) - 1))


def check_failures(failed, total, what="images", limit=0.1):
    """Stop the script if more than `limit` of the jobs failed, with the usual fix.

    Results computed from a few surviving images look normal but mean nothing,
    so it is better to stop than to print them.
    """
    if total and len(failed) / total > limit:
        mem = any("Memory" in str(e) for _, e in failed)
        hint = ("Out of memory: re-run with fewer worker processes, e.g. --workers 2 "
                "(close other programs too)." if mem else "See the first error above.")
        raise SystemExit(f"\n{len(failed)} of {total} {what} failed, too many to trust the results. {hint}")


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