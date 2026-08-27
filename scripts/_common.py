"""Shared bits for the figure scripts: paths, argparse, simple npz cache."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

FIG_DIR = os.path.join(ROOT, "figures")
CACHE_DIR = os.path.join(ROOT, "cache")
DATA_DIR = os.path.join(ROOT, "data")


def parser(desc):
    p = argparse.ArgumentParser(description=desc)
    p.add_argument("--quick", action="store_true",
                   help="small/short simulations to check that everything runs "
                        "(figures will NOT match the paper)")
    p.add_argument("--out", default=FIG_DIR, help="output directory for figures")
    p.add_argument("--no-cache", action="store_true", help="ignore cached simulations")
    return p


def cached(name, fn, use_cache=True):
    """Run ``fn()`` (returning a dict of arrays) unless ``cache/<name>.npz`` exists."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, name + ".npz")
    if use_cache and os.path.exists(path):
        print("  [cache]", name)
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    print("  [run]  ", name, flush=True)
    out = fn()
    np.savez_compressed(path, **{k: np.asarray(v) for k, v in out.items()})
    return out


def _job(name, fn):
    out = fn()
    np.savez_compressed(os.path.join(CACHE_DIR, name + ".npz"),
                        **{k: np.asarray(v) for k, v in out.items()})


def cached_parallel(jobs, workers=1, use_cache=True):
    """``jobs``: list of (name, fn). Missing cache entries are computed, up to
    ``workers`` at a time in forked subprocesses (macOS/Linux), then all
    results are loaded from cache and returned as a dict name -> arrays."""
    import multiprocessing as mp
    os.makedirs(CACHE_DIR, exist_ok=True)
    todo = [(n, f) for n, f in jobs
            if not (use_cache and os.path.exists(os.path.join(CACHE_DIR, n + ".npz")))]
    if workers > 1 and len(todo) > 1 and hasattr(os, "fork"):
        ctx = mp.get_context("fork")
        running = []
        for n, f in todo:
            while len(running) >= workers:
                running = [pr for pr in running if pr.is_alive()]
                if len(running) >= workers:
                    running[0].join(timeout=5)
            print("  [run]  ", n, flush=True)
            pr = ctx.Process(target=_job, args=(n, f))
            pr.start()
            running.append(pr)
        for pr in running:
            pr.join()
    else:
        for n, f in todo:
            print("  [run]  ", n, flush=True)
            _job(n, f)
    return {n: cached(n, f, use_cache=True) for n, f in jobs}
