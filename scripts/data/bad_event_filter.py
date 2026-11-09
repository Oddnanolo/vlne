#!/usr/bin/env python
"""Filter out unphysical values for a given input HDF file."""

import sys
import h5py
import numpy as np

MAX_GEV = 10.0
COLS = ["particle.KE"] + [f"particle.startMomentum.{c}" for c in "txyz"]

src, dst = sys.argv[1:3]
with h5py.File(src, "r") as fs, h5py.File(dst, "w") as fd:
    n = len(fs[COLS[0]])
    bad = np.zeros(n, dtype=bool)
    for col in COLS:
        # written as "not all <=" so NaN and inf are caught too
        bad |= [not np.all(np.abs(v) <= MAX_GEV) for v in fs[col][:]]
    bad |= [np.any(v < 0) for v in fs["particle.KE"][:]]

    for name, node in fs.items():
        fd.create_dataset(name, data=node[:][~bad], dtype=node.dtype,
                          maxshape=(None,), chunks=True)
print(f"kept {n - bad.sum()} of {n} events, dropped {bad.sum()}")
