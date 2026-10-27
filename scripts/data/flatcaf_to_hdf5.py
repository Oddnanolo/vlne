#!/usr/bin/env python3
"""Extract WireCell PF-style particle and event inputs from SBND *.wc.flat.caf.root
files and write them to a single HDF5 file.

Source (recTreeWireCell, all branches prefixed "wc."), uB-name -> SBND-name:
  T_PFeval reco particles   -> T_pf.*          (positions cm, ke_mev MeV, truncated)
  T_PFeval truth particles  -> T_truth_pf.*    (positions cm, E/KE/p GeV, t ns)
  T_BDTvars contain.full    -> T_tagger.match_isFC  (one row per reco nu candidate)
Reco particles have no momentum vector: startMomentum is built from KE (full
precision from T_kine.kine_energy_particle[kine_index]) + PDG mass, with
direction = unit(end - start), or for zero-length pseudo particles (pi0) the
direction of the summed momenta of their children.

HDF5 layout (one 1-D dataset per column, all tables flat, all energies GeV):
  /files                         input file paths, indexed by file_index
  /events/{...}                  one row per reco nu candidate (contain_full)
  /particles/{...}               reco particles; event_row -> row in /events
  /truth_particles/{...}         (--truth) T_truth_pf; nu_row -> row of that event's truth_nu
  /truth_nu/{...}                (--truth) T_truth_nu

Environment: run inside the cafpyana env (needs h5py), e.g.
  cd /exp/sbnd/app/users/eespinoz/cafpyana && source setup.sh && cd -

Usage:
  python3 wc_flatcaf_to_hdf5.py out.h5 FILE_OR_DIR [...] [--filelist list.txt] [--truth] [-j 8] [-n N] [--force]
Directories are searched one level deep for *.wc.flat.caf.root.
Each file is read whole into memory before parsing (uproot's small reads are
slow on dCache NFS); expect a few seconds per file per worker.

Read back:
  import h5py, pandas as pd
  with h5py.File("out.h5") as f:
      parts = pd.DataFrame({k: v[:] for k, v in f["particles"].items()})
"""
import argparse
import functools
import glob
import io
import multiprocessing
import os
import sys
import time
from datetime import datetime

import awkward as ak
import h5py
import numpy as np
import pandas as pd
import uproot

TREE = "recTreeWireCell"
PATTERN = "*.wc.flat.caf.root"
EVT = ["wc.run", "wc.subrun", "wc.evt"]
EVT_KEYS = ["run", "subrun", "evt"]

MASS_GEV = {11: 0.000511, 13: 0.105658, 22: 0.0, 111: 0.134977,
            211: 0.139570, 2112: 0.939565, 2212: 0.938272}
ONEHOT = {"electron": 11, "gamma": 22, "muon": 13, "neutron": 2112,
          "pion": 211, "pizero": 111, "proton": 2212}

UNITS = {"start_x": "cm", "start_y": "cm", "start_z": "cm", "start_t": "ns",
         "end_x": "cm", "end_y": "cm", "end_z": "cm", "end_t": "ns",
         "vtx_x": "cm", "vtx_y": "cm", "vtx_z": "cm", "t": "ns",
         "ke_mev": "MeV", "KE": "GeV", "E": "GeV", "edep": "GeV",
         "startMomentum_t": "GeV", "startMomentum_x": "GeV",
         "startMomentum_y": "GeV", "startMomentum_z": "GeV"}
DESCRIPTION = {
    "contain_full": "T_tagger.match_isFC: reco nu candidate fully contained (0/1)",
    "ke_mev": "T_pf.ke_mev as stored (truncated to integer MeV)",
    "KE": "kinetic energy; reco: T_kine.kine_energy_particle[kine_index], ke_mev for pseudo particles",
    "startMomentum_t": "total energy E = KE + m",
    "startMomentum_x": "reco: |p| * unit(end - start) (pi0: unit(sum of children p)); truth: T_truth_pf.start_px",
    "event_row": "row in /events for this particle's nu candidate (-1 if none)",
    "nu_row": "position of the truth neutrino within its event's /truth_nu rows",
    "file_index": "row in /files",
}


def add_pdg_onehot(df):
    apdg = df["pdg"].abs()
    for name, code in ONEHOT.items():
        df[f"pdg_{name}"] = (apdg == code).astype(np.int8)
    return df


def per_row(t, prefix, fields):
    """Flatten jagged branches prefix+field into one row per element, with run/subrun/evt."""
    arr = t.arrays([prefix + f for f in fields] + EVT, library="ak")
    n = ak.to_numpy(ak.num(arr[prefix + fields[0]]))
    cols = {k: np.repeat(ak.to_numpy(arr[b]), n) for k, b in zip(EVT_KEYS, EVT)}
    for f in fields:
        cols[f] = ak.to_numpy(ak.flatten(arr[prefix + f]))
    return pd.DataFrame(cols)


def reco_ke_gev(t):
    """Full-precision KE from T_kine.kine_energy_particle[kine_index], keyed by (entry, nu_index)."""
    a = t.arrays(["wc.T_kine.nu_index", "wc.T_kine.kine_energy_particle..idx",
                  "wc.T_kine.kine_energy_particle", "wc.T_pf.nu_index",
                  "wc.T_pf.kine_index", "wc.T_pf.ke_mev"], library="np")
    out = []
    for i in range(t.num_entries):
        row_start = dict(zip(a["wc.T_kine.nu_index"][i], a["wc.T_kine.kine_energy_particle..idx"][i]))
        flat = a["wc.T_kine.kine_energy_particle"][i]
        for nu, ki, ke in zip(a["wc.T_pf.nu_index"][i], a["wc.T_pf.kine_index"][i], a["wc.T_pf.ke_mev"][i]):
            # pseudo particles (kine_index == -1) only have the truncated ke_mev
            out.append(flat[row_start[nu] + ki] if ki >= 0 and nu in row_start else ke)
    return np.asarray(out, dtype=np.float64) / 1000.0


def reco_particles(t):
    df = per_row(t, "wc.T_pf.", ["nu_index", "node_id", "parent_id", "depth", "n_children",
                                 "is_pseudo", "kine_index", "pdg", "ke_mev",
                                 "start_x", "start_y", "start_z", "end_x", "end_y", "end_z"])
    df = add_pdg_onehot(df)
    ke = reco_ke_gev(t)
    m = df["pdg"].abs().map(MASS_GEV).fillna(0.0).to_numpy()
    p = np.sqrt(ke * ke + 2.0 * ke * m)
    d = df[["end_x", "end_y", "end_z"]].to_numpy() - df[["start_x", "start_y", "start_z"]].to_numpy()
    norm = np.linalg.norm(d, axis=1, keepdims=True)
    u = np.divide(d, norm, out=np.zeros_like(d), where=norm > 0)
    df["KE"] = ke
    df["startMomentum_t"] = ke + m
    df["startMomentum_x"], df["startMomentum_y"], df["startMomentum_z"] = (p[:, None] * u).T

    zero = (norm[:, 0] == 0) & (df["n_children"].to_numpy() > 0)
    if zero.any():
        # pseudo pi0 nodes have start == end: use the summed direction of their children (the photons)
        keys = EVT_KEYS + ["nu_index", "parent_id"]
        pcols = ["startMomentum_x", "startMomentum_y", "startMomentum_z"]
        kid_sum = df[keys + pcols].groupby(keys).sum()
        want = df.loc[zero, keys[:-1] + ["node_id"]].rename(columns={"node_id": "parent_id"})
        v = kid_sum.reindex(pd.MultiIndex.from_frame(want)).fillna(0.0).to_numpy()
        n = np.linalg.norm(v, axis=1, keepdims=True)
        df.loc[zero, pcols] = p[zero, None] * np.divide(v, n, out=np.zeros_like(v), where=n > 0)
    return df


def events(t):
    df = per_row(t, "wc.T_tagger.", ["nu_index", "match_isFC"])
    return df.rename(columns={"match_isFC": "contain_full"})


def truth_particles(t):
    df = per_row(t, "wc.T_truth_pf.", ["nu_row", "trackid", "parent_trackid", "mother_trackid",
                                       "pdg", "process", "E", "KE",
                                       "start_x", "start_y", "start_z", "start_t",
                                       "end_x", "end_y", "end_z", "end_t",
                                       "start_px", "start_py", "start_pz"])
    df = add_pdg_onehot(df)
    return df.rename(columns={"E": "startMomentum_t", "start_px": "startMomentum_x",
                              "start_py": "startMomentum_y", "start_pz": "startMomentum_z"})


def truth_nu(t):
    df = per_row(t, "wc.T_truth_nu.", ["nu_idx", "pdg", "ccnc", "mode", "int_type", "flavor",
                                       "E", "vtx_x", "vtx_y", "vtx_z", "t", "edep"])
    df.insert(3, "nu_row", df.groupby(EVT_KEYS).cumcount().astype(np.int32))
    return df


def open_root(path):
    if "://" in path:
        return uproot.open(path)
    # one sequential read is ~5x faster on /pnfs than uproot's many small reads
    with open(path, "rb") as fh:
        return uproot.open(io.BytesIO(fh.read()))


def extract(path, truth=False):
    """Read one file -> (path, {group: DataFrame}, error message or None)."""
    try:
        with open_root(path) as f:
            t = f[TREE]
            out = {"events": events(t), "particles": reco_particles(t)}
            if truth:
                out["truth_particles"] = truth_particles(t)
                out["truth_nu"] = truth_nu(t)
        return path, out, None
    except Exception as e:  # keep going on bad/missing files
        return path, None, f"{type(e).__name__}: {e}"


class H5Tables:
    """Append DataFrames to per-column resizable 1-D datasets under /<group>."""

    def __init__(self, h5):
        self.h5 = h5
        self.nrows = {}

    def append(self, group, df):
        if df.empty:
            return
        g = self.h5.require_group(group)
        start = self.nrows.get(group, 0)
        for c in df.columns:
            v = df[c].to_numpy()
            if c not in g:
                ds = g.create_dataset(c, shape=(0,), maxshape=(None,), dtype=v.dtype,
                                      chunks=(1 << 16,), compression="gzip", shuffle=True)
                if c in UNITS:
                    ds.attrs["units"] = UNITS[c]
                if c in DESCRIPTION:
                    ds.attrs["description"] = DESCRIPTION[c]
            ds = g[c]
            ds.resize(start + len(v), axis=0)
            ds[start:] = v.astype(ds.dtype, copy=False)
        self.nrows[group] = start + len(df)


def collect_inputs(paths, filelist):
    files = []
    for p in paths:
        if os.path.isdir(p):
            files += glob.glob(os.path.join(p, PATTERN)) + glob.glob(os.path.join(p, "*", PATTERN))
        else:
            files.append(p)
    if filelist:
        with open(filelist) as fl:
            files += [ln.strip() for ln in fl if ln.strip() and not ln.startswith("#")]
    return sorted(set(files))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", help="output .h5 file")
    ap.add_argument("inputs", nargs="*", help="*.wc.flat.caf.root files or directories")
    ap.add_argument("--filelist", help="text file with one input path per line")
    ap.add_argument("--truth", action="store_true", help="also write /truth_particles and /truth_nu")
    ap.add_argument("-j", "--jobs", type=int, default=1, help="parallel reader processes (default 1)")
    ap.add_argument("-n", "--max-files", type=int, help="process at most N files")
    ap.add_argument("--force", action="store_true", help="overwrite an existing output file")
    args = ap.parse_args()

    files = collect_inputs(args.inputs, args.filelist)[:args.max_files]
    if not files:
        sys.exit("no input files found")
    if os.path.exists(args.out) and not args.force:
        sys.exit(f"{args.out} exists; use --force to overwrite")

    # start workers before opening the HDF5 file so they don't inherit its handle
    work = functools.partial(extract, truth=args.truth)
    pool = multiprocessing.Pool(args.jobs) if args.jobs > 1 else None
    results = pool.imap(work, files) if pool else map(work, files)

    done, failed = [], []
    t0 = time.time()
    with h5py.File(args.out, "w") as h5:
        h5.attrs["source_tree"] = TREE
        h5.attrs["created"] = datetime.now().isoformat(timespec="seconds")
        h5.attrs["description"] = "WC PF-style inputs from SBND wc.flat.caf files; see script docstring"
        tables = H5Tables(h5)

        for i, (path, out, err) in enumerate(results):
            if err:
                failed.append(path)
                print(f"WARNING: skipping {path}: {err}", file=sys.stderr)
                continue

            fidx = len(done)
            done.append(path)
            ev, parts = out["events"], out["particles"]
            ev["file_index"] = fidx
            ev["_row"] = np.arange(len(ev)) + tables.nrows.get("events", 0)
            parts = parts.merge(ev, on=EVT_KEYS + ["nu_index"], how="left")
            parts["event_row"] = parts.pop("_row").fillna(-1).astype(np.int64)
            parts["file_index"] = fidx
            tables.append("events", ev.drop(columns="_row"))
            tables.append("particles", parts)
            if args.truth:
                for group in ("truth_particles", "truth_nu"):
                    out[group]["file_index"] = fidx
                    tables.append(group, out[group])

            if (i + 1) % 100 == 0:
                print(f"[{i + 1}/{len(files)}] {time.time() - t0:.0f}s {tables.nrows}", flush=True)

        h5.create_dataset("files", data=np.array(done, dtype=h5py.string_dtype()))
        h5.attrs["n_files_failed"] = len(failed)

    if pool:
        pool.close()
        pool.join()
    print(f"wrote {args.out}: {len(done)} files ok, {len(failed)} failed in {time.time() - t0:.0f}s")
    for group, n in tables.nrows.items():
        print(f"  /{group}: {n} rows")


if __name__ == "__main__":
    main()
