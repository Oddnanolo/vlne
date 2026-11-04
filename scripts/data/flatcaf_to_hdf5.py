#!/usr/bin/env python3
"""Extract WireCell reco neutrino candidates from SBND *.wc.flat.caf.root files
into a single vlne-ready HDF5 file (the layout of vlne_data_extractors_uboone).

Source (recTreeWireCell, all branches prefixed "wc."), uB-name -> SBND-name:
  T_PFeval reco particles   -> T_pf.*          (positions cm, ke_mev MeV, truncated)
  T_PFeval truth particles  -> T_truth_pf.*    (positions cm, E/KE/p GeV, t ns)
  T_BDTvars                 -> T_tagger.*      (one row per reco nu candidate)
  T_KINEvars                -> T_kine.*        (one row per reco nu candidate)
  T_eval truth              -> T_truth_nu.*, T_match (reco <-> truth matching)
Reco particles have no momentum vector: startMomentum is built from KE (full
precision from T_kine.kine_energy_particle[kine_index]) + PDG mass, with
direction = unit(end - start), or for zero-length pseudo particles (pi0) the
direction of the summed momenta of their children.

Truth matching: a candidate's level-3 reco object in T_match (side 0, same
nu_index) is matched to the truth object with the highest charge IoU
(match_iou_q); that object's truth_nu_idx is the candidate's row in T_truth_nu.

Selection (mirrors the uboone training selection):
  matched truth nu of --flavor/--current, match IoU >= --min-iou,
  true vertex inside the active volume, numu_cc_flag >= 0.

HDF5 layout: one row per selected candidate, one dataset per column in the file
root (vlndata HDF5Frame). Scalars are 1-D; particle.* are variable-length
float32 arrays with one entry per reco particle. Energies GeV, positions cm.
  addr.{run,subrun,event,nu_index,file_index}  file_index -> line of OUT.files.txt
  config.{contain.full,flavor.numu,flavor.nue,current.cc}
  truth.{nuE,lepE,hadE,match_iou}              lepE: primary lepton total energy
  reco.{nuE,lepE,hadE}                         nuE: T_kine.kine_reco_Enu; lepE: leading reco lepton E
  event.{numu_cc_flag,numu_score,nue_score}
  particle.{start,end}.{x,y,z}
  particle.pdg.{electron,gamma,muon,neutron,pion,pizero,proton}
  particle.startMomentum.{t,x,y,z}
  particle.{KE,depth,is_pseudo}
Rows are in input order; shuffle the training file before use (scripts/data/shuffle.py).

Environment: needs uproot, awkward, pandas, h5py (the vlne env, or cafpyana on the gpvm:
  cd /exp/sbnd/app/users/eespinoz/cafpyana && source setup.sh && cd -)

Usage:
  python3 flatcaf_to_hdf5.py out.hdf FILE_OR_DIR [...] [--filelist list.txt]
      [--flavor numu [nue]] [--current cc|nc] [--min-iou 0.5] [-j 8] [-n N] [--force]
Directories are searched one level deep for *.wc.flat.caf.root.
Each file is read whole into memory before parsing (uproot's small reads are
slow on dCache NFS); expect a few seconds per file per worker.

Read back:
  import h5py
  with h5py.File("out.hdf") as f:
      nuE, muon_flags = f["truth.nuE"][:], f["particle.pdg.muon"][0]
"""
import argparse
import collections
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
CAND_KEYS = EVT_KEYS + ["nu_index"]

MASS_GEV = {11: 0.000511, 13: 0.105658, 22: 0.0, 111: 0.134977,
            211: 0.139570, 2112: 0.939565, 2212: 0.938272}
ONEHOT = {"electron": 11, "gamma": 22, "muon": 13, "neutron": 2112,
          "pion": 211, "pizero": 111, "proton": 2212}
FLAVOR_PDG = {"numu": 14, "nue": 12}
# SBND active volume (cm), the analogue of uboone's truth_vtxInside
ACTIVE_VOLUME = {"vtx_x": (-200.0, 200.0), "vtx_y": (-200.0, 200.0), "vtx_z": (0.0, 500.0)}

PARTICLE_COLS = {
    **{f"particle.{p}.{c}": f"{p}_{c}" for p in ("start", "end") for c in "xyz"},
    **{f"particle.pdg.{n}": f"pdg_{n}" for n in ONEHOT},
    **{f"particle.startMomentum.{c}": f"startMomentum_{c}" for c in "txyz"},
    "particle.KE": "KE", "particle.depth": "depth", "particle.is_pseudo": "is_pseudo",
}

Selection = collections.namedtuple("Selection", ["flavors", "current", "min_iou"])


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


def candidates(t):
    """One row per reco nu candidate: T_tagger flags/scores + T_kine reco energy."""
    tag = per_row(t, "wc.T_tagger.", ["nu_index", "match_isFC", "numu_cc_flag", "numu_score", "nue_score"])
    kine = per_row(t, "wc.T_kine.", ["nu_index", "kine_reco_Enu"])
    return tag.merge(kine, on=CAND_KEYS, how="left")


def truth_match(t):
    """Reco nu candidate -> truth_nu_idx and charge IoU of its best level-3 T_match match."""
    p = "wc.T_match."
    a = t.arrays([p + f for f in ("level", "side", "obj", "nu_index", "truth_nu_idx", "nmatch",
                                  "match_obj", "match_obj..idx", "match_iou_q")] + EVT, library="np")
    rows = []
    for i in range(t.num_entries):
        lvl, side, obj, nui = (a[p + f][i] for f in ("level", "side", "obj", "nu_index"))
        truth_side = (lvl == 3) & (side == 1)
        truth_nu_of = dict(zip(obj[truth_side], a[p + "truth_nu_idx"][i][truth_side]))
        # match_* arrays are flat per entry; row r owns [match_obj..idx[r], +nmatch[r])
        for r in np.flatnonzero((lvl == 3) & (side == 0) & (nui >= 0)):
            s, n = a[p + "match_obj..idx"][i][r], a[p + "nmatch"][i][r]
            if n == 0:
                continue
            iou = a[p + "match_iou_q"][i][s:s + n]
            k = int(np.argmax(iou))
            best = a[p + "match_obj"][i][s + k]
            rows.append((*(a[b][i] for b in EVT), nui[r], truth_nu_of.get(best, -1), iou[k]))
    df = pd.DataFrame(rows, columns=CAND_KEYS + ["truth_nu_idx", "match_iou"])
    return df.sort_values("match_iou").drop_duplicates(CAND_KEYS, keep="last")


def truth_nu(t):
    df = per_row(t, "wc.T_truth_nu.", ["nu_idx", "pdg", "ccnc", "E", "vtx_x", "vtx_y", "vtx_z"])
    return df.rename(columns={"nu_idx": "truth_nu_idx", "pdg": "nu_pdg", "E": "nuE"})


def truth_leptons(t):
    """Total energy of the leading primary charged lepton of each truth nu, per lepton |pdg|."""
    df = per_row(t, "wc.T_truth_pf.", ["nu_row", "parent_trackid", "pdg", "E"])
    df = df[(df["parent_trackid"] == 0) & df["pdg"].abs().isin([11, 13])]
    df = df.assign(lep_pdg=df["pdg"].abs()).rename(columns={"nu_row": "truth_nu_idx", "E": "lepE"})
    return df.groupby(EVT_KEYS + ["truth_nu_idx", "lep_pdg"], as_index=False)["lepE"].max()


def in_active_volume(df):
    keep = np.ones(len(df), dtype=bool)
    for col, (lo, hi) in ACTIVE_VOLUME.items():
        keep &= (df[col] > lo).to_numpy() & (df[col] < hi).to_numpy()
    return keep


def select_candidates(t, sel):
    """Matched, selected candidates with truth and reco event-level quantities."""
    cand = candidates(t).merge(truth_match(t), on=CAND_KEYS, how="left")
    cand = cand[cand["truth_nu_idx"].fillna(-1) >= 0].astype({"truth_nu_idx": np.int64})
    cand = cand.merge(truth_nu(t), on=EVT_KEYS + ["truth_nu_idx"], how="inner")

    keep = (cand["match_iou"] >= sel.min_iou).to_numpy()
    keep &= cand["nu_pdg"].abs().isin([FLAVOR_PDG[f] for f in sel.flavors]).to_numpy()
    keep &= ((cand["ccnc"] == 0) == (sel.current == "cc")).to_numpy()
    keep &= in_active_volume(cand)
    keep &= (cand["numu_cc_flag"] >= 0).to_numpy()
    cand = cand[keep]

    # lepton of the neutrino's own flavor: muon for numu, electron for nue
    cand = cand.assign(lep_pdg=cand["nu_pdg"].abs() - 1)
    cand = cand.merge(truth_leptons(t), on=EVT_KEYS + ["truth_nu_idx", "lep_pdg"], how="left")
    if sel.current == "cc":
        # ~0.4% of CC nus have no primary lepton in T_truth_pf (only its decay products,
        # re-parented to the nu); drop them rather than train on lepE = 0
        cand = cand[cand["lepE"].notna()]
    cand["lepE"] = cand["lepE"].fillna(0.0)
    return cand.reset_index(drop=True)


def pack_particles(parts, n_rows):
    """One row per particle (with local "row") -> {name: object array of per-row float32 arrays}."""
    if n_rows == 0:
        return {name: np.empty(0, dtype=object) for name in PARTICLE_COLS}
    parts = parts.sort_values("row", kind="stable")
    bounds = np.cumsum(np.bincount(parts["row"], minlength=n_rows))[:-1]
    out = {}
    for name, col in PARTICLE_COLS.items():
        # fill element-wise: np.array(list_of_arrays) turns equal-length lists into a 2-D array
        arr = np.empty(n_rows, dtype=object)
        for i, x in enumerate(np.split(parts[col].to_numpy(np.float32), bounds)):
            arr[i] = x
        out[name] = arr
    return out


def to_columns(cand, parts):
    """Selected candidates + their reco particles -> {vlne column: array}."""
    parts = parts.merge(cand[CAND_KEYS].assign(row=np.arange(len(cand))), on=CAND_KEYS, how="inner")
    lep = parts[parts["pdg"].abs() == cand["lep_pdg"].to_numpy()[parts["row"]]]
    reco_lepE = lep.groupby("row")["startMomentum_t"].max().reindex(range(len(cand)), fill_value=0.0)
    reco_nuE = cand["kine_reco_Enu"].to_numpy() / 1000.0
    f32 = lambda v: np.asarray(v, dtype=np.float32)

    cols = {
        "addr.run": cand["run"].to_numpy(),
        "addr.subrun": cand["subrun"].to_numpy(),
        "addr.event": cand["evt"].to_numpy(),
        "addr.nu_index": cand["nu_index"].to_numpy(),
        "config.contain.full": f32(cand["match_isFC"] == 1),
        "config.flavor.numu": f32(cand["nu_pdg"].abs() == 14),
        "config.flavor.nue": f32(cand["nu_pdg"].abs() == 12),
        "config.current.cc": f32(cand["ccnc"] == 0),
        "truth.nuE": f32(cand["nuE"]),
        "truth.lepE": f32(cand["lepE"]),
        "truth.hadE": f32(cand["nuE"] - cand["lepE"]),
        "truth.match_iou": f32(cand["match_iou"]),
        "reco.nuE": f32(reco_nuE),
        "reco.lepE": f32(reco_lepE),
        "reco.hadE": f32(reco_nuE - reco_lepE.to_numpy()),
        "event.numu_cc_flag": f32(cand["numu_cc_flag"]),
        "event.numu_score": f32(cand["numu_score"]),
        "event.nue_score": f32(cand["nue_score"]),
    }
    cols.update(pack_particles(parts, len(cand)))
    return cols


def open_root(path):
    if "://" in path:
        return uproot.open(path)
    # one sequential read is ~5x faster on /pnfs than uproot's many small reads
    with open(path, "rb") as fh:
        return uproot.open(io.BytesIO(fh.read()))


def extract(path, sel):
    """Read one file -> (path, {column: array}, {count: n}, error message or None)."""
    try:
        with open_root(path) as f:
            t = f[TREE]
            n_cand = sum(len(x) for x in t["wc.T_tagger.nu_index"].array(library="np"))
            cand = select_candidates(t, sel)
            cols = to_columns(cand, reco_particles(t))
        return path, cols, {"candidates": n_cand, "selected": len(cand)}, None
    except Exception as e:  # keep going on bad/missing files
        return path, None, None, f"{type(e).__name__}: {e}"


class VlneColumns:
    """Append rows to flat root-level datasets (the layout vlndata's HDF5 frames read)."""

    def __init__(self, h5):
        self.h5 = h5
        self.nrows = 0

    def append(self, cols):
        n = len(next(iter(cols.values())))
        if n == 0:
            return
        for name, v in cols.items():
            if name not in self.h5:
                if v.dtype == object:
                    self.h5.create_dataset(name, shape=(0,), maxshape=(None,), chunks=True,
                                           dtype=h5py.vlen_dtype(np.float32))
                else:
                    self.h5.create_dataset(name, shape=(0,), maxshape=(None,), chunks=True,
                                           dtype=v.dtype, compression="gzip")
            ds = self.h5[name]
            ds.resize(self.nrows + n, axis=0)
            if v.dtype == object and len({len(x) for x in v}) == 1:
                # h5py turns equal-length rows into a 2-D array; write them one at a time
                for j, x in enumerate(v):
                    ds[self.nrows + j] = x
            else:
                ds[self.nrows:] = v
        self.nrows += n


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
    ap.add_argument("out", help="output .hdf file (file list goes to OUT.files.txt)")
    ap.add_argument("inputs", nargs="*", help="*.wc.flat.caf.root files or directories")
    ap.add_argument("--filelist", help="text file with one input path per line")
    ap.add_argument("--flavor", nargs="+", choices=list(FLAVOR_PDG), default=["numu"],
                    help="true neutrino flavor(s) to keep (default numu)")
    ap.add_argument("--current", choices=["cc", "nc"], default="cc", help="true current (default cc)")
    ap.add_argument("--min-iou", type=float, default=0.5,
                    help="minimum charge IoU of the reco-truth match (default 0.5)")
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
    sel = Selection(flavors=args.flavor, current=args.current, min_iou=args.min_iou)
    work = functools.partial(extract, sel=sel)
    pool = multiprocessing.Pool(args.jobs) if args.jobs > 1 else None
    results = pool.imap(work, files) if pool else map(work, files)

    done, failed = [], []
    counts = collections.Counter()
    t0 = time.time()
    with h5py.File(args.out, "w") as h5:
        h5.attrs["source_tree"] = TREE
        h5.attrs["created"] = datetime.now().isoformat(timespec="seconds")
        h5.attrs["description"] = "vlne inputs from SBND wc.flat.caf files; see script docstring"
        table = VlneColumns(h5)

        for i, (path, cols, n, err) in enumerate(results):
            if err:
                failed.append(path)
                print(f"WARNING: skipping {path}: {err}", file=sys.stderr)
                continue

            cols["addr.file_index"] = np.full(n["selected"], len(done), dtype=np.int64)
            done.append(path)
            counts.update(n)
            table.append(cols)

            if (i + 1) % 100 == 0:
                print(f"[{i + 1}/{len(files)}] {time.time() - t0:.0f}s {table.nrows} rows", flush=True)

    with open(args.out + ".files.txt", "w") as fl:
        fl.write("\n".join(done) + "\n")

    if pool:
        pool.close()
        pool.join()
    print(f"wrote {args.out}: {len(done)} files ok, {len(failed)} failed in {time.time() - t0:.0f}s")
    print(f"  {counts['selected']} of {counts['candidates']} reco nu candidates selected "
          f"({', '.join(args.flavor)} {args.current.upper()}, IoU >= {args.min_iou}, vertex in AV)")


if __name__ == "__main__":
    main()
