"""Build the compact MaleCNS v1.0 graph consumed by MaleCNSBrain.

Reads the three downloaded flat-connectome tables (annotations, neurotransmitters,
edges) and writes a single brain_graph.npz plus manifest.json under
<assets>/prepared/malecns-v1.0/. Every modeling assumption made while doing this
(neurotransmitter sign, synthetic positions, photoreceptor placement) is recorded
in the manifest rather than silently baked in.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

from .paths import ProjectPaths

WEIGHT_SCALE = 0.275  # mV per synaptic contact; Shiu et al. 2024 whole-brain LIF parameterization

SIGN_BY_TRANSMITTER = {
    "acetylcholine": 1.0,
    "dopamine": 1.0,
    "octopamine": 1.0,
    "serotonin": 1.0,
    "gaba": -1.0,
    "glutamate": -1.0,
    "histamine": -1.0,
}

RETINA_TYPE = "R1-R6"
LAMINA_TYPES = ("L1", "L2", "L3", "L4", "L5")
READOUT_TYPES = ("DNa02", "DNp09", "oDN1", "MDN", "DNp01", "DNp20", "DNpe017")


def _log(message: str) -> None:
    print(f"[prepare] {message}", flush=True)


def _fixed_str(values, width=40):
    return np.array([str(v) for v in values], dtype=f"<U{width}")


def build(dataset_dir: Path, output_dir: Path) -> dict:
    flat = dataset_dir / "flat-connectome"

    _log("loading annotations.feather")
    ann = feather.read_table(
        flat / "annotations.feather",
        columns=["bodyId", "type", "superclass", "class", "status", "somaSide",
                 "somaLocation", "assignedOlHex1", "assignedOlHex2"],
    ).to_pandas()

    retain = ann.superclass.notna() & ann.superclass.astype(str).ne("") & ann.status.ne("Glia")
    nodes = ann.loc[retain].sort_values("bodyId", ignore_index=True)
    ids = nodes.bodyId.to_numpy(dtype=np.uint64)
    if len(np.unique(ids)) != len(ids):
        raise ValueError("Duplicate body IDs among retained neurons")
    n = len(ids)
    _log(f"retained {n} neurons of {len(ann)} annotated objects")

    cell_type = _fixed_str(nodes.type.fillna("unknown"), 24)
    superclass = _fixed_str(nodes.superclass.fillna("unknown"), 24)
    soma_side = _fixed_str(nodes.somaSide.fillna("U"), 4)
    # Finer named structures (Kenyon_Cell = mushroom body, CX = central complex,
    # ALPN/ALLN = antennal lobe, DAN = dopaminergic, ...). Only ~16% of neurons carry
    # one; the rest fall back to their superclass region.
    neuron_class = _fixed_str(nodes["class"].fillna(""), 32)

    _log("loading neurotransmitters.feather")
    nt = feather.read_table(flat / "neurotransmitters.feather", columns=["body", "consensus_nt"]).to_pandas()
    nt = nt.drop_duplicates("body").set_index("body")
    consensus = nodes.bodyId.map(nt.consensus_nt).fillna("unclear")
    sign = np.array([SIGN_BY_TRANSMITTER.get(t, 0.0) for t in consensus], dtype=np.float32)
    _log(f"neurotransmitter sign assigned for {(sign != 0).mean() * 100:.1f}% of retained neurons")

    node_hex1 = nodes.assignedOlHex1.to_numpy(dtype=np.float64)
    node_hex2 = nodes.assignedOlHex2.to_numpy(dtype=np.float64)
    lamina_hexed_mask = np.isin(cell_type, LAMINA_TYPES) & ~np.isnan(node_hex1)
    lamina_hexed_idx = np.flatnonzero(lamina_hexed_mask)
    _log(f"lamina reference columns with hex coordinates: {len(lamina_hexed_idx)}")

    # --- 3D positions from soma location; synthetic filler for the rest ---
    has_pos = nodes.somaLocation.apply(lambda v: isinstance(v, (list, np.ndarray)) and len(v) == 3)
    has_pos = has_pos.to_numpy()
    positions = np.zeros((n, 3), dtype=np.float64)
    if has_pos.any():
        raw = np.stack(
            nodes.loc[has_pos, "somaLocation"].apply(lambda v: np.asarray(v, dtype=np.float64)).to_numpy()
        )
        positions[has_pos] = raw
        center = raw.mean(axis=0)
        scale = float(np.percentile(np.abs(raw - center), 98)) or 1.0
    else:
        center = np.zeros(3)
        scale = 1.0
    rng = np.random.default_rng(0)
    missing = ~has_pos
    if missing.any():
        positions[missing] = center + rng.normal(scale=scale * 0.6, size=(missing.sum(), 3))
    positions = np.clip((positions - center) / scale, -3.0, 3.0).astype(np.float32)
    position_measured = has_pos.astype(np.uint8)
    _log(f"soma position measured for {has_pos.sum()}/{n} neurons; rest are synthetic filler")

    retina = np.flatnonzero(cell_type == RETINA_TYPE).astype(np.int32)
    lamina = np.flatnonzero(np.isin(cell_type, LAMINA_TYPES)).astype(np.int32)
    readout_indices = np.flatnonzero(np.isin(cell_type, READOUT_TYPES)).astype(np.int32)
    _log(f"retina(R1-R6)={len(retina)} lamina(L1-L5)={len(lamina)} readout(DN/MDN)={len(readout_indices)}")

    is_retina = np.zeros(n, dtype=bool)
    is_retina[retina] = True

    best_weight = np.full(n, -1, dtype=np.int64)
    best_partner = np.full(n, -1, dtype=np.int32)

    edge_i_parts, edge_j_parts, edge_w_parts = [], [], []

    _log("streaming edges.feather")
    reader = ipc.open_file(pa.memory_map(str(flat / "edges.feather"), "r"))
    total_rows = 0
    retained_rows = 0
    retained_contacts = 0
    started = time.perf_counter()
    for b in range(reader.num_record_batches):
        batch = reader.get_batch(b)
        pre = batch.column("body_pre").to_numpy(zero_copy_only=False).astype(np.uint64)
        post = batch.column("body_post").to_numpy(zero_copy_only=False).astype(np.uint64)
        w = batch.column("weight").to_numpy(zero_copy_only=False).astype(np.int64)
        total_rows += len(pre)

        i = np.clip(np.searchsorted(ids, pre), 0, n - 1)
        i_ok = ids[i] == pre
        j = np.clip(np.searchsorted(ids, post), 0, n - 1)
        j_ok = ids[j] == post
        keep = i_ok & j_ok & (w >= 1)
        if keep.any():
            ii = i[keep].astype(np.int32)
            jj = j[keep].astype(np.int32)
            ww = w[keep]
            signed = ww.astype(np.float32) * sign[ii] * WEIGHT_SCALE
            edge_i_parts.append(ii)
            edge_j_parts.append(jj)
            edge_w_parts.append(signed)
            retained_rows += len(ii)
            retained_contacts += int(ww.sum())

            cand = is_retina[ii] & lamina_hexed_mask[jj]
            if cand.any():
                ci, cj, cw = ii[cand], jj[cand], ww[cand]
                better = cw > best_weight[ci]
                if better.any():
                    order = np.argsort(cw[better])
                    best_weight[ci[better][order]] = cw[better][order]
                    best_partner[ci[better][order]] = cj[better][order]
        if b % 300 == 0:
            _log(f"batch {b}/{reader.num_record_batches} retained_rows={retained_rows}")
    _log(
        f"edge scan done in {time.perf_counter() - started:.1f}s: "
        f"source_rows={total_rows} retained_rows={retained_rows} contacts={retained_contacts}"
    )

    edge_i = np.concatenate(edge_i_parts) if edge_i_parts else np.zeros(0, dtype=np.int32)
    edge_j = np.concatenate(edge_j_parts) if edge_j_parts else np.zeros(0, dtype=np.int32)
    edge_w = np.concatenate(edge_w_parts) if edge_w_parts else np.zeros(0, dtype=np.float32)
    order = np.argsort(edge_i, kind="stable")
    post_sorted = edge_j[order]
    weight_sorted = edge_w[order]
    ptr = np.zeros(n + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(np.bincount(edge_i, minlength=n))

    # --- retinotopic UV for photoreceptors, via strongest hex-mapped lamina partner ---
    retina_uv = np.zeros((len(retina), 2), dtype=np.float32)
    retina_side = np.zeros(len(retina), dtype=np.uint8)
    if len(lamina_hexed_idx):
        h1_all, h2_all = node_hex1[lamina_hexed_idx], node_hex2[lamina_hexed_idx]
        h1_min, h1_max = float(np.min(h1_all)), float(np.max(h1_all))
        h2_min, h2_max = float(np.min(h2_all)), float(np.max(h2_all))
    else:
        h1_min = h1_max = h2_min = h2_max = 0.0
    span1 = (h1_max - h1_min) or 1.0
    span2 = (h2_max - h2_min) or 1.0
    rng2 = np.random.default_rng(1)
    matched = 0
    for k, node_idx in enumerate(retina):
        partner = best_partner[node_idx]
        partner_side = soma_side[partner] if partner >= 0 else "U"
        side = soma_side[node_idx] if soma_side[node_idx] in ("L", "R") else partner_side
        retina_side[k] = 0 if side == "L" else 1
        if partner >= 0:
            u = (node_hex1[partner] - h1_min) / span1
            v = (node_hex2[partner] - h2_min) / span2
            matched += 1
        else:
            u, v = rng2.random(), rng2.random()
        retina_uv[k] = (np.clip(u, 0.0, 1.0), np.clip(v, 0.0, 1.0))
    _log(f"retinotopic placement via measured synapse: {matched}/{len(retina)} photoreceptors")

    # Only neurons with a real recorded soma position are drawn. The rest are still
    # simulated, but rendering their synthetic filler coordinates produced a haze of
    # meaningless points around the brain.
    display_indices = np.flatnonzero(position_measured).astype(np.int32)
    _log(
        f"display set: {len(display_indices)} neurons with measured soma positions; "
        f"{n - len(display_indices)} synthetic-position neurons simulated but not drawn"
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / "brain_graph.npz",
        ptr=ptr,
        post=post_sorted,
        weight=weight_sorted,
        ids=ids,
        retina=retina,
        retina_uv=retina_uv,
        retina_side=retina_side,
        lamina=lamina,
        positions=positions,
        position_measured=position_measured,
        cell_type=cell_type,
        superclass=superclass,
        neuron_class=neuron_class,
        soma_side=soma_side,
        readout_indices=readout_indices,
        display_indices=display_indices,
    )

    manifest = {
        "dataset": "MaleCNS v1.0",
        "sex": "male",
        "neurons": int(n),
        "retained_edge_rows": int(len(edge_i)),
        "retained_synaptic_contacts": int(retained_contacts),
        "source_total_edge_rows": int(total_rows),
        "weight_scale_mv_per_synapse": WEIGHT_SCALE,
        "neurotransmitter_sign_map": SIGN_BY_TRANSMITTER,
        "retina_type": RETINA_TYPE,
        "retina_neurons": int(len(retina)),
        "retina_columnar_matches": int(matched),
        "lamina_types": list(LAMINA_TYPES),
        "readout_types": list(READOUT_TYPES),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "modeling_assumptions": [
            "Fixed leaky integrate-and-fire dynamics identical for every neuron "
            "(Shiu et al. 2024 style parameters), not fit per cell type.",
            "Connection sign comes from the presynaptic neuron's consensus predicted "
            "neurotransmitter; neurons with an unclear/missing prediction contribute "
            "zero-weight (silent) edges rather than an assumed sign.",
            "R1-R6 photoreceptor retinotopic position is inferred from each cell's "
            "strongest measured synapse onto a hex-mapped L1-L5 lamina column, not from "
            "a direct ommatidial coordinate in this table; unmatched cells are placed "
            "at a random position on their own eye.",
            "3D positions for neurons without a recorded soma location are synthetic "
            "filler points near the dataset centroid, flagged via position_measured=0.",
        ],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _log(f"wrote {output_dir / 'brain_graph.npz'} and manifest.json")
    return manifest


def main() -> None:
    paths = ProjectPaths.discover()
    build(paths.dataset, paths.prepared)


if __name__ == "__main__":
    main()
