"""Export the corrected SNN-2 (verified labels from all ten detection runs, GRID_ID join,
land-cover class code removed) to the JSON files the static digital twin reads.

The model is the full-data SNN-2 from Corrected_Evaluation.ipynb. Running the notebook's own
code cells guarantees identical features, screening and training; the refit is checked
against the notebook's cached full-model predictions before anything is written.

Run from anywhere:  python3 scripts/export_static_v2.py
"""
import os, sys, json, glob
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
APP = os.path.dirname(HERE)
BUNDLE = os.path.abspath(os.path.join(APP, "..", "Table2_notebook_bundle"))
OUT = os.path.join(APP, "data")

# ---- run the notebook's setup, data, feature and model cells inside the bundle folder
os.chdir(BUNDLE)
import matplotlib; matplotlib.use("Agg")
nb = json.load(open("Corrected_Evaluation.ipynb"))
cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
G = {"display": lambda *a, **k: None, "__name__": "__main__"}
for s in cells[:4]:
    exec(s, G)
FEAT, GEO, fit_snn, snn_predict, RES_SQMI = G["FEAT"], G["GEO"], G["fit_snn"], G["snn_predict"], G["RES_SQMI"]

# ---- labels and groups (reuse the app's meta, add the Esri codes the corrected model uses)
meta = json.load(open(os.path.join(OUT, "meta.json")))
CODE = {
    "CRMCYTOTC": ("Total Crime Index", "Socioeconomic"),
    "TOTPOP_CY": ("Total Population", "Socioeconomic"),
    "UNEMPRT_CY_I": ("Unemployment Rate (index)", "Socioeconomic"),
    "ACSHHDIS_P": ("Households with Disability (%)", "Socioeconomic"),
    "ACSPUBAI_P": ("Public Assistance Income (%)", "Socioeconomic"),
    "ACSBLWPV_P": ("Below Poverty Level (%)", "Socioeconomic"),
    "ACSGRNTI50_P": ("Severe Rent Burden (%)", "Socioeconomic"),
    "ACSEDNOSCH_P": ("No Schooling Completed (%)", "Socioeconomic"),
    "ACS35NOHI_P": ("Pop. 35-64 with No Health Insurance (%)", "Socioeconomic"),
    "ACSVET_P": ("Veteran Population, 2022 (%)", "Socioeconomic"),
    "ACSVET_P_1": ("Veteran Population, 2023 (%)", "Socioeconomic"),
    "ACSHSGRAD_P": ("High School Graduate (%)", "Socioeconomic"),
    "X14068FY_X_I": ("Credit Card Debt (market index, forecast)", "Socioeconomic"),
    "MEDHINC_CY_I": ("Median Household Income (index)", "Socioeconomic"),
    "caltrans_highway_segments_count": ("State Highway Segments", "Roads / Waterways"),
    "nearest_caltrans_highway_m": ("Distance to Nearest State Highway (m)", "Roads / Waterways"),
}
for k, (lab, grp) in CODE.items():
    meta["labels"][k] = lab; meta["groups"][k] = grp

def net1(n):
    return {"centers": n.c.tolist(), "widths": np.exp(n.lw).tolist(), "coef": [float(n.b[0])] + n.a.tolist()}

def net2(n):
    return {"cj": n.cj.tolist(), "ck": n.ck.tolist(), "wj": np.exp(n.lwj).tolist(), "wk": np.exp(n.lwk).tolist(),
            "coef": [float(n.b[0])] + n.a.tolist()}

det = pd.read_csv(os.path.join(BUNDLE, "..", "..", "documented_content", "Output", "tent_detections_full_18060_combined.csv"))
report = []
for res in ["quarter", "half", "three_fourth"]:
    F = FEAT[res]; Xdf = F["X"]; X = Xdf.values.astype(float); t = F["t"]; names = list(Xdf.columns)
    m = fit_snn(X, F["y_ver"], t.block.values)
    s1, s2, c1, c2 = snn_predict(m, X)
    cache = np.load(glob.glob(os.path.join("cache_fulllabels_nolc", f"{res}_full_SNN_*.npz"))[0], allow_pickle=True)
    gap = float(np.abs(s2 - cache["s2"]).max())
    # multithreaded training (BLAS summation order over thousands of Adam steps) gives ~1e-4 to 1e-3 noise;
    # require the same model up to that noise: tiny score gap and near-identical ranking
    from scipy.stats import spearmanr
    rho = spearmanr(s2, cache["s2"]).correlation
    assert gap < 1e-2 and rho > 0.999, f"{res}: refit differs from the notebook's full model (max gap {gap}, rho {rho})"
    pairs = [(int(j), int(k)) for j, k in m["pairs"]]
    assert sorted(map(tuple, cache["pairs"].tolist())) == sorted(pairs), f"{res}: interaction pairs differ from the notebook"
    off1, off2 = c1.mean(0), c2.mean(0)
    risk = np.clip(s2, 0, 1)
    cc2 = c2 - off2
    top = np.abs(cc2).argmax(1)
    polys = GEO[res]["polys"]
    geom = [[[[round(x, 5), round(y, 5)] for x, y in polys[g].exterior.coords]] for g in t.GRID_ID]
    summary = {f: [float(np.percentile(X[:, j], 5)), float(np.percentile(X[:, j], 95)), float(X[:, j].min()), float(X[:, j].max())]
               for j, f in enumerate(names)}
    data = {
        "res": res, "features": names,
        "model": {
            "base_features": names, "interaction_pairs": [[names[j], names[k]] for j, k in pairs],
            "scaler_base_mean": m["sc"].mean_.tolist(), "scaler_base_scale": m["sc"].scale_.tolist(),
            "main_nets": [net1(n) for n in m["nets1"]], "pair_nets": [net2(m["nets2"][p]) for p in pairs],
            "link": "identity",                       # SNN is fitted to teacher probabilities: score = clip(sum of terms, 0, 1)
            "main_offsets": off1.tolist(), "pair_offsets": off2.tolist(),   # contributions shown relative to the average cell
            "threshold": None, "n_rbf": 10, "n_rbf2": 6,
            "provenance": "Corrected_Evaluation.ipynb, full-data SNN-2; verified labels from all ten detection runs (verified_all_*.csv); GRID_ID join; dominant_landcover_class excluded",
        },
        "summary": summary,
        "hexes": {
            "gid": t.GRID_ID.tolist(), "oid": list(range(1, len(t) + 1)), "risk": risk.tolist(),
            "tent": t.verified_pts.astype(int).tolist(), "detected": (t.verified_pts > 0).astype(int).tolist(),
            "n311": t.n311.astype(int).tolist(),
            "lat": t.lat.round(6).tolist(), "lon": t.lon.round(6).tolist(),     # hexagon centroid, never detection points
            "ia": [pairs[i][0] for i in top], "ib": [pairs[i][1] for i in top], "iv": cc2[np.arange(len(t)), top].tolist(),
            "X": np.round(X, 6).tolist(),
        },
        "geom": geom,
    }
    json.dump(data, open(os.path.join(OUT, f"{res}.json"), "w"), separators=(",", ":"))
    np.save(os.path.join(HERE, f"_check_{res}.npy"), np.c_[s2, c1 - off1, cc2])
    report.append(f"{RES_SQMI[res]:.2f} sq mi: {len(t)} hexes, {len(names)} features, {len(pairs)} pairs, refit gap {gap:.1e}, rank corr {rho:.5f}, "
                  f"verified-positive {int((t.verified_pts > 0).sum())}")

json.dump(meta, open(os.path.join(OUT, "meta.json"), "w"), indent=1)
print("\n".join(report))
