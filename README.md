# LA Encampment Vulnerability Digital Twin (static / Netlify version)

Same digital twin as the Streamlit app, rebuilt to run entirely in the browser:
SNN-2 scoring on every slider move, the greedy optimizer (all three modes), and
population scenarios. No server, no cold start.

## Run locally
    python3 -m http.server 8000      # then open http://localhost:8000

## Deploy (Netlify, from Git)
1. Push this folder to a GitHub repo.
2. Netlify > Add new project > Import an existing project > GitHub > pick the repo.
3. Leave the build command empty; publish directory is "." (set in netlify.toml). Deploy.

## What is here
- `index.html`, `css/style.css`, `js/app.js` : the interface
- `js/snn.js` : JavaScript port of the Python SNN-2 inference, optimizer and scenarios
- `js/charts.js` : small SVG charts
- `data/*.json` : model weights, per-hex features, slider bounds and hex polygons (exported by `scripts/export_static.py` in the main project)
- `vendor/leaflet` : Leaflet 1.9.4 (BSD-2), bundled so the page does not depend on a CDN

## Model update (Nov 2026): corrected SNN-2 from the DSS paper
`data/*.json` now hold the full-data SNN-2 from `Corrected_Evaluation.ipynb`:
verified labels from all ten Street View + Roboflow runs, labels and features joined on
`GRID_ID`, collinearity screening as in the paper (43 / 43 / 42 predictors, no duplicated
variables), and the land-cover class code excluded. Regenerate with
`python3 scripts/export_static_v2.py`; it refits the model with the notebook's own code and
stops unless the refit matches the notebook's full model (same 20 pairs, rank correlation
> 0.999). The interface is unchanged. Model-side changes in `js/snn.js`, all backward
compatible with the original export:
- one width per basis function (`widths`, `wj`, `wk`) instead of one per network;
- `link: "identity"`: the SNN is fitted to teacher probabilities, so the score is the clipped
  sum of terms (the original export used a sigmoid);
- `main_offsets` / `pair_offsets`: "Why this score" and the interaction callouts show each
  term relative to the average hexagon, as in the paper (the score itself is unchanged).

Text changes in `js/app.js`: the line under the hex title now shows verified encampment
locations and 311 requests (hexagon centroids are used, never detection points), and the
"Feasible Only" caption says results are associations, not causal effects.
The browser model matches the Python model on every hexagon to about 1e-15.
The previous data files are kept outside the site in `../digital_twin_original_data/`.

## Fidelity (original export)
`js/snn.js` was checked against the Python implementation on all three resolutions:
risk scores, per-feature and per-interaction contributions, optimizer steps in all
three modes, and population scenarios agree to about 1e-16. Basemap tiles come from
Esri's public Canvas service (same as the Streamlit app) and need an internet connection.
