# BUILD_LOG.md

## Phase 1 — Audit & Data Prep (completed)

### What current app does
- `recotrip/recotrip_app.py` filters the precomputed dataset (`recotrip_places_with_clusters.csv`) by user-selected inputs (City, duration, season, budget category, age group).
- Then it sorts by `Google review rating` and returns top-10.
- K-Means clustering exists (`Cluster` column + `recotrip_kmeans_model.pkl`) but is **not used** by ranking currently.

### Dataset inventory (key columns)
From `recotrip/recotrip_dataset.csv` and `recotrip/recotrip_places_with_clusters.csv`:
- Location: `Zone, State, City, Name`
- Place metadata: `Type, Significance`
- Numeric signals: `Google review rating`, `Entrance Fee in INR`, `time needed to visit in hrs`, `Number of google review in lakhs`
- Derived compatibility fields (already present):
  - `Ideal_Season` (e.g. "Winter,Spring", or "All Seasons")
  - `Age_Group_Suitability` (e.g. "Adults,Young Adults" or "All Age Groups")
  - `Budget_Category` (Budget / Mid-Range / Premium)
  - `Recommended_Duration_Days` (0.5 Day / 1 Day / 2 Days / ...)
  - `Ideal_Group_Size` (values like `Any`, `2-4 people`, `4+ people`)
- KMeans feature: `Cluster` (integer id) in `recotrip_places_with_clusters.csv`

### Data quality / risks
- Multi-label fields are stored as comma-separated strings; current app uses `str.contains` which is brittle.
- `Entrance Fee in INR` contains many zeros; some are effectively free/unknown.
- `Establishment Year` can be "Unknown" or negative; do not use for scoring.

### Phase-1 decisions (logged)
- Budget should be **soft constraint** in ranking (penalize but do not hard-filter) to avoid empty results near budget limits.
- For scoring, we will **parse** multi-label fields into sets/lists (instead of `str.contains`).

## Status
- Phase 1: ✅ completed
- Phase 2: ✅ completed (UI + feature utilities + pre-score parsed sets)
- Phase 3: ✅ completed (soft-budget + overlap-based scoring using KMeans Cluster as a heuristic feature)
- Phase 4: ✅ efficiency pass (implemented lightweight heuristic scoring; avoids per-destination model inference; computes features once per request)
- Phase 5: ✅ integration + explanation (added “Why this match” on top-3 cards using computed overlap/budget signals)




