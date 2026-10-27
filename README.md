# Spatial Recommendation of Urban Activity Areas for Individuals via Deep Reinforcement Learning

This repository contains a reproducible Python prototype for personalized urban activity-area recommendation. It supports the original synthetic-data workflow and an initial New York City real-data integration layer that reads public datasets from local files.

The project does not require private APIs or API keys.

## Setup

```bash
cd urban_activity_recommendation
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use Python 3.10 or newer. Inside the virtual environment, `python -m src.main` should run the demo.

## Run synthetic mode

```bash
python -m src.main --data-mode synthetic
```

Regenerate sample data before running:

```bash
python -m src.main --data-mode synthetic --generate-sample
```

## Validate NYC raw data

```bash
python -m src.main --validate-data nyc
```

This checks supported files under `data/raw/nyc/`, reports missing optional files, reports invalid columns or coordinates, and saves:

```text
outputs/nyc_data_validation_report.csv
outputs/nyc_data_validation_report.json
```

Print template locations:

```bash
python -m src.main --print-nyc-template-info
```

Templates are in `data/raw/nyc/templates/`. Copy a template into `data/raw/nyc/`, remove `_template` from the filename, and replace the fictional rows with real public data. See `docs/nyc_data_dictionary.md` for column definitions.

## Run NYC mode

```bash
python -m src.main --data-mode nyc
```

NYC mode reads local public-data files from `data/raw/nyc/`, writes standardized files to `data/processed/nyc/`, then runs the same recommendation workflow. If required NYC data is missing, it prints a clear message and falls back to synthetic data.

Fail instead of falling back:

```bash
python -m src.main --data-mode nyc --no-fallback
```

Run the fictional minimal NYC-like example without fallback:

```bash
python -m src.main --data-mode nyc --nyc-raw-dir data/raw/nyc/example_minimal --no-fallback
```

Useful options:

```bash
python -m src.main --data-mode nyc --max-areas 50 --max-resources 500 --output-dir outputs/nyc_run
python -m src.main --data-mode nyc --nyc-raw-dir data/raw/nyc/example_minimal --output-dir outputs/example_minimal
```

## Supported NYC files

Place manually downloaded files here:

```text
data/raw/nyc/
  candidate_areas.csv
  acs_tracts.csv
  facilities.csv
  schools.csv
  hospitals.csv
  parks.csv
  libraries.csv
  gtfs.zip
```

The first implementation supports local CSV/JSON/GeoJSON-style facility files, lightweight `transit_stops.csv`, and local GTFS zip files. Candidate-area files should include centroid longitude/latitude and may include ACS-style fields such as population, median income, and median rent. See `docs/nyc_data_pipeline.md` and `docs/nyc_data_dictionary.md` for accepted column names.

The validation report is a checklist, not a blocker for every file: missing optional files are reported clearly; invalid optional files are skipped; NYC mode fails only if no valid resource/POI file is available and `--no-fallback` is used.

## How real data maps to the research model

- Candidate areas become census tracts, neighborhoods, or fallback grid cells.
- NYC Open Data facilities become resources/POIs in categories such as school, hospital, park, transit, leisure, and public service.
- ACS fields become area-level population, income, and housing-cost variables.
- GTFS stops become transit resources and improve area transit scores.
- OpenStreetMap is optional through OSMnx and is skipped gracefully if unavailable.
- Approximate distance-based travel times feed the same accessibility and reward equations used by the synthetic prototype.



## Real NYC public-data workflow

The real NYC workflow is optional and does not require API keys. Dataset metadata and manual download instructions are listed in `config/nyc_datasets.yaml`.

Attempt public direct downloads where a direct URL is listed:

```bash
python -m src.main --download-nyc-data --output-dir outputs/download_test
python -m src.main --download-nyc-data --dataset-id mta_subway_gtfs --skip-existing
```

Datasets without direct downloadable URLs are skipped and recorded in `outputs/nyc_download_report.json` with manual download instructions. Place real public files in:

```text
data/raw/nyc/real/
  census_areas.csv
  facilities.csv
  schools.csv
  hospitals.csv
  parks.csv
  libraries.csv
  grocery.csv
  transit_stops.csv
  gtfs.zip
  housing_cost.csv
  risk.csv
```

Run the real variant:

```bash
python -m src.main --data-mode nyc --nyc-data-variant real --output-dir outputs/real_nyc --generate-reports
```

Run automatic fallback mode, which tries real data first, then `example_minimal`, then synthetic unless `--no-fallback` is set:

```bash
python -m src.main --data-mode nyc --nyc-data-variant auto --output-dir outputs/auto_step6 --generate-reports
```

Use `--nyc-raw-dir` to point at another local raw directory. Reports include `data_provenance` labels: `real_public_data`, `partial_real`, `fictional_minimal`, or `synthetic`. Partial-real runs may combine real POIs with generated candidate areas or other approximations. Travel times remain distance-based approximations, not routed travel times. See `docs/real_nyc_data_assumptions.md`.

## Research reports and paper figures

Successful demo runs now generate thesis-oriented outputs automatically. You can also pass `--generate-reports` explicitly:

```bash
python -m src.main --data-mode synthetic --generate-reports
python -m src.main --data-mode nyc --nyc-raw-dir data/raw/nyc/example_minimal --no-fallback --generate-reports
```

Additional outputs include `reward_decomposition.csv`, `profile_recommendation_explanations.md`, `baseline_ranking_by_profile.csv`, `baseline_ranking_by_profile.md`, `fairness_interpretation.md`, `method_comparison_summary.md`, `area_diagnostics.csv`, `recommendation_comparison.csv`, and paper figures under `outputs/paper_figures/` or the selected `--output-dir`. The RL entries should be interpreted as environment validation because the current policy is heuristic, not trained PPO.

## Run tests

```bash
pytest
```

The tests do not require internet access.

## Project structure

```text
urban_activity_recommendation/
  config/                  Data mode, NYC paths, reward, and environment settings
  data/raw/nyc/templates/  CSV templates for manual public-data onboarding
  data/raw/nyc/example_minimal/ Fictional minimal NYC-like dataset
  data/raw/nyc/            Manually downloaded NYC public data files
  data/processed/nyc/      Standardized NYC pipeline outputs
  data/sample/             Generated synthetic sample data
  docs/                    Data-source review, methodology, NYC pipeline docs
  src/data/                Synthetic and NYC data generation/loading/preprocessing
  src/graph/               NetworkX graph construction and area features
  src/profiles/            Representative individual profiles
  src/accessibility/       Accessibility and reward functions
  src/baselines/           Baseline recommendation methods
  src/rl/                  Environment, GNN placeholder, PPO placeholder
  src/evaluation/          Metrics, fairness, robustness helpers
  src/visualization/       Matplotlib plots
  tests/                   Unit tests
```

## Current limitations

- NYC mode is local-file based; it does not yet automate Socrata or Census API downloads.
- GeoJSON support is lightweight; centroid CSVs are the most reliable current input.
- Travel times are distance-based proxies, not routed travel times.
- POI quality scores are simple defaults unless source data provides a quality/rating field.
- PPO and GNN modules remain clean placeholders, not trained models.
- Reward design is preliminary and should be calibrated with domain experts and real data.

## Next development steps

1. Add dataset-specific NYC Open Data download adapters with pinned dataset IDs.
2. Add Census API or bulk ACS ingestion with margin-of-error handling.
3. Integrate OpenTripPlanner with OSM and GTFS for realistic multimodal travel times.
4. Add polygon-based tract/neighborhood processing with geopandas as an optional dependency.
5. Train and evaluate a real GNN-PPO policy against stronger baselines.
