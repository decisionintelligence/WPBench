# Forecasting Data Placeholder

CSV datasets are not bundled in the GitHub-ready code artifact.

Download the selected 26-dataset bundle from Google Drive:

https://drive.google.com/drive/folders/1j8siMZ-SG6Q2yrfC0hPeAjG28fqImCib?usp=sharing

The uploaded bundle is named:

```text
wpbench_26_bundle_20260611/
  forecasting_wpbench_26/
  metadata/
  upload_manifest.csv
  README_UPLOAD.md
```

To run the released scripts, restore the CSVs under the artifact dataset root:

```bash
mkdir -p dataset/forecasting/forecasting_wp1_all_shapes_v1
cp -a wpbench_26_bundle_20260611/forecasting_wpbench_26/*.csv \
  dataset/forecasting/forecasting_wp1_all_shapes_v1/
mkdir -p dataset/metadata
cp -a wpbench_26_bundle_20260611/metadata/* dataset/metadata/
```

The 26-file manifest is included as:

```text
dataset/metadata/forecasting_wpbench_26_selected_files.csv
```

Some legacy Yalova scripts may reference
`forecasting_wp1_batch00_smoke/tfb-Yalova_final_ready.csv`. If needed, copy the
same CSV into `dataset/forecasting/forecasting_wp1_batch00_smoke/`.
