# Checkpoints

Large foundation/STFM checkpoints are not bundled by default.

The minimal checkpoint bundle for the six checkpoint-dependent algorithms is
hosted on Google Drive:

https://drive.google.com/drive/folders/1sK-XpZl1J7DGdZW_xoZsrbFnLrgFZiCe?usp=sharing

It covers:

- FactoST_STA
- FactoST_UTP
- OpenCity_STFM
- SEMPO
- TinyTimeMixer
- Toto

After download, restore it from the artifact root:

```bash
cp -a wpbench_6algos_checkpoints_20260611/checkpoints ./
```

Scripts refer to relative paths under:

- `checkpoints/tsfm/`
- `checkpoints/stfm/`

The source manifest for this bundle is `checkpoint_upload_manifest.csv` inside
the Google Drive folder. The full checkpoint inventory originally observed in
the development repository is recorded in `checkpoints/checkpoint_manifest.csv`.
