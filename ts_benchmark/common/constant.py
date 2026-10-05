# -*- coding: utf-8 -*-
import os

# Artifact root. Environment variables allow the same code to run from a copied
# artifact, a mounted dataset directory, or the original development checkout.
ROOT_PATH = os.environ.get(
    "WPBENCH_ROOT",
    os.path.abspath(os.path.join(__file__, "..", "..", "..")),
)

# Build the path to the forecasting dataset folder.
FORECASTING_DATASET_PATH = os.environ.get(
    "WPBENCH_FORECASTING_DATASET_PATH",
    os.path.join(ROOT_PATH, "dataset", "forecasting"),
)

# Benchmark config path.
CONFIG_PATH = os.environ.get(
    "WPBENCH_CONFIG_PATH",
    os.path.join(ROOT_PATH, "configs"),
)


# Result output root for benchmark records and reports.
RESULT_ROOT = os.environ.get(
    "WPBENCH_RESULT_ROOT",
    os.path.join(ROOT_PATH, "results", "raw_metrics"),
)

# Third-party library path. Kept for compatibility with the original TFB runner.
THIRD_PARTY_PATH = os.environ.get(
    "WPBENCH_THIRD_PARTY_PATH",
    os.path.join(ROOT_PATH, "ts_benchmark", "baselines", "third_party"),
)
