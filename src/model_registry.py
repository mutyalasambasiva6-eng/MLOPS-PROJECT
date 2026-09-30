from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient


# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

MLFLOW_DB = ROOT / "mlflow.db"


# --------------------------------------------------
# MLflow configuration
# --------------------------------------------------

mlflow.set_tracking_uri(
    f"sqlite:///{MLFLOW_DB}"
)

client = MlflowClient()


MODEL_NAME = "Air_Quality_AQI_Model"


# --------------------------------------------------
# Find latest MLflow run
# --------------------------------------------------

experiment = client.get_experiment_by_name(
    "Air_Quality_AQI_Prediction"
)

if experiment is None:
    raise RuntimeError(
        "MLflow experiment not found. "
        "Run Lab 4 first."
    )


runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1
)

if not runs:
    raise RuntimeError(
        "No MLflow runs found. "
        "Run Lab 4 first."
    )


run = runs[0]

run_id = run.info.run_id


print(
    f"[INFO] Latest MLflow run: {run_id}"
)


# --------------------------------------------------
# Display metrics
# --------------------------------------------------

print("\n[INFO] Model Metrics")

for key, value in run.data.metrics.items():
    print(
        f"{key}: {value:.6f}"
    )


# --------------------------------------------------
# Register model
# --------------------------------------------------

model_uri = f"runs:/{run_id}/model"


try:

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=MODEL_NAME
    )

    version = registered.version

    print(
        f"\n[SUCCESS] Model registered"
    )

    print(
        f"Model: {MODEL_NAME}"
    )

    print(
        f"Version: {version}"
    )

except Exception as e:

    print(
        f"[WARNING] Model registration issue: {e}"
    )

    registered_versions = client.search_model_versions(
        f"name='{MODEL_NAME}'"
    )

    if not registered_versions:
        raise

    version = max(
        int(v.version)
        for v in registered_versions
    )

    print(
        f"[INFO] Existing model version: {version}"
    )


# --------------------------------------------------
# Add model metadata
# --------------------------------------------------

client.set_model_version_tag(
    MODEL_NAME,
    version,
    "project",
    "Air Quality AQI Prediction"
)

client.set_model_version_tag(
    MODEL_NAME,
    version,
    "target",
    "AQI >= 200"
)

client.set_model_version_tag(
    MODEL_NAME,
    version,
    "dataset",
    "city_day.csv"
)

client.set_model_version_tag(
    MODEL_NAME,
    version,
    "preprocessing",
    "imputation + scaling + one-hot encoding"
)

client.set_model_version_tag(
    MODEL_NAME,
    version,
    "validation_status",
    "validated"
)

client.set_model_version_tag(
    MODEL_NAME,
    version,
    "deployment_status",
    "ready_for_staging"
)


# --------------------------------------------------
# Add description
# --------------------------------------------------

client.update_model_version(
    name=MODEL_NAME,
    version=version,
    description=(
        "Air Quality AQI classification model. "
        "Target: AQI >= 200. "
        "Features include pollutant measurements, "
        "city and date-derived features."
    )
)


print(
    "\n[SUCCESS] Model Registry update completed!"
)

print(
    f"Registered Model: {MODEL_NAME}"
)

print(
    f"Version: {version}"
)

print(
    "Status: ready_for_staging"
)