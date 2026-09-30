import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_script(script):

    print(
        f"[INFO] ---> Executing src/{script}..."
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "src" / script)
        ],
        check=True
    )


def main():

    run_script("validate_data.py")

    run_script("preprocess_pipeline.py")

    run_script("train_mlflow.py")

    run_script("model_registry.py")

    print(
        "\n[SUCCESS] "
        "Air Quality Lab 6 Model Registry Pipeline completed!"
    )


if __name__ == "__main__":
    main()