import subprocess,sys
from pathlib import Path
R=Path(__file__).resolve().parents[1]
for s in ['validate_data.py','preprocess_pipeline.py','train_mlflow.py']:
 print(f'[INFO] ---> Executing src/{s}...'); subprocess.run([sys.executable,str(R/'src'/s)],check=True)
print('[SUCCESS] Air Quality Lab 4 MLflow Tracking Pipeline completed!')
