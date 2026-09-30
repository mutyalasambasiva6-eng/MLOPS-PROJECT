from pathlib import Path
import numpy as np, joblib, mlflow, mlflow.sklearn
from scipy import sparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1]; P=ROOT/'data'/'processed'; M=ROOT/'models'; M.mkdir(exist_ok=True)
Xtr=sparse.load_npz(P/'X_train_final.npz'); Xte=sparse.load_npz(P/'X_test_final.npz'); ytr=np.load(P/'y_train.npy'); yte=np.load(P/'y_test.npy')
params={'n_estimators':150,'max_depth':14,'random_state':42,'class_weight':'balanced','n_jobs':-1}
mlflow.set_tracking_uri(f"sqlite:///{ROOT/'mlflow.db'}"); mlflow.set_experiment('Air_Quality_AQI_Prediction')
with mlflow.start_run(run_name='RandomForest_AQI'):
 mlflow.log_params(params); mlflow.log_param('target_definition','AQI >= 200')
 model=RandomForestClassifier(**params); model.fit(Xtr,ytr); pred=model.predict(Xte); prob=model.predict_proba(Xte)[:,1]
 metrics={'accuracy':accuracy_score(yte,pred),'precision':precision_score(yte,pred,zero_division=0),'recall':recall_score(yte,pred,zero_division=0),'f1':f1_score(yte,pred,zero_division=0),'roc_auc':roc_auc_score(yte,prob)}
 mlflow.log_metrics(metrics); mlflow.sklearn.log_model(model,'model',skops_trusted_types=['sklearn.tree._tree.Tree']); joblib.dump(model,M/'random_forest_mlflow.pkl'); print(metrics)
