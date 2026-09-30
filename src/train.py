from pathlib import Path
import numpy as np, joblib
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score
ROOT=Path(__file__).resolve().parents[1]; P=ROOT/'data'/'processed'; M=ROOT/'models'; M.mkdir(exist_ok=True)
Xtr=sparse.load_npz(P/'X_train_final.npz'); Xte=sparse.load_npz(P/'X_test_final.npz'); ytr=np.load(P/'y_train.npy'); yte=np.load(P/'y_test.npy')
models={'logistic_regression':LogisticRegression(max_iter=1000,class_weight='balanced'),'decision_tree':DecisionTreeClassifier(max_depth=12,class_weight='balanced',random_state=42),'random_forest':RandomForestClassifier(n_estimators=100,max_depth=14,class_weight='balanced',n_jobs=-1,random_state=42)}
results={}
for name,model in models.items():
 model.fit(Xtr,ytr); pred=model.predict(Xte); results[name]=recall_score(yte,pred,zero_division=0); print(f'{name}: recall={results[name]:.4f}')
best=max(results,key=results.get); joblib.dump(models[best],M/'best_aqi_model.pkl'); print(f'Best baseline by recall: {best} ({results[best]:.4f})')
