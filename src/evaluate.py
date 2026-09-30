from pathlib import Path
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score
ROOT=Path(__file__).resolve().parents[1]; P=ROOT/'data'/'processed'
Xtr=sparse.load_npz(P/'X_train_final.npz'); Xte=sparse.load_npz(P/'X_test_final.npz'); ytr=np.load(P/'y_train.npy'); yte=np.load(P/'y_test.npy')
models={'logistic_regression_baseline':LogisticRegression(max_iter=1000,class_weight='balanced'),'decision_tree_baseline':DecisionTreeClassifier(max_depth=12,class_weight='balanced',random_state=42),'random_forest_baseline':RandomForestClassifier(n_estimators=100,max_depth=14,class_weight='balanced',n_jobs=-1,random_state=42)}
rows=[]
for name,m in models.items():
 m.fit(Xtr,ytr); p=m.predict(Xte); prob=m.predict_proba(Xte)[:,1]; rows.append([name,accuracy_score(yte,p),precision_score(yte,p,zero_division=0),recall_score(yte,p,zero_division=0),f1_score(yte,p,zero_division=0),roc_auc_score(yte,prob)])
r=pd.DataFrame(rows,columns=['model','accuracy','precision','recall','f1','roc_auc']); print(r.to_string(index=False)); print('Best model by F1:',r.loc[r.f1.idxmax(),'model'])
r.to_csv(ROOT/'outputs'/'model_comparison.csv',index=False); (ROOT/'outputs').mkdir(exist_ok=True)
