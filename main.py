import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df=pd.read_csv("Anemia_Dataset.csv")
df.drop(columns=['Name'], inplace=True)
#print(df.head(5))
#print(df.isnull().sum())
#print(df.duplicated().sum())
#print(df.info())

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.metrics import accuracy_score,classification_report,roc_auc_score,roc_curve
X=df.drop(columns=['Anaemic'],axis=1)
y=df['Anaemic'].map({'No' : 0,'Yes' : 1})
print(y.value_counts())
catcols=X.select_dtypes(include='object').columns
numcols=X.select_dtypes(include='float64').columns
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
import math
#print(math.sqrt(len(y_train)))
preprocessing=ColumnTransformer(
    transformers=[
        ('cat',OneHotEncoder(),catcols),
        ('num',StandardScaler(),numcols)
    ]
)
model=Pipeline(steps=[
    ('preprocessor',preprocessing),
    ('classifier',KNeighborsClassifier(n_neighbors=9,metric='euclidean',weights='distance'))
])
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
y_proba=model.predict_proba(X_test)[:,1]

print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
fpr,tpr,thresholds=roc_curve(y_test,y_proba)
plt.figure()
plt.plot(fpr,tpr,label='area = %0.2f'%roc_auc_score(y_test,y_proba))
plt.plot([0,1],[0,1],'r--')
plt.legend(loc='best')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC CURVE")
plt.show()