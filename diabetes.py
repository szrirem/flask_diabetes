# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 16:26:11 2022

@author: iremsezer
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pickle

#Veri setini içeri aktaralım
diyabet = pd.read_csv('https://raw.githubusercontent.com/cads-tedu/DSPG/master/Veri%20Setleri/diabetes.csv')

#İlk 5 gözlem
diyabet.head()

#Özet istatistikler
diyabet.describe().T

#Dağılımlar
axler = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
kolonlar = diyabet.columns
fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize = (10,10))
for i in range(len(kolonlar)):
    sns.histplot(diyabet[kolonlar[i]], ax = ax[axler[i]], kde = True)
    
# 0 değerlerini ortalama değerlerle doldurma
diyabet['BMI'] = diyabet['BMI'].replace(0,diyabet['BMI'].mean())
diyabet['BloodPressure'] = diyabet['BloodPressure'].replace(0,diyabet['BloodPressure'].mean())
diyabet['Glucose'] = diyabet['Glucose'].replace(0,diyabet['Glucose'].mean())
diyabet['Insulin'] = diyabet['Insulin'].replace(0,diyabet['Insulin'].mean())
diyabet['SkinThickness'] = diyabet['SkinThickness'].replace(0,diyabet['SkinThickness'].mean())

#Dağılımlar
axler = [(0,0), (0,1), (0,2), (1,0), (1,1), (1,2), (2,0), (2,1), (2,2)]
kolonlar = diyabet.columns
fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize = (10,10))
for i in range(len(kolonlar)):
    sns.histplot(diyabet[kolonlar[i]], ax = ax[axler[i]], kde = True)
    
#Kutu grafiği (Boxplot)
fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(data=diyabet)

fig, ax = plt.subplots(nrows = 2, ncols = 4, figsize = (13, 7))
kolonlar = diyabet.columns[0:8]
axler = [(0,0), (0,1), (0,2), (0,3), (1,0), (1,1), (1,2), (1,3)]
for i in range(len(kolonlar)):
    sns.stripplot(x = 'Outcome', y = kolonlar[i], data = diyabet, ax = ax[axler[i]])
    
#Bağımsız değişkenlerimiz ve bağımlı değişkenimiz
X = diyabet.drop('Outcome', axis = 1)
y = diyabet.Outcome

#Train-Test Ayırma
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

#Bağımsız nümerik değişkenlerimizi standardize edelim
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Modelimizi eğitelim
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#Test veri seti tahminleri
y_pred = classifier.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

#Modelimizin doğruluk oranı nedir? (Accuracy)
print(accuracy_score(y_test, y_pred))

conf = confusion_matrix(y_test,  y_pred)
print(conf)

#Eğri altında kalan (Area Under the Curve (AUC))
roc_auc_score(y_test, y_pred)

#ROC eğrisi
fpr, tpr, thresholds = roc_curve(y_test, y_pred)

plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % 0.734375)
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

#Modelimizi kaydedelim
pickle.dump(classifier, open('diabetes.pkl','wb'))




