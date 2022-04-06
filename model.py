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

#Bağımsız nümerik değişkenlerimizi standardize edelim
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from statsmodels.stats.outliers_influence import variance_inflation_factor

#İlk değişkenin vif skoru
variance_inflation_factor(X_scaled, 0)

#İkinci değişkenin vif skoru
variance_inflation_factor(X_scaled, 1)

#Boş dataframe
vif = pd.DataFrame()
vif['Degiskenler'] = X.columns
vif['VIF'] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

#Train-Test Ayırma
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.25, random_state = 42)

#Bağımsız değişkenlerimizin train veri seti
print(X_train)
print(X_scaled.shape)
print(X_train.shape)

#Bağımsız değişkenlerimizin train veri seti
print(X_test)
print(X_scaled.shape)
print(X_test.shape)

#Modelimizi eğitelim
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

#Model katsayıları
pd.DataFrame({'Degiskenler':X.columns, 'Katsayılar':logreg.coef_[0]})

#Test veri seti tahminleri
logreg.predict(X_test)

#Tahminlerimizi bir değişkene atayalım
tahminler = logreg.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, roc_auc_score

#Modelimizin doğruluk oranı nedir? (Accuracy)
print(accuracy_score(y_test, tahminler))

conf = confusion_matrix(y_test, tahminler)
print(conf)

#Eğri altında kalan (Area Under the Curve (AUC))
roc_auc_score(y_test, tahminler)

#ROC eğrisi
fpr, tpr, thresholds = roc_curve(y_test, tahminler)

plt.plot(fpr, tpr, color='orange', label='ROC curve (area = %0.2f)' % 0.734375)
plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--',)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

pickle.dump(logreg, open('diabetes.pkl','wb'))




