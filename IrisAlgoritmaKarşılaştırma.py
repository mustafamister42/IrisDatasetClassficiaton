# -*- coding: utf-8 -*-
"""
Created on Sat Jan 18 11:47:42 2025

@author: Mustafa
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix
veriler=pd.read_excel("Iris.xls")
print(veriler)
x=veriler.iloc[:,1:4].values#bağımsız değişken
y=veriler.iloc[:,4:]#bağımlı değişken
print(y)

#VERİ KÜMESİNİN EĞİTİM VE TEST OLARAK BÖLÜNMESİ
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33,random_state=0)

#VERİ ÖZNİTELİK ÖLÇEKLEME
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(x_train)
X_test=sc.transform(x_test)

#BURADAN İTİBAREN SINIFLANDIRMA ALGORİTMALARI BAŞLAR
#1- LOGİSTİC REGRESSİON
from sklearn.linear_model import LogisticRegression
logr=LogisticRegression(random_state=0)
logr.fit(X_train,y_train)
y_pred=logr.predict(X_test)
print(y_pred)
print(y_test)

#KARMAŞIKLIK MATRİSİ
#CONFUSİON MATRİX
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print("Logistic Regression")
print(cm)

#2-K-NN(K NEAREST NEİGHBORHOOD) ALGORİTMASI
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1,metric="minkowski")
knn.fit(X_train, y_train)
y_pred2=knn.predict(X_test)
cm=confusion_matrix(y_test, y_pred2)
print("KNN")
print(cm)

#3-SUPPORT VECTOR CLASSİFİCATİON İLE SINIFLADNIRMA
from sklearn.svm import SVC
svc=SVC(kernel="rbf")
svc2=SVC(kernel="poly")
svc3=SVC(kernel="linear")
svc4=SVC(kernel="sigmoid")

svc.fit(X_train,y_train)
svc2.fit(X_train,y_train)
svc3.fit(X_train,y_train)
svc4.fit(X_train,y_train)

y_pred3=svc.predict(X_test)
y_pred4=svc.predict(X_test)
y_pred5=svc.predict(X_test)
y_pred6=svc.predict(X_test)
cm=confusion_matrix(y_test,y_pred3)
cm2=confusion_matrix(y_test, y_pred4)
cm3=confusion_matrix(y_test, y_pred5)
cm4=confusion_matrix(y_test, y_pred6)

print("SVC(RBF)")
print(cm)
print("SVC(POLY)")
print(cm2)
print("SVC(LINEAR)")
print(cm3)
print("SVC(SIGMOID)")
print(cm4)

#KERNEK-L FONKSİYONUNUN ALABİLECEĞİ FARKLI PARAMETRELER DENENMİŞ OLUP HEPSİNDE AYNI SONUÇ ÇIKMIŞTIR.

#4-NAİVE BAYES
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import ComplementNB
gnb=GaussianNB()
bnb=BernoulliNB()
gnb.fit(X_train,y_train)
bnb.fit(X_train, y_train)
y_pred4=gnb.predict(X_test)
y_pred5=bnb.predict(X_test)
cm=confusion_matrix(y_test, y_pred4)
cm2=confusion_matrix(y_test, y_pred5)
print("BNB")
print(cm2)
print("GNB")
print(cm)


#5-DECISION TREE
from sklearn.tree import DecisionTreeClassifier
dct=DecisionTreeClassifier(criterion="entropy")
dct2=DecisionTreeClassifier(criterion="log_loss")
dct3=DecisionTreeClassifier(criterion="gini")

dct.fit(X_train,y_train)
dct2.fit(X_train,y_train)
dct3.fit(X_train,y_train)

y_pred=dct.predict(X_test)
y_pred2=dct2.predict(X_test)
y_pred3=dct3.predict(X_test)


cm=confusion_matrix(y_test, y_pred)
cm2=confusion_matrix(y_test, y_pred)
cm3=confusion_matrix(y_test, y_pred)


print("DTC(entropy)")
print(cm)
print("DTC(log_loss)")
print(cm2)
print("DTC(gini)")
print(cm3)
#3 FARKLI CRİTERİON YAPISINDA DA AYNI SONUÇU VERDİ

#6-RANDOM FOREST
from sklearn.ensemble import RandomForestClassifier
rfc=RandomForestClassifier(n_estimators=10,criterion="gini")
rfc.fit(X_train,y_train)
y_pred=rfc.predict(X_test)
cm=confusion_matrix(y_test, y_pred)
print("RFC")
print(cm)

#Yine 3 farklı criterion denenmiş olup sonuçlar birbirine çok yakın çıkmıştır.
'''
## GENEL SONUÇLARI KARŞILAŞTIRDIĞIMIZDA SVC VE DTC ALGORİTMA SINIFLARININ BU VERİ KÜMESİNDE DAHA İYİ SONUÇ VERDİĞİNİ GÖRÜYORUZ.