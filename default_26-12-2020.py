#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
import math
from scipy.stats import f_oneway
get_ipython().run_line_magic('matplotlib', 'inline')
from scipy import stats
import datetime
import warnings
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from scipy.stats import zscore
from statsmodels.graphics.regressionplots import influence_plot
from sklearn.metrics import r2_score,mean_squared_error
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn import metrics
import statsmodels.stats.outliers_influence
from sklearn.tree import export_graphviz
from IPython.display import Image
from sklearn.tree import DecisionTreeClassifier
import pydotplus as pdot
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.utils import resample
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle
warnings.filterwarnings('ignore')
from sklearn.model_selection import LeaveOneOut
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVC
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.ensemble import VotingClassifier
from collections import Counter
from sklearn.decomposition import PCA


# In[2]:


os.chdir('D:\DS')


# In[3]:


df=pd.read_excel('ds_3_1.xlsx')


# In[4]:


df.info()


# In[5]:


df=df.fillna(0)


# In[6]:


df['Status'].value_counts()


# In[7]:


x=df.drop(['Status','Name'],axis=1)


# In[8]:


y=df.Status


# In[9]:


for i in x.columns:
    x[i]=(x[i]-min(x[i]))/(max(x[i]-min(x[i])))


# In[10]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[11]:


model=ExtraTreesClassifier()


# In[12]:


model.fit(x_train,y_train)


# In[13]:


model.feature_importances_


# In[14]:


importance=pd.Series(model.feature_importances_,index=x_train.columns)


# In[15]:


importance.nlargest(20).plot(kind='barh')


# In[16]:


importance.columns=['Scores']


# In[17]:


tree=pd.DataFrame(importance)


# In[18]:


tree.columns=['Scores']


# In[19]:


tree['importance']=np.round((tree['Scores']/tree['Scores'].sum())*100)


# In[20]:


tree=tree.sort_values('importance',ascending=False)


# In[21]:


tree.head(10)


# In[22]:


best=SelectKBest(score_func=chi2,k=5)
fit=best.fit(x_train,y_train)


# In[23]:


dfscores=pd.DataFrame(fit.scores_)
dfcolumns=pd.DataFrame(x_train.columns)


# In[24]:


featureScores=pd.concat([dfcolumns,dfscores],axis=1)
featureScores.columns=['Variable','Scores']


# In[25]:


chi=pd.DataFrame(featureScores.reset_index())
chi['importance']=np.round((chi['Scores']/chi['Scores'].sum())*100)


# In[26]:


chi=chi.sort_values('importance',ascending=False)
chi.head(10)


# In[27]:


features=['b','a','g','d','e']


# In[28]:


# Top 5 variables explain 96% of the variation


# # UNSUPERVISED LEARNING 

# In[29]:


from sklearn.cluster import KMeans


# In[34]:


df_cluster=df.drop(['Name'],axis=1)
clusters_new=KMeans(3,random_state=42)
clusters_new.fit(df_cluster)
df_cluster['cluster_id']=clusters_new.labels_


# In[47]:


markers=['+','^','.']


# In[48]:


sns.lmplot('Status','b',data=df_cluster,hue='cluster_id',fit_reg=False,markers=markers,size=4)


# In[49]:


sns.lmplot('Status','a',data=df_cluster,hue='cluster_id',fit_reg=False,markers=markers,size=4)


# In[50]:


sns.lmplot('Status','g',data=df_cluster,hue='cluster_id',fit_reg=False,markers=markers,size=4)


# In[51]:


sns.lmplot('Status','d',data=df_cluster,hue='cluster_id',fit_reg=False,markers=markers,size=4)


# In[52]:


sns.lmplot('Status','e',data=df_cluster,hue='cluster_id',fit_reg=False,markers=markers,size=4)


# # Logistic Regression

# In[199]:


logit=sm.Logit(y_train,x_train)
logit_model=logit.fit()


# In[200]:


logit_model.summary2()


# In[201]:


features_1=['b','e','h']


# In[202]:


logit=LogisticRegression()
logit.fit(x_train[features_1],y_train)


# In[203]:


pred_logit=logit.predict(x_test[features_1])


# In[204]:


accuracy_score(y_test,pred_logit)


# In[205]:


def draw_roc_curve(actual,probs):
    fpr,tpr,thresholds=metrics.roc_curve(actual,probs,drop_intermediate=False)
    auc_score=metrics.roc_auc_score(actual,probs)
    plt.figure(figsize=(8,6))
    plt.plot(fpr,tpr,label='ROC Curve (area=%0.2f)'%auc_score)
    plt.plot([0,1],[0,1],'k--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel(['False Positive Rate'])
    plt.ylabel(['True Positive Rate'])
    plt.title(['ROC Curve'])
    plt.legend(loc='lower right')
    plt.show()
    return fpr,tpr,thresholds


# In[206]:


draw_roc_curve(y_test,pred_logit)


# In[207]:


ada_clf=AdaBoostClassifier(logit,n_estimators=50)


# In[208]:


ada_clf.fit(x_train[features_1],y_train)


# In[209]:


ada_pred=ada_clf.predict(x_test[features_1])


# In[210]:


draw_roc_curve(y_test,ada_pred)


# In[211]:


gboost_clf=GradientBoostingClassifier(n_estimators=500,max_depth=10)
gboost_clf.fit(x_train[features],y_train)


# In[212]:


gboost_pred=gboost_clf.predict(x_test[features])


# In[213]:


draw_roc_curve(y_test,gboost_pred)


# In[214]:


tuned_parameters=[{'n_neighbors':range(3,10),'metric':['canberra','euclidean','minkowski']}]
clf=RandomizedSearchCV(KNeighborsClassifier(),tuned_parameters,cv=5,scoring='roc_auc')
clf.fit(x_train[features],y_train)


# In[215]:


clf.best_score_


# In[216]:


clf.best_params_


# In[217]:


tuned_parameters=[{'n_neighbors':range(3,10),'metric':['canberra','euclidean','minkowski']}]
grid_clf=GridSearchCV(KNeighborsClassifier(),tuned_parameters,cv=5,scoring='roc_auc')
grid_clf.fit(x_train[features],y_train)


# In[218]:


grid_clf.best_score_


# In[219]:


grid_clf.best_params_


# In[220]:


knn_clf=KNeighborsClassifier(metric='canberra', n_neighbors= 8)
knn_clf.fit(x_train[features],y_train)


# In[221]:


knn_pred=knn_pred=knn_clf.predict(x_test[features])


# In[222]:


draw_roc_curve(y_test,knn_pred)


# In[227]:


tuned_parameters=[{'max_depth':[10,15,20,25],'n_estimators':[10,20,30,40],'max_features':['sqrt',0.2]}]
tree=RandomForestClassifier()
clf=RandomizedSearchCV(tree,tuned_parameters,cv=5,scoring='roc_auc')
clf.fit(x_train[features],y_train)


# In[228]:


clf.best_score_


# In[229]:


clf.best_params_


# In[230]:


tuned_parameters=[{'max_depth':[10,15,20,25],'n_estimators':[10,20,30,40],'max_features':['sqrt',0.2]}]
tree=RandomForestClassifier()
grid_rdm_clf=GridSearchCV(tree,tuned_parameters,cv=5,scoring='roc_auc')
grid_rdm_clf.fit(x_train[features],y_train)


# In[231]:


grid_rdm_clf.best_score_


# In[232]:


grid_rdm_clf.best_params_


# In[233]:


random_clf=RandomForestClassifier(max_depth=15,max_features='sqrt',n_estimators=30)


# In[234]:


random_clf.fit(x_train[features],y_train)


# In[235]:


rdm_pred=random_clf.predict(x_test[features])


# In[236]:


draw_roc_curve(y_test,rdm_pred)


# In[237]:


svm_clf=SVC(kernel='linear',random_state=0)
svm_clf.fit(x_train[features],y_train)


# In[238]:


svc_lin_pred=svm_clf.predict(x_test[features])


# In[239]:


draw_roc_curve(y_test,svc_lin_pred)


# In[240]:


poly_svm_clf=SVC(kernel='poly',degree=3,random_state=0)
poly_svm_clf.fit(x_train[features],y_train)


# In[241]:


svc_poly_pred=poly_svm_clf.predict(x_test[features])


# In[242]:


draw_roc_curve(y_test,svc_poly_pred)


# In[243]:


rbf_svm_clf=SVC(kernel='rbf',gamma=5,random_state=0)
rbf_svm_clf.fit(x_train[features],y_train)


# In[244]:


svc_rbf_pred=rbf_svm_clf.predict(x_test[features])


# In[245]:


draw_roc_curve(y_test,svc_rbf_pred)


# In[246]:


softmax_reg=LogisticRegression(solver='lbfgs')
softmax_reg.fit(x_train[features_1],y_train)


# In[247]:


softmax_pred=softmax_reg.predict(x_test[features_1])


# In[248]:


draw_roc_curve(y_test,softmax_pred)


# In[274]:


voting_clf=VotingClassifier(estimators=[('svm',rbf_svm_clf),('tree',random_clf),('gboost',gboost_clf)],voting='hard')


# In[275]:


voting_clf.fit(x_train[features],y_train)


# In[276]:


voting_pred=voting_clf.predict(x_test[features])


# In[277]:


draw_roc_curve(y_test,voting_pred)


# In[254]:


model=tf.keras.models.Sequential()


# In[255]:


model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(25,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(25,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(25,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(25,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(25,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(25,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2,activation=tf.nn.softmax))


# In[256]:


model.compile(Adam(lr=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[257]:


model.fit(x_train[features],y_train,batch_size=5,epochs=30)


# In[258]:


pred_ann=model.predict_classes(x_test[features],batch_size=5)


# In[259]:


draw_roc_curve(y_test,pred_ann)


# # MODEL VALIDATION

# # Random Forest Classifier

# In[260]:


print(metrics.classification_report(y_test,rdm_pred))


# # CROSS VALIDATION

# In[267]:


cv_scores=cross_val_score(random_clf,x_test[features],y_test,cv=5,scoring='roc_auc')
print(cv_scores)
print('Mean Accuracy:',np.mean(cv_scores),'std deviation:',np.std(cv_scores))


# In[264]:


from sklearn.model_selection import cross_val_score
from sklearn import model_selection


# In[266]:


loocv=model_selection.LeaveOneOut()
cv_scores=model_selection.cross_val_score(random_clf,x_test[features],y_test,cv=loocv)
print(cv_scores)
print('Mean Accuracy:',np.mean(cv_scores),'std deviation:',np.std(cv_scores))


# # VOTING CLASSIFIER

# In[278]:


print(metrics.classification_report(y_test,gboost_pred))


# # CROSS VALIDATION

# # Artificial Neural Network

# In[280]:


print(metrics.classification_report(y_test,pred_ann))


# In[285]:


from sklearn.model_selection import KFold


# In[289]:


batch_size = 5
no_classes = 5
no_epochs = 30
optimizer = Adam()
verbosity = 1
num_folds = 5


# In[282]:


acc_per_fold = []
loss_per_fold = []


# In[283]:


inputs = np.concatenate((x_train, x_test), axis=0)
targets = np.concatenate((y_train,y_test), axis=0)


# In[287]:


kfold = KFold(n_splits=5, shuffle=True)


# In[294]:


fold_no = 1
for train,test in kfold.split(x, y):
    model = Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(25,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(25,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(25,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(25,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(25,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(25,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(2,activation=tf.nn.softmax))


# In[295]:


model.compile(Adam(lr=0.001),loss='sparse_categorical_crossentropy',metrics=['accuracy'])


# In[302]:


history = model.fit(x_test, y_test,
              batch_size=batch_size,
              epochs=no_epochs,
              verbose=verbosity)


# In[303]:


print('------------------------------------------------------------------------')
print(f'Training for fold {fold_no} ...')


# In[307]:


print('------------------------------------------------------------------------')
print(f'Training for fold {fold_no} ...')


# In[311]:


print('------------------------------------------------------------------------')
print(f'Training for fold {fold_no} ...')


# In[315]:


print('------------------------------------------------------------------------')
print(f'Training for fold {fold_no} ...')


# In[316]:


scores = model.evaluate(inputs[test], targets[test], verbose=0)
print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
acc_per_fold.append(scores[1] * 100)
loss_per_fold.append(scores[0])


# In[317]:


fold_no = fold_no + 1


# In[318]:


print('------------------------------------------------------------------------')
print('Score per fold')
for i in range(0, len(acc_per_fold)):
  print('------------------------------------------------------------------------')
  print(f'> Fold {i+1} - Loss: {loss_per_fold[i]} - Accuracy: {acc_per_fold[i]}%')
print('------------------------------------------------------------------------')
print('Average scores for all folds:')
print(f'> Accuracy: {np.mean(acc_per_fold)} (+- {np.std(acc_per_fold)})')
print(f'> Loss: {np.mean(loss_per_fold)}')
print('------------------------------------------------------------------------')


# # GENERATING CONFUSION MATRIX

# In[319]:


def draw_cm(actual,predicted):
    cm=metrics.confusion_matrix(actual,predicted,[1,0])
    sns.heatmap(cm,annot=True,fmt='.2f',
    xticklabels=['Default','Standard'],
    yticklabels=['Default','Standard'])
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.autoscale(enable=True,axis='y')
    plt.show()


# In[320]:


draw_cm(y_test,pred_ann)


# In[321]:


print(metrics.classification_report(y_test,pred_ann))


# In[ ]:




