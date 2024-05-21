#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sqlalchemy import create_engine
import pymysql
import configparser

import pandas as pd 
import numpy as np


# In[2]:


config = configparser.ConfigParser()
config.read('sql.ini')

# Compiling login info
DB_TYPE = config['default']['DB_TYPE']
DB_DRIVER = config['default']['DB_DRIVER']
DB_USER = config['default']['DB_USER']
DB_PASS = config['default']['DB_PASS']
DB_HOST = config['default']['DB_HOST']
DB_PORT = config['default']['DB_PORT']
DB_NAME = config['default']['DB_NAME']

SQLALCHEMY_DATABASE_URI = f'{DB_TYPE}+{DB_DRIVER}://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}'

# Creating engine with login info
engine = create_engine(SQLALCHEMY_DATABASE_URI)
print(engine)
# this connects to the sql engine
con = engine.connect()


# ## Load Data

# In[3]:


df = pd.read_sql_table('ctrdata', con)
con.close()
engine.dispose()


# In[4]:


print("Shape: ", df.shape, "\n")
display(df.head())


# In[5]:


display(df.info())


# In[6]:


#replace empty string with Null and count total number of Null cells
df = df.replace([''], None)
print(df.isnull().sum())


# ## Data Cleaning

# In[7]:


#drop empty columns
df.drop(['min_money', 'order_num', 'idu_category', 'price', 'ad_network'], axis=1, inplace=True)


# In[8]:


#Drop row_id, begin_time, end_time, pic_url, ad_url, ad_desc_url columns because it's not useful for analysis
df.drop(['row_id', 'begin_time', 'end_time', 'pic_url', 'ad_url', 'ad_desc_url'], axis=1, inplace=True)


# In[9]:


#display unique values in categorical columns

col=['kind_pay', 'kind_card', 'store_id', 'network', 'industry','gender', 'ad_loc', 'ad_label', 'ad_Copy', 'mid', 'maid',
'city_id', 'click_hide', 'sys', 'user_gender', 'payment_kind']
for c in col:
    print(c, ": ", df[c].unique(), "\n")


# In[10]:


#Drop mid, maid, user_gender, payment_kind columns because they contain duplicate information
df.drop(['mid', 'maid', 'user_gender', 'payment_kind'], axis=1, inplace=True)


# In[11]:


#Drop click_hide, sys, city_id columns because they are not useful for analysis
df.drop(['click_hide', 'sys', 'city_id'], axis=1, inplace=True)


# In[12]:


#Drop address, ad_Copy column because cannot extract information 
df.drop(['address', 'ad_Copy'], axis=1, inplace=True)


# In[13]:


#Drop all rows where view_time is Null, because view_time=Null means user did not see the ad, therefore
#directly implies clicked=0.
df.dropna(subset=['view_time'], inplace=True)


# In[14]:


#In ad_loc, replace numpy.nan with 0
df['ad_loc'].replace(np.nan, 0, inplace=True)
print("ad_loc: ", df["ad_loc"].unique(), "\n")

#In gender, replace Null value with "male/female".
df["gender"].fillna("male/female", inplace = True)
print("gender: ", df["gender"].unique(), "\n")


# In[ ]:





# ## Data Exploration

# In[15]:


import plotly.express as ex
import plotly.graph_objects as go

display(df.head(10))


# In[16]:


clicks=df['click_time'].count()
impressions=df['view_time'].count()

# Create a bar chart
fig = go.Figure(data=[go.Bar(x=['Impressions', 'Clicks'], y=[impressions, clicks], marker_color=['orange', 'green'], 
                             text=[impressions, clicks])])
fig.update_layout(title='Click Through Rate of the dataset is {:.2f}%.'.format(100*clicks/impressions), bargap = 0.6,
                  width=500, height=600)

# Display the chart
fig.show()


# In[17]:


crtfig=ex.pie(df,names='clicked',title='Click Through Rate',hole=0.33)
crtfig.update_traces(textinfo='value+percent')
crtfig.show()


# In[18]:


ex.pie(df,names='gender',title='Proportion Of Genders',hole=0.33)


# In[19]:


ex.pie(df,names='kind_card',title='Payment Type',hole=0.33)


# In[20]:


ex.pie(df,names='network',title='Proportion Of Network Type',hole=0.33)


# In[21]:


ex.pie(df,names='kind_pay',title='Prorpotion Of Payment Kind',hole=0.33)


# In[22]:


ex.pie(df,names='ad_loc',title='Proportion Of Ad Location',hole=0.33)


# In[23]:


ex.pie(df,names='ad_label',title='Proportion Of Ad Label',hole=0.33)


# In[24]:


fig_ind=ex.pie(df,names='industry',title='Proportion Of Industry', hole=0.33)
fig_ind.update_traces(textposition='inside')


# In[25]:


fig_ind=ex.pie(df,names='ad_id',title='Proportion by Ad ID', hole=0.33)
fig_ind.update_traces(textposition='inside')


# In[26]:



days=df['payment_time'].dt.hour
days = pd.DataFrame({'hour':days.values})

vc=days.value_counts().reset_index()

vc.columns=['hour','count']
vcfig = ex.bar(vc, x='hour', y='count', title='Payment by Hour')
vcfig.update_xaxes(tickmode='linear')
# Display the chart
vcfig.show()


# In[27]:


days=df['click_time'].dt.hour
days = pd.DataFrame({'hour':days.values})

vc=days.value_counts().reset_index()

vc.columns=['hour','count']
vcfig = ex.bar(vc, x='hour', y='count', title='Clicks by Hour')
vcfig.update_xaxes(tickmode='linear')
# Display the chart
vcfig.show()


# In[28]:


user_imp = df.groupby('user_id')['view_time'].count().sort_values(ascending=False).head(10)
ex.bar(x=user_imp.index, y=user_imp.values, title='Top 10 Users by Impressions')


# In[29]:


user_click = df.groupby('user_id')['click_time'].count().sort_values(ascending=False).head(10)
ex.bar(x=user_click.index, y=user_click.values, title='Top 10 Users by Clicks')


# ## Feature Engineering

# In[30]:


#Add column that shows total number of impressions by user_id
user_impressions = df.groupby('user_id')['view_time'].count()
df['user_impressions'] = df['user_id'].map(user_impressions)


# In[31]:


#Drop user_id column so the model is more generalizable 
df.drop(['user_id'], axis=1, inplace=True)


# In[32]:


#Split view_time into 2 columns: dayofweek column and hour column

df["view_time_day"]=df["view_time"].dt.dayofweek
df["view_time_hour"]=df["view_time"].dt.hour


# In[33]:


#add features that represent the time difference between payment and view

df["payment_view_diff"] = (df["view_time"]-df["payment_time"]).dt.total_seconds()


# In[34]:


#Drop payment_time, view_time and click_time columns 
#Drop click_time because missing click_time implies clicked=0 

df.drop(['payment_time', 'view_time', 'click_time'], axis=1, inplace=True)


# In[35]:


target=df['clicked']
df.drop(['clicked'], axis=1, inplace=True)
df.insert(0, 'clicked', target)


# In[36]:


import seaborn as sns


display(df.corr().style.background_gradient(cmap='coolwarm',axis=None))
sns.heatmap(df.corr());


# In[37]:


#One-hot encode all the categorical features

df_kindpay = pd.get_dummies(df['kind_pay'], prefix='kind_pay')
df = pd.concat([df, df_kindpay], axis=1)
del df_kindpay

df_kindcard = pd.get_dummies(df['kind_card'], prefix='kind_card')
df = pd.concat([df, df_kindcard], axis=1)
del df_kindcard

df_network = pd.get_dummies(df['network'], prefix='network')
df = pd.concat([df, df_network], axis=1)
del df_network

df_industry = pd.get_dummies(df['industry'], prefix='industry')
df = pd.concat([df, df_industry], axis=1)
del df_industry

df_gender = pd.get_dummies(df['gender'], prefix='gender')
df = pd.concat([df, df_gender], axis=1)
del df_gender

df_adid = pd.get_dummies(df['ad_id'], prefix='ad_id')
df = pd.concat([df, df_adid], axis=1)
del df_adid

df_adloc = pd.get_dummies(df['ad_loc'], prefix='ad_loc')
df = pd.concat([df, df_adloc], axis=1)
del df_adloc

df_adlabel = pd.get_dummies(df['ad_label'], prefix='ad_label')
df = pd.concat([df, df_adlabel], axis=1)
del df_adlabel


# In[38]:


#Drop one-hot encoded columns 
df.drop(['kind_pay', 'kind_card', 'network', 'industry', 'gender', 'ad_id', 'ad_loc', 'ad_label'], axis=1, inplace=True)


# In[ ]:





# ## Train Test Split

# In[39]:


get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install imbalanced-learn')
from sklearn.preprocessing import TargetEncoder
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate


# In[40]:




X = df.drop(columns='clicked')
y = df[['clicked']]
X.columns = X.columns.astype(str)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=88)


# In[ ]:





# ## Target Encoding

# In[41]:



target_encoder = TargetEncoder()

#Target encode the store_id feature
X_train['store_id_encoded'] = target_encoder.fit_transform(X_train['store_id'].values.reshape(-1, 1), y_train['clicked'])
# Use the fitted encoder to transform the test set
X_test['store_id_encoded'] = target_encoder.transform(X_test['store_id'].values.reshape(-1, 1))

#Drop store_id columns
X_train.drop(['store_id'], axis=1, inplace=True)
X_test.drop(['store_id'], axis=1, inplace=True)


# ## Smote Oversampling

# In[42]:


from imblearn.under_sampling import RandomUnderSampler
us = RandomUnderSampler(random_state=88)

X_resampled, y_resampled = us.fit_resample(X_train, y_train)

X_resampled.shape


# In[56]:


y_resampled.value_counts()


# In[57]:


y_train.value_counts()


# ## Baseline Model

# In[43]:


from sklearn.metrics import confusion_matrix

dt = DecisionTreeClassifier()
dt.fit(X_resampled, y_resampled)

y_pred = dt.predict(X_test)

print(classification_report(y_test, y_pred))


# In[44]:


from sklearn.metrics import accuracy_score, precision_score, recall_score

accuracy=accuracy_score(y_test, y_pred)
print("accuracy: ", accuracy)
precision=precision_score(y_test, y_pred)
print("precision: ", precision)
recall=recall_score(y_test, y_pred)
print("recall: ", recall)


# In[ ]:





# ## XGBoost

# In[58]:


from sklearn.model_selection import KFold

xgb = XGBClassifier(random_state=88)

xgb.fit(X_resampled, y_resampled)

#K-Fold cross validation
cv = KFold(n_splits=5, shuffle=True, random_state=88) 
scores = cross_validate(xgb, X_test, y_test, cv=cv, scoring=['f1', 'accuracy'])

pd.DataFrame(scores)


# In[46]:


#High test accuracy across each fold of the test data means the model is
#likely not overfitiing


# In[47]:


y_pred = xgb.predict(X_test)

accuracy=accuracy_score(y_test, y_pred)
print("accuracy: {:.2f}".format(accuracy))
precision=precision_score(y_test, y_pred)
print("precision: {:.2f}".format(precision))
recall=recall_score(y_test, y_pred)
print("recall: {:.2f}".format(recall))


# In[48]:


from sklearn.metrics import confusion_matrix

#Plot confusion matrix
y_pred = xgb.predict(X_test)
conf_matrix = confusion_matrix(y_test, y_pred)
group_names = ['True Neg','False Pos','False Neg','True Pos']
group_percentages = ['{0:.2%}'.format(value) for value in
                     conf_matrix.flatten()/np.sum(conf_matrix)]
labels = [f'{v1}\n{v2}' for v1, v2 in
          zip(group_names,group_percentages)]
labels = np.asarray(labels).reshape(2,2)
sns.heatmap(conf_matrix, annot=labels, fmt='', cmap='Blues').set(title="Confusion Matrix")


# In[49]:


conf_matrix


# In[50]:


from sklearn.metrics import roc_curve, auc

y_prob = xgb.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

fig = go.Figure()

fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash')))

fig.update_layout(
    title='Receiver Operating Characteristic (ROC) Curve',
    xaxis=dict(title='False Positive Rate'),
    yaxis=dict(title='True Positive Rate (Recall)'),
    legend=dict(x=0.7, y=0.2),
)

fig.show()


# In[51]:


feature_importances = xgb.feature_importances_

# Create a DataFrame for plotting
importance_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})

# Sort the DataFrame by importance values
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Plot feature importance using Plotly
fig = ex.bar(importance_df, x='Importance', y='Feature', orientation='h',
             title='XGBoost Model Feature Importance',
             labels={'Importance': 'Feature Importance', 'Feature': 'Feature'})
fig.show()


# ## Hyperparameter Tunning

# In[52]:


from sklearn.model_selection import GridSearchCV


#parameters = {
    'max_depth': [1, 5, 20],
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2]
}


gs = GridSearchCV(xgb, parameters, cv=3, scoring=['f1', 'accuracy'], verbose=2, n_jobs=-1, refit='f1')

gs.fit(X_resampled, y_resampled)


# In[ ]:


print("Best hyperparameters: ", gs.best_params_)
print("Best model score: ", gs.best_score_)


# In[ ]:


xgb.predict_proba


# In[1]:


import shap
shap.initjs()

explainer = shap.KernelExplainer(dt.predict_proba, X_resampled, link='logit')
shap_values = explainer.shap_values(X_test, nsamples=100)

shap.force_plot(explainer.expected_value[0], shap_values[0][0, :], X_test.iloc[0, :], link='logit')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




