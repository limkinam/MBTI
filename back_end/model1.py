#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV, StratifiedKFold
from xgboost import XGBClassifier,plot_importance
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.feature_selection import SelectFromModel
from itertools import compress


from pylab import rcParams
import seaborn as sns
import matplotlib.pyplot as plt
#from scipy.misc import imread
from wordcloud import WordCloud, STOPWORDS

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

# Any results you write to the current directory are saved as output.

import pickle


# In[2]:


df = pd.read_csv('./dataset/mbti_1.csv')


# In[3]:


# In[4]:


dist = df['type'].value_counts()


# In[5]:


dist.index


# In[6]:


df['seperated_post'] = df['posts'].apply(lambda x: x.strip().split("|||"))
df['num_post'] = df['seperated_post'].apply(lambda x: len(x))



# In[7]:


num_post_df = df.groupby('type')['num_post'].apply(list).reset_index()
rcParams['figure.figsize'] = 10,5


# In[8]:


def count_youtube(posts):
    count = 0
    for p in posts:
        if 'youtube' in p:
            count += 1
    return count
        
df['youtube'] = df['seperated_post'].apply(count_youtube)


# In[9]:


df['id']=df.index


# In[10]:


expanded_df = pd.DataFrame(df['seperated_post'].tolist(), index=df['id']).stack().reset_index(level=1, drop=True).reset_index(name='idposts')


# In[11]:


def clean_text(text):
    result = re.sub(r'http[^\s]*', '',text)
    result = re.sub('[0-9]+','', result).lower()
    result = re.sub('@[a-z0-9]+', 'user', result)
    return re.sub('[%s]*' % string.punctuation, '',result)


# In[12]:


final_df = expanded_df.copy()
final_df['idposts'] = final_df['idposts'].apply(clean_text)


# In[13]:


cleaned_df = final_df.groupby('id')['idposts'].apply(list).reset_index()


# In[14]:


df['clean_post'] = cleaned_df['idposts'].apply(lambda x: ' '.join(x))


# In[15]:


vectorizer = CountVectorizer(stop_words = ['and','the','to','of',
                                           'infj','entp','intp','intj',
                                           'entj','enfj','infp','enfp',
                                           'isfp','istp','isfj','istj',
                                           'estp','esfp','estj','esfj',
                                           'infjs','entps','intps','intjs',
                                           'entjs','enfjs','infps','enfps',
                                           'isfps','istps','isfjs','istjs',
                                           'estps','esfps','estjs','esfjs'],
                            max_features=1500,
                            analyzer="word",
                            max_df=0.8,
                            min_df=0.1)


# In[16]:


corpus = df['clean_post'].values.reshape(1,-1).tolist()[0]
vectorizer.fit(corpus)
X_cnt = vectorizer.fit_transform(corpus)


# In[17]:


tfizer = TfidfTransformer()
tfizer.fit(X_cnt)
X = tfizer.fit_transform(X_cnt).toarray()


# In[18]:


all_words = vectorizer.get_feature_names()
n_words = len(all_words)
df['fav_world'] = df['type'].apply(lambda x: 1 if x[0] == 'E' else 0)
df['info'] = df['type'].apply(lambda x: 1 if x[1] == 'S' else 0)
df['decision'] = df['type'].apply(lambda x: 1 if x[2] == 'T' else 0)
df['structure'] = df['type'].apply(lambda x: 1 if x[3] == 'J' else 0)
df['r_fav_world'] = df['type'].apply(lambda x: 1 if x[0] == 'I' else 0)
df['r_info'] = df['type'].apply(lambda x: 1 if x[1] == 'N' else 0)
df['r_decision'] = df['type'].apply(lambda x: 1 if x[2] == 'F' else 0)
df['r_structure'] = df['type'].apply(lambda x: 1 if x[3] == 'P' else 0)


# In[19]:


X_df = pd.DataFrame.from_dict({w: X[:, i] for i, w in enumerate(all_words)})


# In[20]:


def sub_classifier(keyword):
    y_f = df[keyword].values
    X_f_train, X_f_test, y_f_train, y_f_test = train_test_split(X_df, y_f, stratify=y_f)
    f_classifier = XGBClassifier()
    print(">>> Train classifier ... ")
    f_classifier.fit(X_f_train, y_f_train, 
                     early_stopping_rounds = 10, 
                     eval_metric="logloss", 
                     eval_set=[(X_f_test, y_f_test)], verbose=False)
    print(">>> Finish training")
    print("%s:" % keyword, sum(y_f)/len(y_f))
    print("Accuracy %s" % keyword, accuracy_score(y_f_test, f_classifier.predict(X_f_test)))
    print("AUC %s" % keyword, roc_auc_score(y_f_test, f_classifier.predict_proba(X_f_test)[:,1]))
    return f_classifier


# In[21]:


# fav_classifier = sub_classifier('fav_world')


# In[22]:


#모델 불러오기
model1=pickle.load(open('./models/extro_Intro.pkl','rb'))# 외향/내향 모델
model2=pickle.load(open('./models/Sens_INtui.pkl','rb')) #현실/직관 모델
model3=pickle.load(open('./models/Think_Feel.pkl','rb')) #사고/감정 모델
model4=pickle.load(open('./models/Judg_Percei.pkl','rb')) #판단/인식 모델


# In[23]:


# info_classifier = sub_classifier('info')


# In[24]:


# decision_classifier = sub_classifier('decision')


# In[25]:


# str_classifier = sub_classifier('structure')


# In[26]:


rcParams['figure.figsize'] = 20, 10
# plt.subplots_adjust(wspace = 0.5)
# ax1 = plt.subplot(1, 4, 1)
# plt.pie([sum(df['fav_world']),
#          len(df['fav_world']) - sum(df['fav_world'])],
#         labels = ['Extrovert', 'Introvert'],
#         explode = (0, 0.1),
#        autopct='%1.1f%%')
#
# ax2 = plt.subplot(1, 4, 2)
# plt.pie([sum(df['info']),
#          len(df['info']) - sum(df['info'])],
#         labels = ['Sensing', 'Intuition'],
#         explode = (0, 0.1),
#        autopct='%1.1f%%')
#
# ax3 = plt.subplot(1, 4, 3)
# plt.pie([sum(df['decision']),
#          len(df['decision']) - sum(df['decision'])],
#         labels = ['Thinking', 'Feeling'],
#         explode = (0, 0.1),
#        autopct='%1.1f%%')
#
# ax4 = plt.subplot(1, 4, 4)
# plt.pie([sum(df['structure']),
#          len(df['structure']) - sum(df['structure'])],
#         labels = ['Judging', 'Perceiving'],
#         explode = (0, 0.1),
#        autopct='%1.1f%%')
#
# plt.show()


# In[27]:


# fav_classifier.get_xgb_params()


# In[28]:


params = {
        'min_child_weight': [1, 5],
        'gamma': [0.5, 1, 1.5, 2],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 5, 7]
        }


# In[29]:


# plot_importance(fav_classifier, max_num_features = 10)
# plt.title("Features associated with Extrovert")
# plt.show()
# plot_importance(info_classifier, max_num_features = 10)
# plt.title("Features associated with Extrovert")
# plt.show()
# plot_importance(decision_classifier, max_num_features = 10)
# plt.title("Features associated with Extrovert")
# plt.show()
# plot_importance(str_classifier, max_num_features = 10)
# plt.title("Features associated with Extrovert")
# plt.show()


# In[30]:


EI_imprt=model1.get_booster().get_score(importance_type='weight')
SN_imprt=model2.get_booster().get_score(importance_type='weight')
TF_imprt=model3.get_booster().get_score(importance_type='weight')
JP_imprt=model4.get_booster().get_score(importance_type='weight')


# In[31]:


EI_imprt_tuple = sorted(EI_imprt.items(),reverse=True,key=lambda item:item[1])
SN_imprt_tuple = sorted(SN_imprt.items(),reverse=True,key=lambda item:item[1])
TF_imprt_tuple = sorted(TF_imprt.items(),reverse=True,key=lambda item:item[1])
JP_imprt_tuple = sorted(JP_imprt.items(),reverse=True,key=lambda item:item[1])


# In[32]:


# #모델 저장하기
# import pickle
# model1 = 'extro_Intro.pkl' #외향/내향 모델
# model2 = 'Sens_INtui.pkl' #현실/직관 모델
# model3 = 'Think_Feel.pkl' #사고/감정 모델
# model4 = 'Judg_Percei.pkl' #판단/인식 모델
# #만들기
# pickle.dump(fav_classifier,open(model1,'wb'))
# pickle.dump(info_classifier,open(model2,'wb'))
# pickle.dump(decision_classifier,open(model3,'wb'))
# pickle.dump(str_classifier,open(model4,'wb'))


# In[33]:


def list_mk(self):
    list = []
    for i in self:
        list.append(i[0])
    return list


# In[34]:


EI_keyword_list = ['fun',
 'yourself',
 'friends',
 'guys',
 'family',
 'personality',
 'myself',
 'could',
 'here',
 'thread']
SN_keyword_list = ['world',
 'work',
 'fun',
 'were',
 'person',
 'looking',
 'could',
 'human',
 'maybe',
 'wont']
TF_keyword_list = ['love',
 'why',
 'thank',
 'relate',
 'point',
 'off',
 'feeling',
 'values',
 'use',
 'help']
JP_keyword_list = ['love',
 'able',
 'will',
 'music',
 'trust',
 'theres',
 'hello',
 'fun',
 'best',
 'control']


# In[ ]:


from flask import Flask, request
from flask_cors import CORS
import os
import sys
import urllib.request
import json


client_id = "DDViAHj1CzTI3Xhuvbxg"
client_secret = "l9wT2TKdy0"
app = Flask(__name__)

CORS(app, resources={r'/*': {'origins': '*'}})

@app.route('/trans',methods=['POST'])
def trans():
    post_data = request.data
    encText = urllib.parse.quote(post_data)
    data = "source=ko&target=en&text=" + encText
    url = "https://openapi.naver.com/v1/papago/n2mt"
    req = urllib.request.Request(url)
    req.add_header("X-Naver-Client-Id",client_id)
    req.add_header("X-Naver-Client-Secret",client_secret)
    res = urllib.request.urlopen(req, data=data.encode("utf-8"))
    rescode = res.getcode()
    if(rescode==200):
        response_body = res.read()
        print(type(response_body))
        string = response_body.decode('utf-8')
        dict = json.loads(string)
        final= dict['message']['result']['translatedText']
        print(final)
        #예측
        final_test = tfizer.transform(vectorizer.transform([final])).toarray()
        test_point = pd.DataFrame.from_dict({w: final_test[:, i] for i, w in enumerate(all_words)})
        #저장한모델 불러오기
        result1 = model1.predict_proba(test_point)[0]
        result2 = model2.predict_proba(test_point)[0]
        result3 = model3.predict_proba(test_point)[0]
        result4 = model4.predict_proba(test_point)[0]

        result1[0]=round(result1[0]*10000)/81*50
        result1[1] = round(result1[1]*10000)/18*50
        result2[0] = round(result2[0] * 10000)/81*50
        result2[1] = round(result2[1] * 10000)/19*50
        result3[0] = round(result3[0] * 10000)/38*50
        result3[1] = round(result3[1] * 10000)/62*50
        result4[0] = round(result4[0] * 10000)/55*50
        result4[1] = round(result4[1] * 10000)/45*50


        if(result1[0] > result1[1]):
            rst="I"
        else:
            rst="E"
        if(result2[0] > result2[1]):
            rst+="N"
        else:
            rst+="S"
        if(result3[0] > result3[1]):
            rst+="F"
        else:
            rst+="T"
        if(result4[0] > result4[1]):
            rst+="J"
        else:
            rst+="P"
        return {'EI':EI_keyword_list,'SN':SN_keyword_list,'TF':TF_keyword_list,'JP':JP_keyword_list,'MBTI':rst,'I':str(round(result1[0])),
                'E':str(result1[1]),'N':str(result2[0]),'S':str(result2[1]),'F':str(result3[0]),'T':str(result3[1]),'J':str(result4[0]),'P':str(result4[1])}
    else:
        return "Error Code:" + rescode
     
if __name__ == '__main__':
  app.run(host='0.0.0.0')


# In[ ]:




