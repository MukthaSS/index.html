#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


a=pd.read_csv('seattle-weather[1].csv')
a.head()


# In[3]:


a.isnull().sum()


# In[4]:


sns.heatmap(a.corr(),annot=True)


# In[5]:


a.drop(['date'],axis=1,inplace=True)
a


# In[30]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
a['weather']=le.fit_transform(a['weather'])
a


# In[8]:


a['weather'].unique()


# In[9]:


x=a.iloc[:,:-1]
x


# In[10]:


y=a.iloc[:,-1]
y


# In[11]:


from sklearn.model_selection import train_test_split


# In[17]:


xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=60)
xtrain.shape,xtest.shape,ytrain.shape,ytest.shape


# In[18]:


from sklearn.svm import SVC
svm=SVC(kernel='linear')
svm.fit(xtrain,ytrain)


# In[19]:


ypred=svm.predict(xtest)
ypred


# In[20]:


ytest


# In[21]:


svm.score(xtest,ytest)


# In[23]:


import pickle as pkl


# In[24]:


filename='milknew[1].sav'
pkl.dump(svm,open(filename,'bw'))


# In[25]:


loaded_model=pkl.load(open(filename,'br'))
loaded_model


# In[26]:


new_data = np.array([0.0,12.8,5.0,4.7]).reshape(1,-1)
svm.predict(new_data)


# In[27]:


import gradio as gd


# In[28]:


print(a.columns)


# In[31]:


def weather(precipitation,temp_max,temp_min,wind):
    x=np.array([precipitation,temp_max,temp_min,wind])
    prediction=svm.predict(x.reshape(1,-1))
    if prediction==0:
        return 'drizzle'
    elif prediction==1:
        return 'fog'
    elif prediction==2:
        return 'rain'
    elif prediction==3:
        return 'snow'
    else:
        return 'sun'


# In[32]:


app=gd.Interface(fn=weather,
                inputs=[gd.inputs.Number(label='Enter the Precipitation'),
                        gd.inputs.Number(label='Enter the Temp_max'),
                        gd.inputs.Number(label='Enter the Temp_min'),
                        gd.inputs.Number(label='Enter the wind')],
                outputs=gd.outputs.Label(),
                title='Developing an Ml model for predicting the weather')


# In[33]:


app.launch()


# In[ ]:




