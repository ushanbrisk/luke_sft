#!/usr/bin/env python
# coding: utf-8

# # 仅仅是把数据转为huggingface能够认出的格式
# 
# 
# 
#  # pip install datasets==2.18.0
#  
#  # modelscope需要的datasets不能超过2.19.0
#  
#  

# In[4]:


from modelscope.msdatasets import MsDataset

ds = MsDataset.load('krisfu/delicate_medical_r1_data', subset_name='default', split='train')


# In[5]:


ds


# In[6]:


ds[0]


# In[12]:


import datasets
total_ds = datasets.DatasetDict({"train":ds})


# In[13]:


total_ds


# In[14]:


total_ds.push_to_hub("lukedai/delicate_medical_r1_data")


# In[15]:


from datasets import load_dataset
dataset2 = load_dataset("lukedai/delicate_medical_r1_data")


