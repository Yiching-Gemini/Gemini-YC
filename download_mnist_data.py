#!/usr/bin/env python
# coding: utf-8

# ## CNN 卷積神經網路

# * 影像的特徵提取: 透過 Convolution 與 Max Pooling 提取影像特徵.
# * Fully connected Feedforward network: Flatten layers, hidden layers and output layers
# 
# http://puremonkey2010.blogspot.com/2017/07/toolkit-keras-mnist-cnn.html

# ## STEP1. 資料讀取與轉換 

# In[1]:



# TensorFlow and tf.keras
import sys
# Images, plots, display, and visualization
import matplotlib.pyplot as plt
import pandas as pd
import IPython
from six.moves import urllib


# ## STEP2. 將 Features 進行標準化與 Label 的 Onehot encoding 

# In[2]:


# download training data from s3

import boto #pip install boto

import boto.s3.connection
access_key = 'AB83M7JZ5B7A4FM81XGR'
secret_key = 'bKcA4tOaBLMjgA3J0WBFM3C0bLgGCcYeOYSOTOKl'
conn = boto.connect_s3(
        aws_access_key_id = access_key,
        aws_secret_access_key = secret_key,
        host = ' us-east-1.linodeobjects.com',
        #is_secure=False,               # uncomment if you are not using ssl
        calling_format = boto.s3.connection.OrdinaryCallingFormat(),
        )

for bucket in conn.get_all_buckets():
        print ("{name}\t{created}".format(
                name = bucket.name,
                created = bucket.creation_date,)
        )
key = bucket.get_key('dataset.gz')
key.get_contents_to_filename('/mnt/dataset.gz')

print('download completed')






