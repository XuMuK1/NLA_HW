
# coding: utf-8

# ## Preparing the data.
# 
# You're given convenience functions `get_movielens_data` and `split_data`. 
# * Use these functions to download the data, put it into the memory and split it into the training and testing set with 80/20 rule (80% of the data goes into training set and the rest - into test set).
# 
# *Note*: downloading the dataset may take a couple of minutes. If you already have a copy of MovieLens 10M data on your computer you may force program to use it by specifying *local_file* argument in the `get_movielens_data` function. The *local_file* argument should be a full path to the zip file.

# In[ ]:

# Libraries and provided functions
import pandas as pd
import zipfile
import wget
from io import StringIO 
import numpy as np
import scipy as sp
from scipy import sparse
import scipy.sparse.linalg
from tqdm import tqdm # Very useful library to see progress bar during range iterations: just type `for i in tqdm(range(10)):`
from matplotlib import pyplot as plt
get_ipython().magic(u'matplotlib inline')

from collections import namedtuple
import sys

def get_movielens_data(local_file=None):
    '''Downloads movielens data, normalizes users and movies ids,
    returns data in sparse CSR format.
    '''
    if not local_file:
        print('Downloading data...')
        zip_file_url = 'http://files.grouplens.org/datasets/movielens/ml-10m.zip'
        zip_contents = wget.download(zip_file_url)
        print('Done.')
    else:
        zip_contents = local_file
    
    print('Loading data into memory...')
    with zipfile.ZipFile(zip_contents) as zfile:
        zdata = zfile.read('ml-10M100K/ratings.dat').decode()
        delimiter = ';'
        zdata = zdata.replace('::', delimiter) # makes data compatible with pandas c-engine
        ml_data = pd.read_csv(StringIO(zdata), sep=delimiter, header=None, engine='c',
                                  names=['userid', 'movieid', 'rating', 'timestamp'],
                                  usecols=['userid', 'movieid', 'rating'])
    
    # normalize indices to avoid gaps
    ml_data['movieid'] = ml_data.groupby('movieid', sort=False).grouper.group_info[0]
    ml_data['userid'] = ml_data.groupby('userid', sort=False).grouper.group_info[0]
    
    # build sparse user-movie matrix
    data_shape = ml_data[['userid', 'movieid']].max() + 1
    data_matrix = sp.sparse.csr_matrix((ml_data['rating'],
                                       (ml_data['userid'], ml_data['movieid'])),
                                        shape=data_shape, dtype=np.float64)
    
    print('Done.')
    return data_matrix

def split_data(data, test_ratio=0.2):
    '''Randomly splits data into training and testing datasets. Default ratio is 80%/20%.
    Returns datasets in namedtuple format for convenience. Usage:
    
    train_data, test_data = split_data(data_matrix)
    
    or
    
    movielens_data = split_data(data_matrix)
    
    and later in code: 
    
    do smth with movielens_data.train 
    do smth with movielens_data.test
    '''
    
    num_users = data.shape[0]
    idx = np.zeros((num_users,), dtype=bool)
    sel = np.random.choice(num_users, int(test_ratio*num_users), replace=False)
    np.put(idx, sel, True)
    
    Movielens_data = namedtuple('MovieLens10M', ['train', 'test'])
    movielens_data = Movielens_data(train=data[~idx, :], test=data[idx, :])
    return movielens_data

