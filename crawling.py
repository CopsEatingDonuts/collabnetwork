#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from tqdm.notebook import tqdm
import time


# In[2]:


scientists = pd.read_excel('./Input/DataScientists.xls')


# In[3]:


scientists.head()


# In[4]:


scientists.shape


# In[5]:


scientists.drop_duplicates(subset='dblp', inplace=True, ignore_index=True) #remove duplicates by url


# In[6]:


scientists.shape


# # Crawling for Q1 & Q2 (All collaborations, collaborations in yearly granularity)

# In[7]:


def get_author_year_series(root):
    
    if root == None:
        return None
    
    author = root.attrib['pid']

    year_coauthor_dict = dict()
    for i in range(0, len(root)): 
    
        if root[i].tag == 'r': #only look at article entries
        
            publish_work = root[i][0].attrib['key']
            publish_year = root[i][0].attrib['mdate'][:4] #year
        
            current_year_coauthor_list = []
        
            for j in range(0, len(root[i][0])):
                if root[i][0][j].tag == 'author': 
                    current_year_coauthor_list.append(root[i][0][j].attrib['pid'])
        
            if publish_year not in year_coauthor_dict:
                year_coauthor_dict[publish_year] = current_year_coauthor_list
            
            else: year_coauthor_dict[publish_year] = year_coauthor_dict[publish_year] + current_year_coauthor_list
        
    for year in year_coauthor_dict:
        year_coauthor_dict[year] = [year_coauthor_dict[year]]
        
    year_coauthor_series = pd.Series(year_coauthor_dict, name = author)
    return year_coauthor_series

def get_author_root(url):
    try:
        r = requests.get(url[:-4] + 'xml').text
        root = ET.fromstring(r)
    except:
        return None
    
    return root


# In[9]:


collab_network_list = []
problem_list = []
join_series_list = []

for i in range(0, len(scientists)):
    url = scientists.iloc[i]['dblp']  
    institute = scientists.iloc[i]['institution']
    country = scientists.iloc[i]['country']
    r = requests.get(url[:-4] + 'xml').text
    
    try:
        root = ET.fromstring(r)
    
    except:
        problem_list.append([root.attrib['pid'], url]) #track problematic entries
        continue
    
    author_pid = root.attrib['pid'] #figure out the pid of the author
    author_name = root.attrib['name']
    
    coauthors = []
    
    for j in range(0, len(root)): 
        if root[j].tag == 'r': #only look at article entries
            #print(root[i][0].attrib['key']) #article name
            for k in range(0, len(root[j][0])):
                if root[j][0][k].tag == 'author': #coauthors
                    #print(root[i][0][j].tag, "{0:<30}".format(root[i][0][j].text), 'pid: ' + root[i][0][j].attrib['pid'])
                    coauthors.append(root[j][0][k].attrib['pid'])
    
    coauthors = set(coauthors) #remove duplicates
    collab_network_list.append([author_name, institute, country, author_pid, coauthors])

    join_series = get_author_year_series(root)
    join_series_list.append(join_series)
    
    if i%100 == 0: print('progress: ', i)
    
collab_network_csv = pd.DataFrame(collab_network_list, columns = ['author_name', 'institute', 'country', 'author_pid', 'coauthors_list'])
problem_list_csv = pd.DataFrame(problem_list, columns = ['problem_pid', 'url'])


# In[10]:


print(len(collab_network_list), len(problem_list))
print(len(join_series_list))


# In[11]:


newpath = f"./output"
if not os.path.exists(newpath):
    os.makedirs(newpath)

collab_network_csv.to_csv('output/collab_network_csv.csv')
problem_list_csv.to_csv('output/problem_list_csv.csv')


# In[12]:


join_series_list[:10]


# In[13]:


year_granularity_df = pd.concat(join_series_list, axis=1, join='outer')


# In[14]:


year_granularity_df.shape


# In[15]:


year_granularity_df.to_csv('output/year_granularity_df.csv')


# In[ ]:




