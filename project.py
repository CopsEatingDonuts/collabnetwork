#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import networkx as nx
import nxviz as nv
import matplotlib.pyplot as plt 
import ast
import json
import os


# # Q1 Analyis of collab network (All)

# In[2]:


collab_network = pd.read_csv('./output/collab_network_csv.csv', index_col='Unnamed: 0')


# In[3]:


collab_network


# In[4]:


collab_network['coauthors_list'] = collab_network['coauthors_list'].apply(lambda x: ast.literal_eval(x))


# In[5]:


G = nx.Graph()
G.add_nodes_from(collab_network['author_pid'])


# In[6]:


for i in range(0, len(collab_network)):
    
    author = collab_network.iloc[i]['author_pid']
    coauthors = collab_network.iloc[i]['coauthors_list']
    
    for coauthor in coauthors:
        if coauthor in G.nodes():
            G.add_edge(author, coauthor)


# # Graph visualisation

# In[7]:


plt.figure(figsize=(40, 40))
nx.draw(G, with_labels=True)
plt.savefig('./Results/q1_graph_visualisation.png')


# In[8]:


plt.figure(figsize=(100, 100))
ax = nv.circos(G)
plt.savefig('./Results/q1_circos.png')


# # Degree Centrality and Distribution

# In[9]:


# Compute the degree distribution
degrees = [len(list(G.neighbors(n))) for n in G.nodes()]


# In[10]:


# Compute the degree centrality
deg_cent = nx.degree_centrality(G)


# In[11]:


# Plot histogram of the degree centrality distribution of the graph
plt.figure(figsize=(20, 20))
plt.xlabel('Degree_centrality')
plt.ylabel('Number_of_nodes')
plt.hist(list(deg_cent.values()), bins=50)
plt.savefig('./Results/q1_deg_cent.png')


# In[12]:


# Plot a histogram of the degree distribution of the graph
plt.figure(figsize=(20, 20))
plt.xlabel('Node_degree')
plt.ylabel('Number_of_nodes')
plt.hist(degrees, bins=50)
plt.savefig('./Results/q1_deg_dist.png')


# In[13]:


# Plot a scatter plot of the centrality distribution and the degree distribution
plt.figure(figsize=(20, 20))
plt.xlabel('Node_degree')
plt.ylabel('Degree_centrality')
plt.scatter(degrees, list(deg_cent.values()))
plt.savefig('./Results/q1_cent_and_deg_corr.png')


# # Betweenness Centrality

# In[14]:


bet_cent = nx.betweenness_centrality(G)


# In[15]:


# Plot a scatter plot of the betweeness centrality and the degree centrality
plt.figure(figsize=(20, 20))
plt.scatter(list(bet_cent.values()), list(deg_cent.values()))
plt.xlabel('Betweeness_centrality')
plt.ylabel('Degree_centrality')
plt.savefig('./Results/q1_bet_and_deg_cent.png')


# # Find the node with the highest Degree Centrality

# In[16]:


def find_nodes_with_highest_deg_cent(graph):

    # Compute the degree centrality of G: deg_cent
    deg_cent = nx.degree_centrality(graph)

    # Compute the maximum degree centrality: max_dc
    max_dc = max(list(deg_cent.values()))

    nodes = set()

    # Iterate over the degree centrality dictionary
    for k, v in deg_cent.items():

        # Check if the current value has the maximum degree centrality
        if v == max_dc:

            # Add the current node to the set of nodes
            nodes.add(k)

    return nodes, max_dc

# Find the node(s) that has the highest degree centrality in T: top_dc
top_dc_node, top_dc_value = find_nodes_with_highest_deg_cent(G)

print(f'Author with highest degree centrality at {top_dc_value}: ', top_dc_node)

for node in top_dc_node:
    assert nx.degree_centrality(G)[node] == max(nx.degree_centrality(G).values())


# # Find the node with the highest Betweeness Centrality

# In[17]:


# Define find_node_with_highest_bet_cent()
def find_node_with_highest_bet_cent(graph):

    # Compute betweenness centrality: bet_cent
    bet_cent = nx.betweenness_centrality(graph)

    # Compute maximum betweenness centrality: max_bc
    max_bc = max(list(bet_cent.values()))

    nodes = set()

    # Iterate over the betweenness centrality dictionary
    for k, v in bet_cent.items():

        # Check if the current value has the maximum betweenness centrality
        if v == max_bc:

            # Add the current node to the set of nodes
            nodes.add(k)

    return nodes, max_bc

# Use that function to find the node(s) that has the highest betweenness centrality in the network: top_bc
top_bc_node, top_bc_value = find_node_with_highest_bet_cent(G)
print(f'Author with highest betweeness centrality at {top_bc_value}: ', top_bc_node)

for node in top_bc_node:
    assert nx.betweenness_centrality(G)[node] == max(nx.betweenness_centrality(G).values())


# # Find the node with the highest Closeness Centrality
# 

# In[18]:


def find_node_with_highest_close_cent(graph):

    # Compute betweenness centrality: bet_cent
    close_cent = nx.closeness_centrality(graph)

    # Compute maximum betweenness centrality: max_bc
    max_cc = max(list(close_cent.values()))

    nodes = set()

    # Iterate over the betweenness centrality dictionary
    for k, v in close_cent.items():

        # Check if the current value has the maximum betweenness centrality
        if v == max_cc:

            # Add the current node to the set of nodes
            nodes.add(k)

    return nodes, max_cc

top_cc_node, top_cc_value = find_node_with_highest_close_cent(G)
print(f'Author with highest closeness centrality at {top_cc_value}: ', top_cc_node)


# # Find the maximal clique

# In[19]:


def maximal_cliques(graph):
    mcs = []
    giant_component_size = max([len(i) for i in nx.find_cliques(G)])
    
    for clique in nx.find_cliques(graph):
        if len(clique) == giant_component_size:
            mcs.append(clique)
            
    return giant_component_size, mcs
    

giant_size, giant_cliques = maximal_cliques(G)


print('Size of giant component: ', giant_size) 
print('Cliques of this size: ', giant_cliques)


# In[20]:


#Draw the subgraphs
i=0
for clique in giant_cliques:
    plt.figure(figsize=(20, 20))
    nx.draw(G.subgraph(giant_cliques[0]), with_labels=True)
    plt.savefig(f'./Results/q1_giant_clique_{i}.png')
    i+=1


# # Find the giant component

# In[21]:


Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
G0 = G.subgraph(Gcc[0])
nx.draw(G.subgraph(G0), with_labels=True)


# # Statistics

# In[22]:


statistics = {
    
    'average_node_degree' : (2*len(G.edges()))/len(G.nodes()),
    'max_node_degree' : [list(G)[np.argmax(degrees)], max(degrees)],
    'degree_assortativity' : nx.degree_assortativity_coefficient(G),
    'average_clustering_coeff' : nx.average_clustering(G),
    'giant_clique_size' : giant_size,
    'giant_component_size' : len(G0),
    'giant_component_clustering_coeff' : nx.average_clustering(G0),
    'graph_density' : nx.density(G),
    'node_highest_degree_centrality' : [top_dc_node, top_dc_value],
    'node_highest_betweeness_centrality' : [top_bc_node, top_bc_value],
    'node_highest_closeness_centrality' : [top_cc_node, top_cc_value],

}

with open('./Results/q1_statistics.txt', 'w') as f:
    print(statistics, file=f)


# # Q2 analysis of collab network (Yearly granularity)

# In[23]:


yearly_collab_network = pd.read_csv('./output/year_granularity_df.csv', index_col='Unnamed: 0')
yearly_collab_network.shape


# In[24]:


def literal_eval_special(x):
    try:
        return ast.literal_eval(x)
    
    except:
        return None


# In[ ]:


for i in range(0, len(yearly_collab_network)):
    year = yearly_collab_network.iloc[i].name
    collab_network = pd.DataFrame(yearly_collab_network.iloc[i]).reset_index(names = ['author_pid']).rename({year: 'coauthors_list'}, axis='columns')

    collab_network['coauthors_list'] = collab_network['coauthors_list'].apply(lambda x: literal_eval_special(x))
    
    newpath = f'./Results/{year}' 
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    #run the bunch of analysis
    G = nx.Graph()
    G.add_nodes_from(collab_network['author_pid'])
    for i in range(0, len(collab_network)):
    
        author = collab_network.iloc[i]['author_pid']
        
        if collab_network.iloc[i]['coauthors_list'] == None:
            continue
        
        else:
            coauthors = collab_network.iloc[i]['coauthors_list'][0]
    
        for coauthor in coauthors:
            if coauthor in G.nodes():
                G.add_edge(author, coauthor)
    
    plt.figure(figsize=(40, 40))
    nx.draw(G, with_labels=True)
    plt.savefig(f'./Results/{year}/q2_{year}_graph_visualisation.png')
    
    degrees = [len(list(G.neighbors(n))) for n in G.nodes()]
    deg_cent = nx.degree_centrality(G)
    
    plt.figure(figsize=(20, 20))
    plt.xlabel('Degree_centrality')
    plt.ylabel('Number_of_nodes')
    plt.hist(list(deg_cent.values()), bins=50)
    plt.savefig(f'./Results/{year}/q2_{year}_deg_cent.png')
    
    plt.figure(figsize=(20, 20))
    plt.xlabel('Node_degree')
    plt.ylabel('Number_of_nodes')
    plt.hist(degrees, bins=50)
    plt.savefig(f'./Results/{year}/q2_{year}_deg_dist.png')
    
    plt.figure(figsize=(20, 20))
    plt.xlabel('Node_degree')
    plt.ylabel('Degree_centrality')
    plt.scatter(degrees, list(deg_cent.values()))
    plt.savefig(f'./Results/{year}/q2_{year}_cent_and_deg_corr.png')
    
    bet_cent = nx.betweenness_centrality(G)
    plt.figure(figsize=(20, 20))
    plt.scatter(list(bet_cent.values()), list(deg_cent.values()))
    plt.xlabel('Betweeness_centrality')
    plt.ylabel('Degree_centrality')
    plt.savefig(f'./Results/{year}/q2_{year}_bet_and_deg_cent.png')
    
    top_dc_node, top_dc_value = find_nodes_with_highest_deg_cent(G)
    print(f'Author for year {year} with highest degree centrality at {top_dc_value}: ', top_dc_node)
    
    top_bc_node, top_bc_value = find_node_with_highest_bet_cent(G)
    print(f'Author for year {year} with highest betweeness centrality at {top_bc_value}: ', top_bc_node)
    
    top_cc_node, top_cc_value = find_node_with_highest_close_cent(G)
    print(f'Author for year {year} with highest closeness centrality at {top_cc_value}: ', top_cc_node)
    
    giant_size, giant_cliques = maximal_cliques(G)
    print(f'Size of giant clique in {year}: {giant_size}') 
    print(f'Cliques of this size in {year}: {giant_cliques}')
    
    i=0
    for clique in giant_cliques:
        plt.figure(figsize=(20, 20))
        nx.draw(G.subgraph(giant_cliques[0]), with_labels=True)
        plt.savefig(f'./Results/{year}/q2_{year}_giant_clique_{i}.png')
        i+=1
        
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(Gcc[0])
    nx.draw(G.subgraph(G0), with_labels=True)
    
    statistics = {
    
        'average_node_degree' : (2*len(G.edges()))/len(G.nodes()),
        'max_node_degree' : [list(G)[np.argmax(degrees)], max(degrees)],
        'degree_assortativity' : nx.degree_assortativity_coefficient(G),
        'average_clustering_coeff' : nx.average_clustering(G),
        'giant_component_size' : giant_size,
        'giant_component_clustering_coeff' : nx.average_clustering(G0),
        'giant_component_size' : len(G0),
        'graph_density' : nx.density(G),
        'node_highest_degree_centrality' : [top_dc_node, top_dc_value],
        'node_highest_betweeness_centrality' : [top_bc_node, top_bc_value],
        'node_highest_closeness_centrality' : [top_cc_node, top_cc_value],

    }

    with open(f'./Results/{year}/q2_{year}_statistics.txt', 'w') as f:
        print(statistics, file=f)
        
    print(year, ' Completed')


# # Q4 code

# In[ ]:


from tqdm.notebook import tqdm

# Function to get author year series
def get_author_year_series(root):
    if root is None:
        return None
  
    author = root.attrib['pid']
    year_coauthor_dict = dict()
  
    for i in range(len(root)):
        if root[i].tag == 'r':  # only look at article entries
            publish_work = root[i][0].attrib['key']
            publish_year = root[i][0].attrib['mdate'][:4]  # year
            current_year_coauthor_list = []
    
            for j in range(len(root[i][0])):
                if root[i][0][j].tag == 'author':
                    current_year_coauthor_list.append(root[i][0][j].attrib['pid'])
      
            if publish_year not in year_coauthor_dict:
                year_coauthor_dict[publish_year] = current_year_coauthor_list
            else:
                year_coauthor_dict[publish_year] += current_year_coauthor_list
  
    for year in year_coauthor_dict:
        year_coauthor_dict[year] = [year_coauthor_dict[year]]
    
    year_coauthor_series = pd.Series(year_coauthor_dict, name=author)
    return year_coauthor_series

# Function to get author root
def get_author_root(url):
    try:
        r = requests.get(url[:-4] + 'xml').text
        root = ET.fromstring(r)
    except:
        return None
    
    return root

scientists = pd.read_excel('DataScientists.xls')
scientists.drop_duplicates(subset='dblp', inplace=True, ignore_index=True)

# Step 1: Identify High-Degree Nodes
degree_threshold = 10  # Example threshold for high-degree nodes
high_degree_nodes = set()  # Set to store high-degree nodes

# Placeholder for author information for diversity checks
author_info = {
    'country': pd.Series(np.random.choice(['US', 'UK', 'China', 'India', 'Germany'], len(scientists))),
    'expertise': pd.Series(np.random.choice(['AI', 'ML', 'NLP', 'CV', 'Robotics'], len(scientists))),
    'institution': pd.Series(np.random.choice(['MIT', 'Stanford', 'CMU', 'Berkeley', 'Oxford'], len(scientists)))
}

collab_network_list = []
problem_list = []
join_series_list = []

for i in tqdm(range(len(scientists))):
    url = scientists.iloc[i]['dblp']
    root = get_author_root(url)
    
    if root is None:
        #problem_list.append([scientists.iloc[i]['pid'], url])
        #problem_list.append([root.attrib['pid'], url])
        problem_list.append([scientists.iloc[i]['dblp'], url])  # Use 'dblp' instead of 'pid'
        continue
  
    author_pid = root.attrib['pid']
    author_name = root.attrib['name']
    coauthors = set()
    
    for j in range(len(root)):
        if root[j].tag == 'r':  # only look at article entries
            for k in range(len(root[j][0])):
                if root[j][0][k].tag == 'author': 
                    coauthors.add(root[j][0][k].attrib['pid'])
    
    # Step 1: Identify High-Degree Nodes
    degree = len(coauthors)
    if degree > degree_threshold:
        high_degree_nodes.add(author_pid)
    
    # Step 2: Remove Nodes Connected to High-Degree Nodes
    filtered_coauthors = set()
    for coauthor in coauthors:
        if coauthor not in high_degree_nodes or not is_diverse_removal(author_pid, coauthor, collab_network_list):
            filtered_coauthors.add(coauthor)
    
    # Step 3: Ensure Network Properties (already partially achieved in Step 2)
    
    # Step 4: Enforce Collaboration Cutoff (kmax)
    kmax = 10  # Example collaboration cutoff
    if degree > kmax:
        filtered_coauthors = set(list(filtered_coauthors)[:kmax])  # Truncate coauthors list
  
    collab_network_list.append([author_name, author_pid, filtered_coauthors])

    join_series = get_author_year_series(root)
    join_series_list.append(join_series)

collab_network_csv = pd.DataFrame(collab_network_list, columns=['author_name', 'author_pid', 'coauthors_list'])
problem_list_csv = pd.DataFrame(problem_list, columns=['problem_pid', 'url'])

collab_network_csv.to_csv('./Transformed/q4_transformed_collab_network_csv.csv')

print("Transformed network generated!")


# In[ ]:




