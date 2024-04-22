#!/usr/bin/env python
# coding: utf-8

# In[1]:

import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET

import networkx as nx
import nxviz as nv
import matplotlib.pyplot as plt 
import ast
import os
import random

scientists = pd.read_excel('./Input/DataScientists.xls')

scientists.drop_duplicates(subset='dblp', inplace=True, ignore_index=True) #remove duplicates by url

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

print('Initializing crawling...')

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
    
    if i%100 == 0: print('row progress: ', i)
    
collab_network_csv = pd.DataFrame(collab_network_list, columns = ['author_name', 'institute', 'country', 'author_pid', 'coauthors_list'])
problem_list_csv = pd.DataFrame(problem_list, columns = ['problem_pid', 'url'])


newpath = f"./output"
if not os.path.exists(newpath):
    os.makedirs(newpath)

collab_network_csv.to_csv('output/collab_network_csv.csv')
problem_list_csv.to_csv('output/problem_list_csv.csv')




# In[13]:


year_granularity_df = pd.concat(join_series_list, axis=1, join='outer')


# In[15]:


year_granularity_df.to_csv('output/year_granularity_df.csv')

print('Crawling complete!')

# # Q1 Analyis of collab network (All)
print('Running analysis for Question 1...')

# In[2]:


collab_network = pd.read_csv("./output/collab_network_csv.csv", index_col="Unnamed: 0")


# In[3]:


collab_network["coauthors_list"] = collab_network["coauthors_list"].apply(lambda x: ast.literal_eval(x))


# In[4]:


G = nx.Graph()
G.add_nodes_from(collab_network["author_pid"])


# In[5]:


for i in range(0, len(collab_network)):
    
    author = collab_network.iloc[i]["author_pid"]
    coauthors = collab_network.iloc[i]["coauthors_list"]
    
    for coauthor in coauthors:
        if coauthor in G.nodes():
            G.add_edge(author, coauthor)


# In[6]:


newpath = f"./Results"
if not os.path.exists(newpath):
    os.makedirs(newpath)


# # Graph visualisation

# In[7]:


plt.figure(figsize=(40, 40))
nx.draw(G, with_labels=True)
plt.savefig("./Results/q1_graph_visualisation.png")


# # Degree Centrality and Distribution

# In[8]:


# Compute the degree distribution
degrees = [len(list(G.neighbors(n))) for n in G.nodes()]


# In[9]:


# Compute the degree centrality
deg_cent = nx.degree_centrality(G)


# In[10]:


# Plot histogram of the degree centrality distribution of the graph
plt.figure(figsize=(20, 20))
plt.xlabel("Degree_centrality")
plt.ylabel("Number_of_nodes")
plt.hist(list(deg_cent.values()), bins=50)
plt.savefig("./Results/q1_deg_cent.png")


# In[11]:


# Plot a histogram of the degree distribution of the graph
plt.figure(figsize=(20, 20))
plt.xlabel("Node_degree")
plt.ylabel("Number_of_nodes")
plt.hist(degrees, bins=50)
plt.savefig("./Results/q1_deg_dist.png")


# In[12]:


# Plot a scatter plot of the centrality distribution and the degree distribution
plt.figure(figsize=(20, 20))
plt.xlabel("Node_degree")
plt.ylabel("Degree_centrality")
plt.scatter(degrees, list(deg_cent.values()))
plt.savefig("./Results/q1_cent_and_deg_corr.png")


# # Betweenness Centrality

# In[13]:


bet_cent = nx.betweenness_centrality(G)


# In[14]:


# Plot a scatter plot of the betweeness centrality and the degree centrality
plt.figure(figsize=(20, 20))
plt.scatter(list(bet_cent.values()), list(deg_cent.values()))
plt.xlabel("Betweeness_centrality")
plt.ylabel("Degree_centrality")
plt.savefig("./Results/q1_bet_and_deg_cent.png")


# # Find the node with the highest Degree Centrality

# In[15]:


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

#print(f"Author with highest degree centrality at {top_dc_value}: ", top_dc_node)

for node in top_dc_node:
    assert nx.degree_centrality(G)[node] == max(nx.degree_centrality(G).values())


# # Find the node with the highest Betweeness Centrality

# In[16]:


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
#print(f"Author with highest betweeness centrality at {top_bc_value}: ", top_bc_node)

for node in top_bc_node:
    assert nx.betweenness_centrality(G)[node] == max(nx.betweenness_centrality(G).values())


# # Find the node with the highest Closeness Centrality
# 

# In[17]:


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
#print(f"Author with highest closeness centrality at {top_cc_value}: ", top_cc_node)


# # Find the maximal clique

# In[18]:


def maximal_cliques(graph):
    mcs = []
    giant_component_size = max([len(i) for i in nx.find_cliques(G)])
    
    for clique in nx.find_cliques(graph):
        if len(clique) == giant_component_size:
            mcs.append(clique)
            
    return giant_component_size, mcs
    

giant_size, giant_cliques = maximal_cliques(G)


#print("Size of giant component: ", giant_size) 
#print("Cliques of this size: ", giant_cliques)


# In[19]:


#Draw the subgraphs
i=0
for clique in giant_cliques:
    plt.figure(figsize=(20, 20))
    nx.draw(G.subgraph(giant_cliques[0]), with_labels=True)
    plt.savefig(f"./Results/q1_giant_clique_{i}.png")
    i+=1


# # Find the giant component

# In[20]:


Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
G0 = G.subgraph(Gcc[0])
plt.figure(figsize=(100, 100))
nx.draw(G.subgraph(G0), with_labels=True)
plt.savefig(f"./Results/q1_giant_component_{0}.png")


# # Statistics

# In[21]:


statistics = {
    
    "number of nodes" : len(G.nodes()),
    "number of edges" : len(G.edges()),
    "average_node_degree" : (2*len(G.edges()))/len(G.nodes()),
    "max_node_degree" : [list(G)[np.argmax(degrees)], max(degrees)],
    "degree_assortativity" : nx.degree_assortativity_coefficient(G),
    "average_clustering_coeff" : nx.average_clustering(G),
    "giant_clique_size" : giant_size,
    "giant_component_size" : len(G0),
    "giant_component_clustering_coeff" : nx.average_clustering(G0),
    "graph_density" : nx.density(G),
    "node_highest_degree_centrality" : [top_dc_node, top_dc_value],
    "node_highest_betweeness_centrality" : [top_bc_node, top_bc_value],
    "node_highest_closeness_centrality" : [top_cc_node, top_cc_value],

}

with open("./Results/q1_statistics.txt", "w") as f:
    print(statistics, file=f)
    
plt.clf()

print('Completed analysis for Question 1!')
print('Running analysis for Question 2...')
# # Q2 analysis of collab network (Yearly granularity)

# In[22]:


yearly_collab_network = pd.read_csv("./output/year_granularity_df.csv", index_col="Unnamed: 0")


# In[23]:


def literal_eval_special(x):
    try:
        return ast.literal_eval(x)
    
    except:
        return None


# In[25]:


yearly_stats_dict = {}
valid_authors = list(yearly_collab_network.columns)

for i in range(0, len(yearly_collab_network)):
    
    #get the publishing list per author for the year
    
    year = yearly_collab_network.iloc[i].name
    collab_network = pd.DataFrame(yearly_collab_network.iloc[i]).reset_index(names = ["author_pid"]).rename({year: "coauthors_list"}, axis="columns")

    collab_network["coauthors_list"] = collab_network["coauthors_list"].apply(lambda x: literal_eval_special(x))
    collab_network = collab_network[collab_network["coauthors_list"].notnull()].reset_index() #drop non-publishers
    
    newpath = f"./Results/{year}"
    if not os.path.exists(newpath):
        os.makedirs(newpath)
    
    #load graph
    
    G = nx.Graph()
    for j in range(0, len(collab_network)):
    
        author = collab_network.iloc[j]["author_pid"]
        coauthors = collab_network.iloc[j]["coauthors_list"][0]
        G.add_node(author)
    
        for coauthor in coauthors:
            if coauthor in valid_authors:
                if coauthor not in G.nodes():
                    G.add_node(coauthor)
            
                G.add_edge(author, coauthor)
    
    
    #run the bunch of analysis
    
    #graph visualisation
    plt.figure(figsize=(40, 40))
    nx.draw(G, with_labels=True)
    plt.savefig(f"./Results/{year}/q2_{year}_graph_visualisation.png") 
    
    #degree centrality
    degrees = [len(list(G.neighbors(n))) for n in G.nodes()]
    deg_cent = nx.degree_centrality(G)
    
    plt.figure(figsize=(20, 20))
    plt.xlabel("Degree_centrality")
    plt.ylabel("Number_of_nodes")
    plt.hist(list(deg_cent.values()), bins=50)
    plt.savefig(f"./Results/{year}/q2_{year}_deg_cent.png")
    
    #node degree
    plt.figure(figsize=(20, 20))
    plt.xlabel("Node_degree")
    plt.ylabel("Number_of_nodes")
    plt.hist(degrees, bins=50)
    plt.savefig(f"./Results/{year}/q2_{year}_deg_dist.png")
    
    #correlation of degree centrality and node degree
    plt.figure(figsize=(20, 20))
    plt.xlabel("Node_degree")
    plt.ylabel("Degree_centrality")
    plt.scatter(degrees, list(deg_cent.values()))
    plt.savefig(f"./Results/{year}/q2_{year}_cent_and_deg_corr.png")
    
    #correlation of degree centrality and betweenness centrality
    bet_cent = nx.betweenness_centrality(G)
    plt.figure(figsize=(20, 20))
    plt.scatter(list(bet_cent.values()), list(deg_cent.values()))
    plt.xlabel("Betweeness_centrality")
    plt.ylabel("Degree_centrality")
    plt.savefig(f"./Results/{year}/q2_{year}_bet_and_deg_cent.png")
    
    #centrality measures
    top_dc_node, top_dc_value = find_nodes_with_highest_deg_cent(G)
    #print(f"Author for year {year} with highest degree centrality at {top_dc_value}: ", top_dc_node[0])
    
    top_bc_node, top_bc_value = find_node_with_highest_bet_cent(G)
    #print(f"Author for year {year} with highest betweeness centrality at {top_bc_value}: ", top_bc_node[0])
    
    top_cc_node, top_cc_value = find_node_with_highest_close_cent(G)
    #print(f"Author for year {year} with highest closeness centrality at {top_cc_value}: ", top_cc_node[0])
    
    #giant cliques
    giant_size, giant_cliques = maximal_cliques(G)
    #print(f"Size of giant clique in {year}: {giant_size}") 
    

    if len(giant_cliques) > 5: 
        giant_cliques_top_5 = giant_cliques[:5] #save only the top 5
    
    else: 
        giant_cliques_top_5 = giant_cliques
    
    i=0
    for clique in giant_cliques_top_5:
        plt.figure(figsize=(20, 20))
        nx.draw(G.subgraph(clique), with_labels=True)
        plt.savefig(f"./Results/{year}/q2_{year}_giant_clique_{i}.png")
        i+=1
        
    #giant component
    Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
    G0 = G.subgraph(Gcc[0])
    plt.figure(figsize=(50, 50))
    nx.draw(G.subgraph(G0), with_labels=True)
    plt.savefig(f"./Results/{year}/q2_{year}_giant_component.png")
    
    
    #summary statistics
    statistics = {
        
        "number of nodes" : len(G.nodes()),
        "number of edges" : len(G.edges()),
        "average_node_degree" : (2*len(G.edges()))/len(G.nodes()),
        "max_node_degree" : [list(G)[np.argmax(degrees)], max(degrees)],
        "degree_assortativity" : nx.degree_assortativity_coefficient(G),
        "average_clustering_coeff" : nx.average_clustering(G),
        "giant_component_clustering_coeff" : nx.average_clustering(G0),
        "giant_component_size" : len(G0),
        "graph_density" : nx.density(G),
        "node_highest_degree_centrality" : [top_dc_node, top_dc_value],
        "node_highest_betweeness_centrality" : [top_bc_node, top_bc_value],
        "node_highest_closeness_centrality" : [top_cc_node, top_cc_value],

    }
    
    #save
    with open(f"./Results/{year}/q2_{year}_statistics.txt", "w") as f:
        print(statistics, file=f)
        
    print(year, " Completed")
    
    #clear for next iter
    yearly_stats_dict[year] = statistics
    plt.close("all")
    


# In[51]:


nodes_list = []
edges_list = []
avg_node_degree = []
degree_assortativity = []
avg_cluster = []
density = []
giant_comp = []

for key in sorted(yearly_stats_dict.keys()):
    
    stats = yearly_stats_dict[key]
    
    nodes_list.append(stats["number of nodes"])
    edges_list.append(stats["number of edges"])
    avg_node_degree.append(stats["average_node_degree"])
    degree_assortativity.append(stats["degree_assortativity"])
    avg_cluster.append(stats["average_clustering_coeff"])
    density.append(stats["graph_density"])
    giant_comp.append(stats["giant_component_size"])


# # Visualise graphs

# ## Number of nodes (Publishing researchers)

# In[44]:


plt.figure(figsize=(15, 10))
plt.xlabel("Year")
plt.ylabel("Number of Nodes")
plt.plot(sorted(yearly_stats_dict.keys()), nodes_list)
plt.savefig(f"./Results/q2_node_trend.png")


# ## Number of edges (Collaborations)

# In[45]:


plt.figure(figsize=(15, 10))
plt.xlabel("Year")
plt.ylabel("Number of Edges")
plt.plot(sorted(yearly_stats_dict.keys()), edges_list)
plt.savefig(f"./Results/q2_edge_trend.png")


# ## Average node degree (average number of collaborations per researcher)

# In[46]:


plt.figure(figsize=(15, 10))
plt.xlabel("Year")
plt.ylabel("Average node degree")
plt.plot(sorted(yearly_stats_dict.keys()), avg_node_degree)
plt.savefig(f"./Results/q2_node_degree_trend.png")


# ## Degree assortativity

# In[47]:


plt.figure(figsize=(15, 10))
plt.xlabel("Year")
plt.ylabel("Average degree assortativity")
plt.plot(sorted(yearly_stats_dict.keys()), degree_assortativity)
plt.savefig(f"./Results/q2_degree_assort_trend.png")


# ## Average clustering coefficient

# In[48]:


plt.figure(figsize=(15, 10))
plt.xlabel("Year")
plt.ylabel("Average clustering coefficient")
plt.plot(sorted(yearly_stats_dict.keys()), avg_cluster)
plt.savefig(f"./Results/q2_clustering_coeff_trend.png")


# ## Graph density

# In[49]:


plt.figure(figsize=(15, 10))
plt.xlabel("Year")
plt.ylabel("Graph density")
plt.plot(sorted(yearly_stats_dict.keys()), density)
plt.savefig(f"./Results/q2_graph_density.png")


# ## Giant component size

# In[52]:


plt.figure(figsize=(15, 10))
plt.xlabel("Year")
plt.ylabel("Giant component size")
plt.plot(sorted(yearly_stats_dict.keys()), giant_comp)
plt.savefig(f"./Results/q2_giant_component_trend.png")


print('Completed analysis for Question 2!')
print('Performing transformation for Question 4...')

# # Q4 Transform the collaboration network

# In[167]:


#Augment the original collaboration network
collab_network = pd.read_csv("./output/collab_network_csv.csv", index_col="Unnamed: 0")
collab_network.drop_duplicates(subset = "author_pid", inplace = True, ignore_index = True)
collab_network["coauthors_list"] = collab_network["coauthors_list"].apply(lambda x: ast.literal_eval(x))
collab_network["institute"] = collab_network["institute"].astype(str)
collab_network["expertise"] = pd.Series(np.random.choice(range(1, 11), len(collab_network)))
collab_network["expertise"] = collab_network["expertise"].astype('Int64')

collab_network.set_index("author_pid", inplace=True)


# In[172]:


from sklearn.utils.class_weight import compute_class_weight
from numpy.random import choice

def sort_list(sub_li):
    sub_li.sort(key = lambda x: x[1])
    return sub_li

def get_index(index_list, item):
    return index_list[1][np.where(index_list[0]==item)][0].astype(np.float64)
    
G = nx.Graph()
G.add_nodes_from(collab_network.index)
    
degree_threshold = int(input("input the user-specified kmax for generation: "))  #kmax

for i in range(0, len(collab_network)):
    
    author = collab_network.index[i]
    coauthors = list(collab_network.loc[author]["coauthors_list"])
    
    #process the coauthors list
    #random.shuffle(coauthors)
    
    #track coauthors diversity
    countries = []
    institutions = []
    expertise = []
    
    for coauthor in coauthors: 
        if coauthor in G.nodes():  #keep track of coauthorship stats
            
            country = collab_network.loc[coauthor, "country"]
            countries.append(country)
                
            institute = collab_network.loc[coauthor, "institute"]
            institutions.append(institute)
              
            expert = collab_network.loc[coauthor, "expertise"]
            expertise.append(expert)
    
    unique_countries = list(np.unique(countries))
    class_weights = compute_class_weight(class_weight = "balanced", classes=unique_countries, y = countries)
    countries_prob_distribution = dict(zip(unique_countries, class_weights / np.sum(class_weights)))
    
    unique_institutions = list(np.unique(institutions))
    #use a list to avoid key clashing
    class_weights = compute_class_weight(class_weight = "balanced", classes=unique_institutions, y = institutions)
    institutions_prob_distribution = np.array([unique_institutions, class_weights / np.sum(class_weights)])
    
    unique_expertise = list(np.unique(expertise))
    class_weights = compute_class_weight(class_weight = "balanced", classes=unique_expertise, y = expertise)
    expertise_prob_distribution = dict(zip(unique_expertise, class_weights / np.sum(class_weights)))
    
    #assign each coauthor an index
    
    coauthor_index_list=[]
    
    for coauthor in coauthors: 
        if coauthor in G.nodes():
            coauthor_country_index = countries_prob_distribution[collab_network.loc[coauthor, "country"]]
            
            coauthor_institute = collab_network.loc[coauthor, "institute"]
            coauthor_institute_index = get_index(institutions_prob_distribution, coauthor_institute)
            
            coauthor_expertise_index = expertise_prob_distribution[collab_network.loc[coauthor, "expertise"]]
            
            #print(type(coauthor_country_index), type(coauthor_institute_index), type(coauthor_expertise_index))
            coauthor_index = coauthor_country_index * coauthor_institute_index * coauthor_expertise_index
            coauthor_index_list.append([coauthor, coauthor_index])
    
    coauthor_index_list = sort_list(coauthor_index_list)
    #print(coauthor_index_list)
    

    if len(coauthors) > degree_threshold: #limit node degree
        coauthors = coauthors[:degree_threshold]
    
    for coauthor in coauthors:
        if coauthor in G.nodes() and G.degree[coauthor] < degree_threshold:
            G.add_edge(author, coauthor)
            
print("Transformed graph generated!")
print('Running analysis for Question 4...')

# In[173]:


newpath = f"./Results/Transformed"
if not os.path.exists(newpath):
    os.makedirs(newpath)

plt.figure(figsize=(40, 40))
nx.draw(G, with_labels=True)
plt.savefig(f"./Results/Transformed/q4_graph_visualisation.png") 
    
#degree centrality
degrees = [len(list(G.neighbors(n))) for n in G.nodes()]
deg_cent = nx.degree_centrality(G)
    
plt.figure(figsize=(20, 20))
plt.xlabel("Degree_centrality")
plt.ylabel("Number_of_nodes")
plt.hist(list(deg_cent.values()), bins=50)
plt.savefig(f"./Results/Transformed/q4_deg_cent.png")
    
#node degree
plt.figure(figsize=(20, 20))
plt.xlabel("Node_degree")
plt.ylabel("Number_of_nodes")
plt.hist(degrees, bins=50)
plt.savefig(f"./Results/Transformed/q4_deg_dist.png")
    
#correlation of degree centrality and node degree
plt.figure(figsize=(20, 20))
plt.xlabel("Node_degree")
plt.ylabel("Degree_centrality")
plt.scatter(degrees, list(deg_cent.values()))
plt.savefig(f"./Results/Transformed/q2_q4_cent_and_deg_corr.png")
    
#correlation of degree centrality and betweenness centrality
bet_cent = nx.betweenness_centrality(G)
plt.figure(figsize=(20, 20))
plt.scatter(list(bet_cent.values()), list(deg_cent.values()))
plt.xlabel("Betweeness_centrality")
plt.ylabel("Degree_centrality")
plt.savefig(f"./Results/Transformed/q4_bet_and_deg_cent.png")
    
#centrality measures
top_dc_node, top_dc_value = find_nodes_with_highest_deg_cent(G)
    
top_bc_node, top_bc_value = find_node_with_highest_bet_cent(G)
    
top_cc_node, top_cc_value = find_node_with_highest_close_cent(G)
    
#giant cliques
giant_size, giant_cliques = maximal_cliques(G)
    

if len(giant_cliques) > 5: 
    giant_cliques_top_5 = giant_cliques[:5] #save only the top 5
    
else: 
    giant_cliques_top_5 = giant_cliques
    
i=0
for clique in giant_cliques_top_5:
    plt.figure(figsize=(20, 20))
    nx.draw(G.subgraph(clique), with_labels=True)
    plt.savefig(f"./Results/Transformed/q4_giant_clique_{i}.png")
    i+=1
        
#giant component
Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
G0 = G.subgraph(Gcc[0])
plt.figure(figsize=(50, 50))
nx.draw(G.subgraph(G0), with_labels=True)
plt.savefig(f"./Results/Transformed/q4_giant_component.png")
    
    
#summary statistics
statistics = {
        
    "number of nodes" : len(G.nodes()),
    "number of edges" : len(G.edges()),
    "average_node_degree" : (2*len(G.edges()))/len(G.nodes()),
    "max_node_degree" : [list(G)[np.argmax(degrees)], max(degrees)],
    "degree_assortativity" : nx.degree_assortativity_coefficient(G),
    "average_clustering_coeff" : nx.average_clustering(G),
    "giant_component_clustering_coeff" : nx.average_clustering(G0),
    "giant_component_size" : len(G0),
    "graph_density" : nx.density(G),
    "node_highest_degree_centrality" : [top_dc_node, top_dc_value],
    "node_highest_betweeness_centrality" : [top_bc_node, top_bc_value],
    "node_highest_closeness_centrality" : [top_cc_node, top_cc_value],

}
    
#save
with open(f"./Results/Transformed/q4_statistics.txt", "w") as f:
    print(statistics, file=f)
        
print("Transformation analysis completed!")
print("Program Terminated...")

# In[ ]:




