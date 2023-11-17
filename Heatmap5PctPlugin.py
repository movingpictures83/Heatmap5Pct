#!/usr/bin/env python
# coding: utf-8

# In[31]:


import numpy as np
import seaborn as sb
import pandas as pd
import seaborn as sns
sns.set(color_codes=True)

from scipy import stats
import matplotlib.pyplot as plt
import os.path as path
plt.style.use('seaborn-whitegrid')


# In[51]:

class Heatmap5PctPlugin:
 def input(self, inputfile):
  self.infile = inputfile
  self.df_all = pd.read_csv(inputfile)
 def run(self):
     pass
 def output(self, outputfile):
  self.df_all.columns = ['traces','algorithm','cache_size','hit_rate', 'dataset', 'rank']

  self.df_all = self.df_all[(self.df_all.algorithm != 'alecar6') & (self.df_all.algorithm != 'scanalecar')]

  algorithms = self.df_all['algorithm'].unique()

  print(len(self.df_all))
  num_traces = len(self.df_all)

  sorted_df = self.df_all.sort_values(['dataset', 'cache_size', 'traces', 'hit_rate', 'rank'], ascending=[True, True, True, False, True])
  #print(sorted_df)
  min_ids = sorted_df.groupby(['dataset', 'cache_size', 'traces'])['rank'].transform(min) == sorted_df['rank']
  sorted_min_df = sorted_df[min_ids]
  #print(sorted_max_df)
  sorted_min_grouped = sorted_min_df.groupby(['dataset', 'cache_size'])

  algo_top_count = []
  for name, group in sorted_min_grouped:
    #print(group)
    for algo in algorithms:
        alg_rows = group.loc[group['algorithm'] == algo]
        dataset = group.iloc[0]['dataset']
        cache_size = group.iloc[0]['cache_size']
        alg_count = len(alg_rows)
        algo_top_count.append([dataset, cache_size, algo, alg_count/len(group) * 100])


  for_heat_map = pd.DataFrame(algo_top_count)
  for_heat_map.columns = ['dataset', 'cache_size', 'algorithm', 'percentage']
  sorted_heat_map = for_heat_map.sort_values(['dataset', 'cache_size', 'algorithm'], ascending=[True, True, True])


  # In[49]:


  heatmap_data = pd.pivot_table(sorted_heat_map, values='percentage', 
                     index=['algorithm'], 
                     columns=['dataset', 'cache_size'])
  #print(heatmap_data)
  label = ["0.0005", "0.001", "0.005", "0.01", "0.05", "0.1","0.0005", "0.001", "0.005", "0.01", "0.05", "0.1","0.0005", "0.001", "0.005", "0.01", "0.05", "0.1","0.0005", "0.001", "0.005", "0.01", "0.05", "0.1","0.0005", "0.001", "0.005", "0.01", "0.05", "0.1"]
  top_label = ["CloudCache", "CloudVps", "FIU", "MSR", "Nexus"]
  tick_l = [0.1, 0.3, 0.5, 0.7, 0.9]
  mid = (for_heat_map['percentage'].max() - for_heat_map['percentage'].min()) / 2

  fig, ax1 = plt.subplots(figsize=(15, 2.25))
  sb.set_style('white')
  ax1 = sb.heatmap(heatmap_data,cmap="Greens", annot=True, center=mid)

  ax2 = ax1.twiny()
  ax1.set_xticklabels(label)
  ax2.set_xticklabels(top_label)
  ax2.set_xticks(tick_l)

  ax1.set_xlabel("cache size(% of workload footprint)")
  fig.savefig(outputfile, format='png', bbox_inches = 'tight', dpi=600)
  plt.show()


  # In[52]:


  self.df_all = pd.read_csv(self.infile)
  self.df_all.columns = ['traces','algorithm','cache_size','hit_rate', 'dataset', 'rank']

  self.df_all = self.df_all[(self.df_all.algorithm != 'alecar6')]

  algorithms = self.df_all['algorithm'].unique()

  print(len(self.df_all))
  num_traces = len(self.df_all)

  sorted_df = self.df_all.sort_values(['dataset', 'cache_size', 'traces', 'hit_rate', 'rank'], ascending=[True, True, True, False, True])
  #print(sorted_df)
  min_ids = sorted_df.groupby(['dataset', 'cache_size', 'traces'])['rank'].transform(min) == sorted_df['rank']
  sorted_min_df = sorted_df[min_ids]
  #print(sorted_max_df)
  sorted_min_grouped = sorted_min_df.groupby(['dataset', 'cache_size'])

  algo_top_count = []
  for name, group in sorted_min_grouped:
    #print(group)
    for algo in algorithms:
        alg_rows = group.loc[group['algorithm'] == algo]
        dataset = group.iloc[0]['dataset']
        cache_size = group.iloc[0]['cache_size']
        alg_count = len(alg_rows)
        algo_top_count.append([dataset, cache_size, algo, alg_count/len(group) * 100])


  for_heat_map = pd.DataFrame(algo_top_count)
  for_heat_map.columns = ['dataset', 'cache_size', 'algorithm', 'percentage']
  sorted_heat_map = for_heat_map.sort_values(['dataset', 'cache_size', 'algorithm'], ascending=[True, True, True])


  # In[53]:


  heatmap_data = pd.pivot_table(sorted_heat_map, values='percentage', 
                     index=['algorithm'], 
                     columns=['dataset', 'cache_size'])
  #print(heatmap_data)
  label = ["0.0005", "0.001", "0.005", "0.01", "0.05", "0.1","0.0005", "0.001", "0.005", "0.01", "0.05", "0.1","0.0005", "0.001", "0.005", "0.01", "0.05", "0.1","0.0005", "0.001", "0.005", "0.01", "0.05", "0.1","0.0005", "0.001", "0.005", "0.01", "0.05", "0.1"]
  top_label = ["CloudCache", "CloudVps", "FIU", "MSR", "Nexus"]
  tick_l = [0.1, 0.3, 0.5, 0.7, 0.9]
  mid = (for_heat_map['percentage'].max() - for_heat_map['percentage'].min()) / 2

  fig, ax1 = plt.subplots(figsize=(15, 2.25))
  sb.set_style('white')
  ax1 = sb.heatmap(heatmap_data,cmap="Greens", annot=True, center=mid)

  ax2 = ax1.twiny()
  ax1.set_xticklabels(label)
  ax2.set_xticklabels(top_label)
  ax2.set_xticks(tick_l)

  ax1.set_xlabel("cache size(% of workload footprint)")
  #fig.savefig('figure2_not_all.png', format='png', bbox_inches = 'tight', dpi=600)
  plt.show()


  # In[ ]:




