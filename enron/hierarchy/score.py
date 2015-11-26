# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

df = pd.read_csv('output.csv',nrows=10000, names=['path','date','id','from','to_val','to_type'])

print 'create sub df'

df_1 = df[['id','to_type','to_val']]
#print df_1
df_2 = df_1[df_1['to_type'] == 'Cc']

print '$$$$$$$$$$$$$$$$$$$$$$$$$$'
#print df_2

print df_2['to_val'].value_counts()

quit()

print '======================='
g =  df_1.groupby(['id'])
for member in g:
    print member
print '======================='

df_group = df_1.groupby(['id','to_val']).count()
print df_group

