# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sklearn.cluster
from sklearn.cluster import KMeans 

import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets


def get_score_by_type(df, tpy, filter_func, column='to_val'):
    df_sub = df[['id','to_type', 'from','to_val']]
    df_type_filtered = df_sub[df_sub['to_type'] == tpy]
    filtered_by_func = df_type_filtered[df_type_filtered['to_val'].apply(filter_func)]
    return filtered_by_func[column].value_counts()
    
def get_cc_score_frame(df):
     return get_score_by_type(df, 'Cc',lambda x: not x.startswith('team'))

def get_to_score_frame(df):
     return get_score_by_type(df, 'To',lambda x: not x.startswith('team'))    
    
def get_cc_team_sender_score_frame(df):
     return get_score_by_type(df, 'Cc',lambda x: x.startswith('team'), 'from')    

def get_to_team_sender_score_frame(df):
     return get_score_by_type(df, 'To',lambda x: x.startswith('team'), 'from')  
     


def main(n_rows = None):
    if n_rows != None and n_rows > 0:
        df = pd.read_csv('data/output.csv',nrows=n_rows, names=['path','date','id','from','to_val','to_type'])
    else:
        df = pd.read_csv('data/output.csv', names=['path','date','id','from','to_val','to_type'])
        
    cc_score_frame = get_cc_score_frame(df)
    print 'cc_score_frame[:10]'
    print cc_score_frame[:20]
    print '++++++++++++++++++++++++++++++++++++'    
    to_score_frame = get_to_score_frame(df)
    print to_score_frame[:20]    
    print '++++++++++++++++++++++++++++++++++++'    
    cc_team_score_frame = get_cc_team_sender_score_frame(df)
    print cc_team_score_frame[:20]    
    print '++++++++++++++++++++++++++++++++++++'    
    to_team_score_frame = get_to_team_sender_score_frame(df)
    print to_team_score_frame[:20]    
    print '++++++++++++++++++++++++++++++++++++'    
    df_result = pd.DataFrame({"cc_score_frame": cc_score_frame, "to_score_frame": to_score_frame, 
                              "cc_team_score_frame": cc_team_score_frame, "to_team_score_frame": to_team_score_frame})
    return df_result    
    #return (cc_score_frame,to_score_frame,to_team_score_frame)
    
    
if __name__ == '__main__':
    t = main(50000)
#    t =  main()
    t = t.fillna(0)
    
    kclust = KMeans(n_clusters=4)
    kclust.fit(t)
    clustered=kclust.transform(t)
    
    X = t
    fignum = 1
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

    plt.cla()
    labels = kclust.labels_

    ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('Petal width')
    ax.set_ylabel('Sepal length')
    ax.set_zlabel('Petal length')
    
    fig = plt.figure(fignum, figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)
    
    for name, label in [('Superior', 0),
                        ('Hight', 1),
                        ('Medium', 2),
                        ('Low', 3)]:
#        ax.text3D(X[y == label, 3].mean(),
#                  X[y == label, 0].mean() + 1.5,
#                  X[y == label, 2].mean(), name,
#                  horizontalalignment='center',
#                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
#    # Reorder the labels to have colors matching the cluster results
#        y = np.choose(y, [1, 2, 0]).astype(np.float)
#        ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y)
        
        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('Petal width')
        ax.set_ylabel('Sepal length')
        ax.set_zlabel('Petal length')
        plt.show()


 
    
            