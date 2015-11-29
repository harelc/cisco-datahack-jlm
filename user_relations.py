#-------------------------------------------------------------------------------
# Name:        module1
# Purpose:
#
# Author:      mmanevit
#
# Created:     25/11/2015
# Copyright:   (c) mmanevit 2015
# Licence:     <your licence>
#-------------------------------------------------------------------------------
import pandas as pd
import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans


FILE ="""C:\DataHackaton\enron\output.txt"""
NROWS = 500000
def draw_graph(df_relations):
    dot_string = 'digraph G {\n'
    for idx, row in df_relations.iterrows():
        dot_string = dot_string+ '"%s" -> "%s" [penwidth="%f"];\n' % \
					(row['from'], row['to_val'],np.log(float(row['id'])))
    dot_string = dot_string +'}'

    open(grapg.txt,'w').write(dot_string)

def read_enron_data(path,nrows):

    df = pd.read_csv(path ,names = ['path','date','id','from','to_val', 'to_type'], nrows=nrows)
    print 'data read'
    # clean data
    df.drop_duplicates(subset=['id','to_val'], inplace=True)
    print 0
    df = df[df['from'].apply(lambda x: x.strip().endswith('@enron.com'))]
    print 1
    #df = df.groupby('from').filter(lambda x: len(x)>10)
    vc = df['from'].value_counts()
    vc = vc[vc>10].index
    df = df[df['from'].apply(lambda x: x in vc)]
    users = list(set(df['from']).intersection(set(df['to_val'])))
    df = df[df['to_val'].apply(lambda x: x in users)]
    df = df[df['from'].apply(lambda x: x in users)]
    print df.shape
    print 'data cleaned'

    return df


def main():
    print 'hello'
    #df =pd.DataFrame(np.random.randint(0,10,(200,4)), columns =['message_id','from','to','date'])
    df = read_enron_data(FILE,NROWS)
    #
    # df_users1 = df[['id','from']]
    # df_users1.columns = ['id','user']
    # df_users2 = df[['id','to_val']]
    # df_users2.columns = ['id','user']
    # df_users = pd.concat([df_users1,df_users2])
    # users = df_users['user'].unique()
    # df_pairs = pd.DataFrame(data=None,index=pd.MultiIndex.from_product([users,users]))
    # df_pairs['count'] = 0
    # grouped_id = df_users.groupby('id')
    # for g_name, g in grouped_id:
    #     for c in itertools.combinations(g['user'].unique(),2):
    #         df_pairs.loc[c,'count'] = df_pairs.loc[c,'count']+1
    # # print df_pairs.head(10)
    # df_pairs.reset_index(inplace=True)
    # df_pairs.columns = ['user1','user2','weight']
    # print df_pairs.iloc[df_pairs['weight'].argsort()[-5:]]
    # G=nx.from_pandas_dataframe(df_pairs, 'user1', 'user2', ['weight'])
    # # nx.draw_spring(G)
    # # plt.show()
    # return

   # for k1_idx, k1 in enumerate(groups_keys):
   #     ms1 = set(grouped.get_group(k1)['message_id'])
##        for k2_idx, k2 in enumerate(groups_keys[k1_idx+1:]):
##            ms2 = set(grouped.get_group(k2)['message_id'])
##            intersection = len(list(ms1.intersection(ms2)))
##            union = len(list(ms1.union(ms2)))
##            df_pairs.loc[(k1,k2)] = (intersection, union)
##
##    df_pairs.dropna(how='any',inplace=True)
##    df_pairs['Jaccard'] = df_pairs['intersection'].astype(float) / df_pairs['union']
##    df_pairs.sort(inplace=True)

##    grouped = df.groupby('to_val')
##    groups_keys = grouped.groups.keys()
##    users = df['to'].unique()
##    df_pairs = pd.DataFrame(data=None,index=pd.MultiIndex.from_product([users,users]),columns=['intersection','union'])
##    for k1_idx, k1 in enumerate(groups_keys):
##        ms1 = set(grouped.get_group(k1)['message_id'])
##        for k2_idx, k2 in enumerate(groups_keys[k1_idx+1:]):
##            ms2 = set(grouped.get_group(k2)['message_id'])
##            intersection = len(list(ms1.intersection(ms2)))
##            union = len(list(ms1.union(ms2)))
##            df_pairs.loc[(k1,k2)] = (intersection, union)
##
##    df_pairs.dropna(how='any',inplace=True)
##    df_pairs['Jaccard'] = df_pairs['intersection'].astype(float) / df_pairs['union']
##    df_pairs.sort(inplace=True)
##    print df_pairs.head(10)
##
##    return

    # count number of mails from sender to receiver
    df_relations = df.groupby(['from','to_val'])['id'].count()
    df_relations = df_relations.reset_index()
    df_relations.columns = ['from','to_val','count']
    df_relations['count_log'] = np.log(df_relations['count']+0.01)
    df_relations = df_relations.pivot(index='from',columns='to_val',values='count_log').fillna(0)
    vc = set(df_relations.index).intersection(set(df_relations.columns))
    square = df_relations.loc[vc,vc]
    mat_relations = square.values

    normed_mat_rel = preprocessing.normalize(mat_relations)
    KM = KMeans(6)
    labels = KM.fit_predict(normed_mat_rel)
    labels_ordered = np.argsort(labels)
    var_by_cluster = [normed_mat_rel[labels==i,:].var(axis=0).sum() for i in range(KM.n_clusters)]
    highest_var= np.argmax(var_by_cluster)
    print var_by_cluster, highest_var
    print square.shape
    #small_sqr = square[labels!=highest_var,labels!=highest_var]
    small_sqr =square[labels!=highest_var]
    small_sqr = small_sqr.transpose()[labels!=highest_var].transpose()
    print small_sqr.shape

    labels_small= [x for x in labels if x!=highest_var]
    labels_ordered_small = np.argsort(labels_small)
    ordered_square = small_sqr.loc[[small_sqr.index[i] for i in labels_ordered_small],[small_sqr.columns[i] for i in labels_ordered_small]]

    ordered_mat = ordered_square.values

    # plt.matshow(small_sqr.values)
    # plt.matshow(ordered_mat)
    # plt.colorbar()
    # plt.show()

    D = nx.DiGraph(ordered_mat)

    pos=nx.spring_layout(D,scale=5) # positions for all nodes

    # nodes
    nx.draw_networkx_nodes(D,pos,node_color = [labels_small[i] for i in labels_ordered_small])
    # edges
    nx.draw_networkx_edges(D,pos,width=[d['weight'] for (u,v,d) in D.edges(data=True)])

    plt.axis('off')
    plt.savefig("weighted_graph.png") # save as png
    plt.show() # display



    plt.show()

    return
    df_senders = df_relations.groupby('from')['id'].sum()
    #print df_senders
    #print df_senders.idxmax(),
    #df_receivers = df_relations.groupby('to')['message_id'].sum()
    #print df_receivers
    #print df_receivers.idxmax()
    #print df_relations
    draw_graph(df_relations)
    print len(df_relations['from'].unique())


    #print df_relations



if __name__ == '__main__':
    main()
