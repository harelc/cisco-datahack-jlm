# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

def get_scrore_by_type(df, tpy, filter_func):
    df_sub = df[['id','to_type','to_val']]
    df_type_filtered = df_sub[df_sub['to_type'] == tpy]
    filtered_by_func = df_type_filtered[df_type_filtered['to_val'].apply(filter_func)]
    return filtered_by_func['to_val'].value_counts()
    
    
    

def get_cc_score_frame(df):
    df_sub = df[['id','to_type','to_val']]
    df_cc_column = df_sub[df_sub['to_type'] == 'Cc']
    no_team_cc_column = df_cc_column[df_cc_column['to_val'].apply(lambda x: not x.startswith('team'))]
    return no_team_cc_column['to_val'].value_counts()
    
            
def get_to_score_frame(df):
    df_sub = df[['id','to_type','to_val']]
    df_to_column = df_sub[df_sub['to_type'] == 'To']
    no_team_to_column = df_to_column[df_to_column['to_val'].apply(lambda x: not x.startswith('team'))]
    return no_team_to_column['to_val'].value_counts()
    
def get_to_team_score_frame(df):
    df_sub = df[['id','to_type','to_val']]
    df_to_column = df_sub[df_sub['to_type'] == 'To']
    df_to_team_column = df_to_column[df_to_column['to_val'].str.startswith("team")]
    return df_to_team_column['to_val'].value_counts()

def main(n_rows):
    df = pd.read_csv('data/output.csv',nrows=n_rows, names=['path','date','id','from','to_val','to_type'])
    cc_score_frame = get_cc_score_frame(df)
    to_score_frame = get_to_score_frame(df)
    to_team_score_frame = get_to_team_score_frame(df)
    return (cc_score_frame,to_score_frame,to_team_score_frame)
    
    
if __name__ == '__main__':
    t = main(100)
    print 'cc_score_frame'
    print t[0]
#    print 'to_score_frame'
#    print t[1]
#    print 'to_team_frame'
#    print t[2]
#    
