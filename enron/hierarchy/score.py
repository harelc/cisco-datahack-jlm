# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

def get_cc_score_frame(df):
    df_sub = df[['id','to_type','to_val']]
    df_cc_column = df_sub[df_sub['to_type'] == 'Cc']
    return df_cc_column['to_val'].value_counts()
    
def get_to_score_frame(df):
    df_sub = df[['id','to_type','to_val']]
    df_to_column = df_sub[df_sub['to_type'] == 'To']
    return df_to_column['to_val'].value_counts()
    
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
    t = main(10000)
    print 'cc_score_frame'
    print t[0]
    print 'to_score_frame'
    print t[1]
    print 'to_team_frame'
    print t[2]
    
