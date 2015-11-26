# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd

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

def main(n_rows):
    df = pd.read_csv('data/output.csv',nrows=n_rows, names=['path','date','id','from','to_val','to_type'])
    cc_score_frame = get_cc_score_frame(df)
    print 'cc_score_frame[:10]'
    print cc_score_frame[:10]
    print '++++++++++++++++++++++++++++++++++++'    
    to_score_frame = get_to_score_frame(df)
    print to_score_frame[:10]    
    print '++++++++++++++++++++++++++++++++++++'    
    cc_team_score_frame = get_cc_team_sender_score_frame(df)
    print cc_team_score_frame[:10]    
    print '++++++++++++++++++++++++++++++++++++'    
    to_team_score_frame = get_to_team_sender_score_frame(df)
    print to_team_score_frame[:10]    
    print '++++++++++++++++++++++++++++++++++++'    
    df_result = pd.DataFrame({"cc_score_frame": cc_score_frame, "to_score_frame": to_score_frame, 
                              "cc_team_score_frame": cc_team_score_frame, "to_team_score_frame": to_team_score_frame})
    return df_result    
    #return (cc_score_frame,to_score_frame,to_team_score_frame)
    
    
if __name__ == '__main__':
    t = main(100000)
    print 'cc_score_frame'
    print t.loc['william.kendrick@enron.com']
    print t.loc['larry.campbell@enron.com']
#    print t.iloc[:10]
    print len(t)
#    print 'to_score_frame'
#    print t[1]
#    print 'to_team_frame'
#    print t[2]
#    
