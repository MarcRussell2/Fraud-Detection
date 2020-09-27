import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

def get_new_transaction():
    r = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point')
    soup = BeautifulSoup(r.text, 'html.parser')
    top_split = soup.prettify().split('{')[1].split(',')
    bot_split = soup.prettify().split('{')[-1].split(',')
    return soup, top_split, bot_split

columns=['description',
        'has_logo',
        'listed', 
        'num_payouts',
        'org_name',
        'name',
        'user_age',
        'user_type']

df = pd.DataFrame(columns=columns)
df = df.append({'has_logo':0 }, ignore_index=True)

def get_has_logo(df, top_split, row_num,
                 start_padding = 3, 
                 end_padding = 0,
                 col_str = "has_logo"):
    col_str_len = len(col_str)
    for feature in top_split:
        if col_str in feature:
            idx = feature.find(col_str)
            value = feature[start_padding + idx + col_str_len:]
            df.loc[row_num, col_str] = value
    return df

def get_listed(df, top_split, row_num,
                 start_padding = 4, 
                 end_padding = 1,
                 col_str = "listed"):
    col_str_len = len(col_str)
    for feature in top_split:
        if col_str in feature:
            idx = feature.find(col_str)
            value = feature[start_padding + idx + col_str_len : len(feature)-end_padding]
            if type(value)==type(True):
                df.loc[row_num, col_str] = int(value)
            else:
                df.loc[row_num, col_str] = int(value=='y')
    return df

def get_num_payouts(df, top_split, row_num,
                 start_padding = 2, 
                 end_padding = 0,
                 col_str = "num_payouts"):
    col_str_len = len(col_str)
    for feature in top_split:
        if col_str in feature:
            idx = feature.find(col_str)
            value = feature[start_padding + idx + col_str_len : len(feature)-end_padding]
            df.loc[row_num, col_str] = value
    return df

def get_org_name(df, top_split, row_num,
                 start_padding = 3, 
                 end_padding = 1,
                 col_str = "org_name"):
    col_str_len = len(col_str)
    for feature in top_split:
        if col_str in feature:
            idx = feature.find(col_str)
            value = feature[start_padding + idx + col_str_len : len(feature)-end_padding]
            df.loc[row_num, col_str] = value
    return df

def get_name(df, top_split, row_num,
                 start_padding = 3, 
                 end_padding = 1,
                 col_str = "name"):
    col_str_len = len(col_str)
    for feature in top_split:
        if col_str in feature:
            idx = feature.find(col_str)
            value = feature[start_padding + idx + col_str_len : len(feature)-end_padding]
            df.loc[row_num, col_str] = value
    return df

def get_user_age(df, bot_split, row_num,
                 start_padding = 2, 
                 end_padding = 0,
                 col_str = "user_age"):
    col_str_len = len(col_str)
    for feature in bot_split:
        if col_str in feature:
            idx = feature.find(col_str)
            value = feature[start_padding + idx + col_str_len : len(feature)-end_padding]
            df.loc[row_num, col_str] = value
    return df

def get_user_type(df, bot_split, row_num, 
                 start_padding = 2, 
                 end_padding = 0,
                 col_str = "user_type"):
    col_str_len = len(col_str)
    for feature in bot_split:
        if col_str in feature:
            idx = feature.find(col_str)
            value = feature[start_padding + idx + col_str_len : len(feature)-end_padding]
            df.loc[row_num, col_str] = value
    return df

def get_desc(df, page_string, row_num,
             start_padding = 2,
             end_padding = 0,
             desc_len_cutoff = 100,
             col_str = "description"):
    desc_start_loc = str(page_string.text).find('"description":')
    desc_end_loc = str(page_string.text).find('"email_domain":')
    desc_len = desc_end_loc - desc_start_loc
    if desc_len >= desc_len_cutoff:
        df.loc[row_num, col_str] = 1
    else:
        df.loc[row_num, col_str] = 0
    return df

def get_data():
    num_rows = 10
    for row_num in range(num_rows):
    #   check duplicate function and wait if true 
        page_string, top_split, bot_split = get_new_transaction()
        get_desc(df, page_string, row_num)
        get_has_logo(df, top_split, row_num)
        get_listed(df, top_split, row_num)
        get_num_payouts(df, top_split, row_num)
        get_org_name(df, top_split, row_num)
        get_name(df, bot_split, row_num)
        get_user_age(df, bot_split, row_num)
        get_user_type(df, bot_split, row_num)
    return df