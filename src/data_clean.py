import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup

def get_new_transaction():
    r = requests.get('http://galvanize-case-study-on-fraud.herokuapp.com/data_point')
    soup = BeautifulSoup(r.text, 'html.parser')
    top_split = soup.prettify().split('{')[1].split(',')
    bot_split = soup.prettify().split('{')[-1].split(',')
    return top_split, bot_split

columns=['org_description',
        'has_logo',
        'listed', 
        'num_payouts',
        'org_name',
        'name',
        'user_age',
        'user_type']

df = pd.DataFrame(columns=columns)
df = df.append({'has_logo':0 }, ignore_index=True)

# def get_has_logo(df, top_split, row_num,
#                  start_padding = 3, 
#                  end_padding = 0,
#                  col_str = "org_desc"):
#     col_str_len = len(col_str)
#     for feature in top_split:
#         if col_str in feature:
#             idx = feature.find(col_str)
#             value = feature[start_padding + idx + col_str_len : len(feature)-end_padding]
#             df.loc[row_num, col_str] = value
#     return df

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
            df.loc[row_num, col_str] = value
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

def get_data():
    num_rows = 10
    for row_num in range(num_rows):
    #   check duplicate function and wait if true 
        top_split, bot_split = get_new_transaction()
        get_has_logo(df, top_split, row_num)
        get_listed(df, top_split, row_num)
        get_num_payouts(df, top_split, row_num)
        get_org_name(df, top_split, row_num)
        get_name(df, bot_split, row_num)
        get_user_age(df, bot_split, row_num)
        get_user_type(df, bot_split, row_num)
    return df