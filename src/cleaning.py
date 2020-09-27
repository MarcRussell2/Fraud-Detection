import string, pickle
import pandas as pd 
import numpy as np 
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize


def clean_training_dataframe(raw_fp):
    '''
    Take filepath, creating dataframe from 

    Input: Filepath
    Output: Pandas dataframe
    '''
    df = pd.read_json(raw_fp)
    df['fraud'] = df['acct_type'].str.contains('fraud')
    df_cleaned = df[['description', 'has_logo', 'listed', 'name', 'num_payouts', 'org_desc', 'user_age', 'user_type', 'email_domain','fraud']]
    df_cleaned['org_description'] = (df_cleaned['org_desc']!='').astype(int)
    df_cleaned.drop('org_desc', axis=1, inplace=True)
    df_cleaned['listed']=(df_cleaned['listed']=='y').astype(int)
    return df, df_cleaned

def make_corpus(df_cleaned):
    '''
    Make corpus series out of description and organization name. 

    Input: Pandas dataframe
    Output: Pandas series
    '''
    docs = []
    for doc in df_cleaned['description']:
        soup = BeautifulSoup(doc, 'lxml')
        d = soup.get_text()
        d = d.replace('\n',' ').replace('\xa0','').replace("\'"," ").replace("\r"," ")
        docs.append(d)
    df_cleaned['parsed_desc'] = docs
    corpus = df_cleaned['parsed_desc']+df_cleaned['name']
    return corpus

def prep_corpus(corpus):
    '''
    Prep corpus by lowering, removing punctuation, lemmatizing for tfidf vectorizer.

    Input: Pandas series
    Output: Pandas series
    '''
    corpus_removepunc = []
    for c in corpus: 
        #corpus_removepunc.append(c.translate(str.maketrans('', '', string.punctuation)).lower())
        corpus_removepunc.append(c.translate(str.maketrans('', '', string.punctuation)))
    corpus = pd.Series(corpus_removepunc) 
    corpus = corpus.apply(lambda x: lemmatize_str(x, wordnet=True)) 
    return corpus

def lemmatize_str(string, wordnet=True): 
    ''' 
    Lemmatize string using nltk WordNet 
      
    Input: string 
    Output: string 
    ''' 
    if wordnet: 
        w_tokenizer = word_tokenize 
        lemmatizer = WordNetLemmatizer() 
        lemmed = " ".join([lemmatizer.lemmatize(w) for w in w_tokenizer(string)]) 
        return lemmed

def get_top_features_cluster(tf_idf_array, prediction, n_feats): 
    '''
    Get top words for each cluster.
    '''
    labels = np.unique(prediction) 
    dfs = [] 
    for label in labels: 
        id_temp = np.where(prediction==label) # indices for each cluster 
        x_means = np.mean(tf_idf_array[id_temp], axis = 0) # returns average score across cluster 
        sorted_means = np.argsort(x_means)[::-1][:n_feats] # indices with top 20 scores 
        features = vectorizer.get_feature_names() 
        best_features = [(features[i], x_means[i]) for i in sorted_means] 
        df = pd.DataFrame(best_features, columns = ['features', 'score']) 
        dfs.append(df) 
    return dfs

if __name__ == "__main__":

    original, df = clean_training_dataframe('data/data.json') ## Entire dataset being trained
    print('Making corpus...')
    corpus = make_corpus(df)
    stop_words = set(stopwords.words('english'))
    extra = ['s', 'de', 'la', 'en', 'et', 'le', 'des', 'de la', 'les', 'vous', 'pour', 'rouen', 'us', 'dec', '00', '2013', '30', '10', 'us', 'www', 'new']
    all_stop = stop_words.union(extra)
    print('Prep corpus for tfidf...')
    prepped_corpus = prep_corpus(corpus)

    print('Making modelling dataframe...')
    df_modelling = df.drop(['description', 'name', 'parsed_desc'], axis=1)
    # df_modelling = df.drop(['description', 'name'], axis=1)
    df_modelling['percent_upper'] = [sum(1 for c in entry if c.isupper())/len(entry) if len(entry)>0 else np.NaN for entry in prepped_corpus]
    df_modelling.dropna(how='any', inplace=True)
    # Top 6 email domains.  one-hot encode whether or not a domain is in this list
    # "gmail.com, yahoo.com, hotmail.com, aol.com, live.com, me.com" 
    
    df_modelling['common_domain'] = [('gmail' or 'yahoo' or 'live' or 'me' or 'hotmail' or 'aol') in entry for entry in df_modelling['email_domain']]

    df_modelling.drop(columns='email_domain', axis=1, inplace=True)
    df_modelling.to_pickle('data/fraud_data.pkl')   

          

