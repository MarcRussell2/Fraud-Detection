import string, pickle
import pandas as pd 
import numpy as np 
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier

def clean_training_dataframe(raw_fp):
    '''
    Take filepath, creating dataframe from 

    Input: Filepath
    Output: Pandas dataframe
    '''
    df = pd.read_json(raw_fp)
    df['fraud'] = df['acct_type'].str.contains('fraud')
    df_cleaned = df[['description', 'has_logo', 'listed', 'name', 'num_payouts', 'org_desc', 'user_age', 'user_type', 'fraud']]
    df_cleaned['org_description'] = (df_cleaned['org_desc']!='').astype(int)
    df_cleaned.drop('org_desc', axis=1, inplace=True)
    df_cleaned['listed']=(df_cleaned['listed']=='y').astype(int)
    return df_cleaned

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
        corpus_removepunc.append(c.translate(str.maketrans('', '', string.punctuation)).lower()) 
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
    log_regression=True
    randomforest = True
    df = clean_training_dataframe('data/data.json') ## Entire dataset being trained
    # print('Making corpus...')
    # corpus = make_corpus(df)
    # stop_words = set(stopwords.words('english'))
    # extra = ['s', 'de', 'la', 'en', 'et', 'le', 'des', 'de la', 'les', 'vous', 'pour', 'rouen', 'us', 'dec', '00', '2013', '30', '10', 'us', 'www', 'new']
    # all_stop = stop_words.union(extra)
    # print('Prep corpus for tfidf...')
    # prepped_corpus = prep_corpus(corpus)
    
    # vectorizer = TfidfVectorizer(stop_words=all_stop, strip_accents='ascii', ngram_range=(1,2), max_features=5000)
    # X = vectorizer.fit_transform(prepped_corpus)
    # print('Starting kMeans...')
    # kmeans = KMeans(n_clusters=5, random_state=0) 
    # kmeans.fit(X.toarray())
    print('Making modelling dataframe...')
    # df_modelling = df.drop(['description', 'name', 'parsed_desc'], axis=1)
    df_modelling = df.drop(['description', 'name'], axis=1)

    # df_modelling['cluster'] = kmeans.labels_  
    # df_modelling = pd.get_dummies(df_modelling, columns=['cluster'], drop_first=True) 
    y = df_modelling.pop('fraud') 
    X = df_modelling.values
    ss = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    X_train_scaled = ss.fit_transform(X_train)
    X_test_scaled = ss.transform(X_test)
    if log_regression:
        print('Starting Logistic Regression...')
        lr = LogisticRegression(verbose=True, n_jobs=-1, class_weight='balanced')
        lr.fit(X_train_scaled, y_train)
        preds = lr.predict(X_train_scaled)
        holdout_preds = lr.predict(X_test_scaled)
        print(f"Training: F1: {f1_score(y_train, preds)}, Recall: {recall_score(y_train, preds)}, Accuracy: {lr.score(X_train_scaled, y_train)}, Precision: {precision_score(y_train, preds)}")
        print(f"Test: F1: {f1_score(y_test, holdout_preds)}, Recall: {recall_score(y_test, holdout_preds)}, Accuracy: {lr.score(X_test_scaled, y_test)}, Precision: {precision_score(y_test, holdout_preds)}")
    if randomforest:
        print("Starting Random Forest...")
        rf = RandomForestClassifier(class_weight='balanced', n_estimators=300, max_features=3, max_leaf_nodes=50, random_state=42, n_jobs=-2, oob_score=True)
        rf.fit(X_train_scaled, y_train)
        rfpreds = rf.predict(X_train_scaled)
        holdout_preds_rf = rf.predict(X_test_scaled)
        print(f"Training: \nF1: {f1_score(y_train, rfpreds)}, \nRecall: {recall_score(y_train, rfpreds)}, \nAccuracy: {rf.score(X_train_scaled, y_train)}, \nPrecision: {precision_score(y_train, rfpreds)}")
        print(f"Test: \nF1: {f1_score(y_test, holdout_preds_rf)}, \nRecall: {recall_score(y_test, holdout_preds_rf)}, \nAccuracy: {lr.score(X_test_scaled, y_test)}, \nPrecision: {precision_score(y_test, holdout_preds_rf)}")

    model = rf.fit(X, y) ## Final model created

    pickle_model = False
    if pickle_model:
        with open("model.pkl", 'wb') as f:
            pickle.dump(model, f)




