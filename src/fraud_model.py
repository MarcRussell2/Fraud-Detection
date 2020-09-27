import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from cleaning import clean_training_dataframe, make_corpus, prep_corpus, get_top_features_cluster
from imblearn.over_sampling import RandomOverSampler 


def feat_imp_plots(full_df, rf):
    feature_names = full_df.columns
    feat_imp = pd.DataFrame({'feature_name':feature_names, 'feat_imp': rf.feature_importances_})
    feat_imp.sort_values('feat_imp',ascending=False,inplace=True)
    fig, ax = plt.subplots(1, figsize=(8,6))
    ax.bar(feat_imp['feature_name'].head(9), feat_imp['feat_imp'].head(9))
    # ax.invert_yaxis()
    ax.set_title('Random Forest Feature Importance')
    ax.set_xticklabels(labels=feature_names, rotation=45)
    plt.tight_layout()
    plt.savefig('images/feature_importance_oversample.png')
    plt.close()

if __name__ == "__main__":
    log_regression=True
    randomforest = True
    oversample = True
    
    df_modeling = pd.read_pickle('data/fraud_data.pkl')
    df_modeling.drop(columns='email_domain', axis=1, inplace=True)

    y = df_modeling.pop('fraud') 
    X = df_modeling.values
    
    if oversample:
        print("Imbalanced classes! Oversampling")
        ros = RandomOverSampler(random_state=0)  
        X, y = ros.fit_resample(X,y)  

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
        print(f"Training: F1: {f1_score(y_train, preds)}, Recall: {recall_score(y_train, preds)}, Accuracy: {lr.score(X_train_scaled, y_train)}, Precision: {precision_score(y_train, preds)}\n")
        print(f"Test: F1: {f1_score(y_test, holdout_preds)}, Recall: {recall_score(y_test, holdout_preds)}, Accuracy: {lr.score(X_test_scaled, y_test)}, Precision: {precision_score(y_test, holdout_preds)}\n\n\n")
    if randomforest:
        print("Starting Random Forest...")
        thresh_list = np.arange(0.3, 0.9, 0.1)

        rf = RandomForestClassifier(class_weight='balanced', n_estimators=300, max_features=3, max_leaf_nodes=50, random_state=42, n_jobs=-2, oob_score=True)
        rf.fit(X_train_scaled, y_train)
        predict_proba = rf.predict_proba(X_train_scaled)
        holdout_predict_proba = rf.predict_proba(X_test_scaled)
        # for i in thresh_list:
        #     # create labels for a given threshold
        #     rfpreds = (predict_proba[:,1] >= i).astype('int')
        #     holdout_preds_rf = (holdout_predict_proba[:,1] >= i).astype('int')
        #     # calculate and print results
        #     print(f"With a threshold of {i}:\n")
        #     print(f"Training: \nF1: {f1_score(y_train, rfpreds)}, \nRecall: {recall_score(y_train, rfpreds)}, \nAccuracy: {rf.score(X_train_scaled, y_train)}, \nPrecision: {precision_score(y_train, rfpreds)}\n")
        #     print(f"Test: \nF1: {f1_score(y_test, holdout_preds_rf)}, \nRecall: {recall_score(y_test, holdout_preds_rf)}, \nAccuracy: {rf.score(X_test_scaled, y_test)}, \nPrecision: {precision_score(y_test, holdout_preds_rf)}\n\n\n")
        
        feat_imp_plots(df_modeling, rf)
        score = roc_auc_score(y_test, holdout_predict_proba[:,1])
        print('ROC AUC: %.3f' % score)
    # model = rf.fit(X, y) ## Final model created

    # pickle_model = False
    # if pickle_model:
    #     with open("static/model.pkl", 'wb') as f:
    #         pickle.dump(model, f)

    # Best so far
