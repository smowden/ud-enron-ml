#!/usr/bin/python

import sys
import pickle
import numpy as np
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
import pandas as pd

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
df = pd.DataFrame().from_dict(data_dict).T

df = df.replace("NaN", np.nan)
df = df.drop("THE TRAVEL AGENCY IN THE PARK")
df = df.drop("TOTAL")

df['restricted_stock'].fillna(0, inplace=True)
df['restricted_stock_deferred'].fillna(0, inplace=True)
df['to_messages'].fillna(0, inplace=True)
df['from_messages'].fillna(0, inplace=True)

df['undeferred_restricted_stock'] = df['restricted_stock'] + df['restricted_stock_deferred']


def calc_frac_poi(row):
    frac_from = 0.
    frac_to = 0.
    frac_shared = 0.
    if row['to_messages'] != 0:
        frac_from = float(row["from_poi_to_this_person"]) / row['to_messages']
        frac_shared = float(row["shared_receipt_with_poi"]) / row['to_messages']
    if row['from_messages'] != 0:
        frac_to = float(row['from_this_person_to_poi']) / row["from_messages"]

    row['fraction_to_poi'] = frac_to
    row['fraction_from_poi'] = frac_from
    row['fraction_shared_with_poi'] = frac_shared
    return row


df = df.apply(calc_frac_poi, axis=1)
df[df["from_poi_to_this_person"] > 0].sort_values("fraction_from_poi").tail(5)

financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive', 'restricted_stock', 'director_fees']
email_features = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
all_features = financial_features + email_features

features_list = ['poi'] + all_features + ['undeferred_restricted_stock', 'fraction_from_poi', 'fraction_to_poi', 'fraction_shared_with_poi']

my_dataset = df.replace(np.nan, 0).T.to_dict() # replace nans with zeroes so feature format works
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)


cv = StratifiedKFold(n_splits=6, random_state=42)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.svm import SVC

pipe = Pipeline([('standartize', StandardScaler()), ('select_features', SelectKBest(f_classif)),
                 ('reduce_dim', PCA()), ('classifier', SVC())])

param_grid = [
    {'select_features__k': range(1, len(features_list)-1)},
    {
        'reduce_dim': [PCA()],
        'reduce_dim__whiten': [True, False]
    },
    {
        'classifier': [SVC()],
        'classifier__C': [1, 10, 100, 1000],
    },
    {
        'classifier': [GaussianNB()]
    }
]

grid = GridSearchCV(pipe, cv=cv, param_grid=param_grid, scoring='recall')
grid.fit(features, labels)
print "best score: {0}".format(grid.best_score_)

selected_features = grid.best_estimator_.named_steps["select_features"].scores_

feature_scores = sorted(zip(features_list[1:], selected_features, grid.best_estimator_.named_steps["select_features"].get_support()), key=lambda t: t[1], reverse=True)

new_features_list = ['poi'] + [feat[0] for feat in feature_scores if feat[2]]
print new_features_list


dump_classifier_and_data(grid.best_estimator_, my_dataset, new_features_list)

