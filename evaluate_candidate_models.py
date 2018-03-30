import pandas as pd

# get my prepared dataset
X_train = pd.read_csv('./data/prepared_X_train.csv', index_col = False)
y_train = pd.read_csv('./data/prepared_y_train.csv', index_col = False)


# import candidate models and create the 2 dictionaries
from sklearn.ensemble import RandomForestClassifier
random_forest_classifier_params = {
  'n_estimators': [16, 32]
}

from sklearn.svm import SVC
svc_params = [
  { 'kernel': ['linear'], 'C': [1, 10] },
  { 'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001] }
]


candidate_models = {
  'RandomForestClassifier': RandomForestClassifier(),
  'SVC': SVC()
}

candidate_params = {
  'RandomForestClassifier': random_forest_classifier_params,
  'SVC': svc_params
}


from helpers import EstimatorSelection
estimator_selection = EstimatorSelection(candidate_models, candidate_params)
estimator_selection.fit(X_train, y_train, scoring = 'f1', n_jobs = 2)


estimator_selection.score_summary(sort_by = 'max_score')