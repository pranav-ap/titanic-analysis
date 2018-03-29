import pandas as pd

# get my prepared dataset
X_training = pd.read_csv('./data/prepared_X_train.csv')
y_training = pd.read_csv('./data/prepared_y_train.csv')


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
estimator_selection = EstimatorSelection()

# fit and find