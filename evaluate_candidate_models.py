import pandas as pd

# get my prepared dataset
X_train = pd.read_csv('./data/prepared_X_train.csv', index_col = False)
y_train = pd.read_csv('./data/prepared_y_train.csv', index_col = False)
y_train = y_train['Survived']

candidates = {}

# create candidates
from sklearn.ensemble import RandomForestClassifier
candidates['RandomForestClassifier'] = {
  'classifier': RandomForestClassifier(),
  'params': { 
    'n_estimators': [16] 
  }
}

from sklearn.svm import SVC
candidates['SVC'] = {
  'classifier': SVC(),
  'params': { 
    'kernel': ['linear'], 
    'C': [1] 
  }
}

# Applying Grid Search to find the best model and the best parameters
from sklearn.model_selection import GridSearchCV

for candidate in candidates:
  print(candidates[candidate]['params'])

  grid_search = GridSearchCV(
                  estimator = candidates[candidate]['classifier'], 
                  param_grid = candidates[candidate]['params'],
                  scoring = 'f1', 
                  cv = 10, 
                  n_jobs = 1)
  
  grid_search = grid_search.fit(X_train, y_train)
  print('best_accuracy = ')
  print(grid_search.best_score_)
  print('best_parameters = ')
  print(grid_search.best_params_)
