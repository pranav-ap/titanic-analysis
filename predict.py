import pandas as pd


# get dataset
PassengerId = pd.read_csv('./data/test.csv', index_col = False)
PassengerId = PassengerId['PassengerId']

X_test = pd.read_csv('./data/prepared_X_test.csv', index_col = False)
X_train = pd.read_csv('./data/prepared_X_train.csv', index_col = False)
y_train = pd.read_csv('./data/prepared_y_train.csv', index_col = False)
y_train = y_train['Survived']


from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', C = 5)

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
y_pred = pd.DataFrame(y_pred)

submission = pd.DataFrame({
  'PassengerId': PassengerId,
  'Survived': y_pred[0]
})

submission.to_csv('./data/submission.csv', index = False, header = True)

