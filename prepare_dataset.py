import pandas as pd

from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()


# get original datasets
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

datasets = [ train, test ]


# Fill Nulls
for dataset in datasets:
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)


# Drop unnecessary features
columns_to_drop = ['PassengerId', 'Ticket', 'Name', 'Fare', 'Cabin', 'Embarked']

for dataset in datasets:
    dataset.drop(columns_to_drop, axis = 1, inplace = True)


# create family size
for dataset in datasets:
    dataset["Family_Size"] = dataset['SibSp'] + dataset['Parch'] + 1
    dataset["Family_Size"].head()


# drop no of siblings, parents, children and spouses features
columns_to_drop = ['SibSp', 'Parch']

for dataset in datasets:
    dataset.drop(columns_to_drop, axis = 1, inplace = True)


# encode gender
for dataset in datasets:
    dataset[ 'Sex' ] = label_encoder.fit_transform(dataset[ 'Sex' ])


# split independent and dependent features
X_train = train.drop([ 'Survived' ], axis = 1)

y_train = train['Survived']
X_test = test


# scale the features
X_train = pd.DataFrame(standard_scaler.fit_transform(X_train))
X_test = pd.DataFrame(standard_scaler.transform(X_test))


print('--- Prepared datasets info ---')
for dataset in [ X_train, X_test ]:
    print()
    dataset.info()


# save as csv
X_train.to_csv('./data/prepared_X_train.csv')
y_train.to_csv('./data/prepared_y_train.csv')
X_test.to_csv('./data/prepared_X_test.csv')

print('\n --- Data is prepared ---')