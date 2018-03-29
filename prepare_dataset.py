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


train.info()

# split independent and dependent features
X_train = train.drop([ 'Survived' ], axis = 1)
print(type(X_train))

y_train = train['Survived']

X_test = test


# scale the features
X_train = standard_scaler.fit_transform(X_train)
X_test = standard_scaler.transform(X_train)

#X_train.info()
#for dataset in [ X_train, X_test ]:
 #   dataset.info()

# save as csv