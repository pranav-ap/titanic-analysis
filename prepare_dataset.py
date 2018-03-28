import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

# get original datasets
train = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

datasets = [ train, test ]

# Fill Nulls
for dataset in datasets:    
    #complete missing age with median
    dataset['Age'].fillna(dataset['Age'].median(), inplace = True)

# Drop unnecessary features
columns_to_drop = ['PassengerId', 'Ticket', 'Fare', 'Cabin', 'Embarked']

for dataset in datasets:
    dataset.drop(
        columns_to_drop, 
        axis = 1, 
        inplace = True)

    dataset.info()


# create family size

# drop siblings, parents features

# create age
    
# encode gender
    
# scale the features

# save as csv

