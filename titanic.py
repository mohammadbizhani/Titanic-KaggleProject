# import the libraries
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Read the data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# check the missing values
train_data.isnull().sum()
test_data.isnull().sum()

# Choose features
features = ['Pclass','Sex','SibSp','Parch','Embarked','Age']

# make train datas
X_train = train_data[features]
X_train["Age"].fillna(train_data['Age'].mean(), inplace=True)
X_train = pd.get_dummies(X_train, columns=['Pclass','Sex','Embarked'], drop_first=True)
Y_train = train_data['Survived']

# make test datas just for X
X_test = test_data[features]
X_test = pd.get_dummies(test_data[features], columns=['Pclass','Sex','Embarked'], drop_first=True)
X_test["Age"].fillna(test_data['Age'].mean(), inplace=True)

RFC = RandomForestClassifier(random_state=1234)

list_cv = {'n_estimators':np.arange(20,50), 'max_depth':np.arange(1,10)}
RFC_CV = GridSearchCV(RFC, list_cv)

RFC_CV.fit(X_train, Y_train)

score = RFC_CV.score(X_train, Y_train)
print(RFC_CV.best_params_)
print(RFC_CV.best_score_)

predictions = RFC_CV.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission6.csv', index=False)

print("Your submission was successfully saved!")
