import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

classe = train['Survived']
train = train.drop(['Survived','Name','Ticket', 'Cabin'],axis=1)
train = pd.get_dummies(train)
train.isnull().sum().sort_values(ascending=False)
train['Age'].fillna(train['Age'].mean(), inplace=True)
train['Fare'].fillna(train['Fare'].mean(), inplace=True)


test = test.drop(['Name','Ticket', 'Cabin'],axis=1)
test = pd.get_dummies(test)
test.isnull().sum().sort_values(ascending=False)
test['Age'].fillna(test['Age'].mean(), inplace=True)
test['Fare'].fillna(test['Fare'].mean(), inplace=True)


rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(train, classe)

pred = rfc.predict(test)
rfc.score(test, pred)

submission = pd.DataFrame()
submission['PassengerId'] = test['PassengerId']
submission['Survived'] = rfc.predict(test)

submission.to_csv('submission.csv', index=False)
