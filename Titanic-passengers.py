import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

train_data = pd.read_csv('Titanic-Dataset.csv')
test_data = train_data

print(train_data.head())

def preprocess_data(df):
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Fare'] = df['Fare'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

    df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked_Q', 'Embarked_S']

    return df[features]

X = preprocess_data(train_data)
y = train_data['Survived']

X_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X_train, y_train)

y_pred = model.fit(X_train, y_train).predict(x_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"\nТочность модели (Accuracy) на валидации: {accuracy:.4f}")
print("\nОтчет о классификации:")
print(classification_report(y_val, y_pred))