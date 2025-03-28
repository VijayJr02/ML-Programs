import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler

data = pd.read_csv("data.csv")
data.fillna({'Age': data['Age'].mean(), 'Salary': data['Salary'].median(), 'Gender': data['Gender'].mode()[0]}, inplace=True)

data['Gender'] = LabelEncoder().fit_transform(data['Gender'])
data = pd.concat([data, pd.get_dummies(data['Address'], drop_first=True)], axis=1).drop('Address', axis=1)

scalers = {'Age_normal': MinMaxScaler(), 'Salary_std': StandardScaler()}
for col, scaler in scalers.items():
    data[col] = scaler.fit_transform(data[[col.split('_')[0]]])
print(data)
