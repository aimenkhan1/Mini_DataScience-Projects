# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\ML\bank-full.csv" ,sep=';')
df.rename(columns={'y': 'loan_acceptance'}, inplace=True)
#starting rows for display
df.head()
df.info()
print("Missing values before cleaning:")
print(df.isnull().sum())

#Perform basic data exploration on features such as age, job, and marital status.
print(" Age Summary:")
print(df['age'].describe())
plt.hist(df['age'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()

print("Job Counts:")
print(df['job'].value_counts())
df['job'].value_counts().plot(kind='bar', color='orange', edgecolor='black')
plt.title('Job Distribution')
plt.xlabel('Job')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

print("Marital Status Counts:")
print(df['marital'].value_counts())
df['marital'].value_counts().plot(kind='bar', color='green', edgecolor='black')
plt.title('Marital Status Distribution')
plt.xlabel('Marital Status')
plt.ylabel('Count')
plt.show()

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

le = LabelEncoder()
df['loan_acceptance'] = le.fit_transform(df['loan_acceptance'])
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
for col in categorical_cols:
    df[col] = le.fit_transform(df[col])

# Features and target
X = df.drop(['loan_acceptance'], axis=1)
Y = df['loan_acceptance']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Predictions
y_pred = model.predict(X_test_scaled )

# Evaluation
print("\nClassification Report:\n", classification_report(y_test, y_pred))

new_df = pd.DataFrame({'Feature': X.columns, 'Coefficient': model.coef_[0]})
new_df = new_df.assign(Abs_Coeff=new_df['Coefficient'].abs()).sort_values(by='Abs_Coeff', ascending=False)
new_df
# %%
