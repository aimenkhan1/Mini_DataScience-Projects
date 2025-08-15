# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#cleaning the dataset
df = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\ML\train.csv")
print("Missing values before cleaning:\n", df.isnull().sum())

df = df.sort_values(by='Loan_ID')
df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['LoanAmount'].fillna(df['LoanAmount'].median(),inplace=True)
df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)

print("\nMissing values after cleaning:\n", df.isnull().sum())
df.head()

#Visualizing the dataset
plt.hist(df['LoanAmount'], bins=30)
plt.title('Distribution of Loan Amount')
plt.xlabel('Loan Amount')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

df.groupby('Education')['LoanAmount'].mean().plot(kind='bar')
plt.title("Average Loan Amount by Education")
plt.ylabel("Average Loan Amount")
plt.xlabel("Education")
plt.tight_layout()
plt.show()

sns.scatterplot(data=df, x='ApplicantIncome', y='LoanAmount', hue='Education')
plt.title("Loan Amount vs Applicant Income by Education")
plt.xlabel("Applicant Income")
plt.ylabel("Loan Amount")
plt.tight_layout()
plt.show()


#Training a classification model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


le = LabelEncoder()
df['Loan_Status'] = le.fit_transform(df['Loan_Status']) 
df['Education'] = le.fit_transform(df['Education'])  

df = pd.get_dummies(df, columns=['Gender', 'Married', 'Self_Employed', 'Property_Area'], drop_first=True)

df['Dependents'] = df['Dependents'].replace('3+', 3)
df['Dependents'] = df['Dependents'].astype(float)

df = df.dropna()

X = df.drop(['Loan_ID','Loan_Status'], axis=1)
Y = df['Loan_Status']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

#Evaluating the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))


# %%
