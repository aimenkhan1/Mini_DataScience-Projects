# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%
#cleaning the dataset
df = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\ML\Churn_Modelling.csv")
print("Missing values before cleaning:")
print(df.isnull().sum())

df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True)

#encoding
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender']) 

from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse_output=False, drop='first')
onehot_sklearn = onehot_encoder.fit_transform(df[['Geography']])
onehot_df = pd.DataFrame(onehot_sklearn, columns=onehot_encoder.get_feature_names_out(['Geography']))
print("\nUsing sklearn OneHotEncoder:")
print(onehot_df.head())

df = df.drop('Geography', axis=1)
df = pd.concat([df, onehot_df], axis=1)

print(df.head())

#model selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

#spilting test and train
X = df.drop(['Exited'], axis=1)
Y = df['Exited']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)


model = LogisticRegression()
model.fit(x_train, y_train)

coefficients = model.coef_[0]
features = x_train.columns

newdf = pd.DataFrame({'Feature': features,'Importance': abs(coefficients)})
newdf = newdf.sort_values(by='Importance', ascending=False)

#model evaluation
y_pred = model.predict(x_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#another model selection(no need of scaling here)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

X = df.drop(['Exited'], axis=1)
Y = df['Exited']

model = RandomForestClassifier(random_state=0)
model.fit(x_train, y_train)

importance = model.feature_importances_

newdf = pd.DataFrame({'Feature': x_train.columns, 'Importance': importance})
newdf = newdf.sort_values(by='Importance', ascending=False)
print(newdf)

#model evaluation
y_pred = model.predict(x_test)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#visualizing importance
plt.barh(newdf['Feature'],newdf['Importance'])
plt.xlabel('Importance')
plt.title('Feature Importance from Random Forest')
plt.tight_layout()
plt.show()
# %%
