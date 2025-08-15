# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %%
df = pd.read_csv(r"C:\Users\hp\OneDrive\Desktop\ML\insurance.csv")
#starting rows for display
df.head()
df.info()
print("Missing values before cleaning:")
print(df.isnull().sum()) #already cleaned 

#encoding
df['smoker'] = df['smoker'].map({'yes': 1, 'no': 0})
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['sex'] = le.fit_transform(df['sex']) 

from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse_output=False, drop='first') 
onehot_sklearn = onehot_encoder.fit_transform(df[['region']])
onehot_df = pd.DataFrame(onehot_sklearn, columns=onehot_encoder.get_feature_names_out(['region']))
print("\nUsing sklearn OneHotEncoder:")
print(onehot_df.head())

df = df.drop('region', axis=1)
df = pd.concat([df, onehot_df], axis=1)
df

#Visualize how BMI, age, and smoking status impact insurance charges.
plt.figure(figsize=(6,4))
sns.scatterplot(data=df,x='age',y='charges',hue='smoker',palette='icefire', alpha=0.5)
plt.title("Age vs Insurance Charges (Smoker vs Non-Smoker)")
plt.show()

# 2. BMI vs Charges
plt.figure(figsize=(6,4))
sns.scatterplot(data=df, x='bmi', y='charges', hue='smoker', palette='icefire', alpha=0.7)
plt.title("BMI vs Insurance Charges (Smoker vs Non-Smoker)")
plt.show()

# Train a Linear Regression model to predict charges.
#model selection
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
correlation = df.corr(numeric_only=True)
print(correlation['charges'].sort_values(ascending=False))
x = df.drop('charges', axis=1)
y = df['charges']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train,y_train)

print("Coefficients:", model.coef_)
print("intercept:",model.intercept_)

#metric & evaluation
from sklearn.metrics import mean_squared_error, root_mean_squared_error
y_pred = model.predict(x_test)
#Error evaluation using MAE and RMSE
from sklearn.metrics import mean_absolute_error
print("MAE:", mean_absolute_error(y_test, y_pred))
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)

# %%


