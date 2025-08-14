# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print('--------DevelopersHub Corporation TASK:01--------')
df = sns.load_dataset('iris')  
print(df)

print(f'First 5 rows:{df.head()}')
print(f'Shape: {df.shape}')
print(f'Name of columns:{df.columns}')


plt.scatter(df.sepal_width,df.petal_width,color='green',marker='o',label='Iris Data')
plt.xlabel('sepal width')
plt.ylabel('petal_width')
plt.title("Simple relationship btw sepal_width and petal_width using matplotlib")
plt.legend()
plt.tight_layout()
plt.show()

sns.scatterplot(data=df,x='sepal_length',y='petal_length',style='species',size='species',sizes=(10,50),palette='Oranges',alpha=0.5,markers=['o','>','*'])
plt.title("Sepal vs petal length by species using seaborn")
plt.tight_layout()
plt.show()


plt.hist(df.petal_length,bins=5,color='purple',edgecolor='black')
plt.xlabel('Petal Length Range')
plt.ylabel('Number of Flowers ')
plt.title('Petal_length distribution using matplotlib:')
plt.tight_layout()
plt.show()

sns.displot(data=df, x='sepal_length',kde=True,rug=True,color='red')
plt.title('Sepal_Length distribution using seaborn:')
plt.tight_layout()
plt.show()


plt.figure(figsize=(8,6))
speciesgroups = [df[df['species'] == i]['petal_width'] for i in df['species'].unique()]
plt.boxplot(speciesgroups, labels=df['species'].unique())
plt.title('Petal_width distribution by species using matplotlib')
plt.xlabel('Species')
plt.ylabel('Petal Width')
plt.tight_layout()
plt.show()

sns.boxplot(x='species', y='sepal_width', data=df, palette='pastel')
plt.title('Sepal_width distribution by species using seaborn')
plt.tight_layout()
plt.show()


# %%
