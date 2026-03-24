# EDA_EXP_4   Titanic Survival Analysis using Univariate Analysis

**Aim**

To perform univariate analysis on the Titanic dataset to understand the distribution and characteristics of individual variables (such as Age, Sex, Pclass, Fare, and Survived) and to draw insights about passengers and their survival patterns.


**Algorithm**

**1)Import Libraries:**

Load the required Python libraries (pandas, numpy, matplotlib, seaborn).

**2)Load the Dataset:**

Read the Titanic dataset from available sources (e.g., seaborn’s built-in Titanic dataset or a CSV file).

**3)Data Inspection:**

View the first few rows using head().

Get dataset summary using info() and describe().

**4)Handle Missing Data:**
Identify missing values using isnull().sum() and handle them appropriately (e.g., fill or drop).

**5)Univariate Analysis:**
Perform univariate analysis for each variable:

**Categorical Variables: (e.g., Sex, Pclass, Survived, Embarked)**
Use frequency tables and count plots.

**Numerical Variables: (e.g., Age, Fare)**
Use histograms, box plots, and summary statistics.

**6)Interpretation:**
Analyze distributions, central tendencies, and spread.
Identify patterns (e.g., more passengers in 3rd class, survival differences by gender).


**Program**

**Name :** B VIJAY KUMAR 

**Reg No.:** 212222230173

```

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

titanic = sns.load_dataset('titanic')

print("First 5 records:")
print(titanic.head())

print("\nShape of dataset (rows, columns):", titanic.shape)

print("\nMissing values per column:")
print(titanic.isnull().sum())

print("\nDistribution of 'sex':")
print(titanic['sex'].value_counts())
titanic['sex'].value_counts().plot(kind='bar', color=['skyblue', 'pink'])
plt.title("Distribution of Sex")
plt.xlabel("Sex")
plt.ylabel("Count")
plt.show()

survived_percent = titanic['survived'].mean() * 100
print(f"\nPercentage of passengers survived: {survived_percent:.2f}%")

print("\nPassenger class distribution:")
print(titanic['pclass'].value_counts())
sns.countplot(x='pclass', data=titanic, palette='coolwarm')
plt.title("Passenger Class Distribution")
plt.show()

print("\nPassengers embarked from each port:")
print(titanic['embarked'].value_counts())
sns.countplot(x='embarked', data=titanic, palette='Set2')
plt.title("Embarked Port Distribution")
plt.show()

print("\nDeck distribution:")
print(titanic['deck'].value_counts(dropna=False))
sns.countplot(x='deck', data=titanic, palette='Set3')
plt.title("Deck Distribution")
plt.show()

print("\nAge Distribution Statistics:")
print(f"Mean Age: {titanic['age'].mean():.2f}")
print(f"Median Age: {titanic['age'].median():.2f}")
print(f"Range of Age: {titanic['age'].max() - titanic['age'].min():.2f}")

sns.histplot(titanic['age'].dropna(), kde=True, bins=30, color='teal')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

sns.boxplot(x=titanic['fare'], color='orange')
plt.title("Boxplot for Fare")
plt.show()
print("\nObservation: The boxplot shows several outliers — passengers who paid much higher fares, likely from higher classes.")

sns.histplot(titanic['fare'], bins=30, kde=True, color='purple')
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Frequency")
plt.show()

mean_fare = titanic['fare'].mean()
median_fare = titanic['fare'].median()
print(f"\nMean Fare: {mean_fare:.2f}")
print(f"Median Fare: {median_fare:.2f}")

if mean_fare > median_fare:
    print("Inference: The distribution is right-skewed (positively skewed).")
else:
    print("Inference: The distribution is left-skewed (negatively skewed).")

print("\n--- INSIGHTS ---")
print("1. Most common passengers were males traveling in 3rd class.")
print("2. Higher survival observed among females, children, and upper-class passengers.")
print("3. Age distribution shows most passengers were between 20–40 years old.")
print("4. Fare distribution is highly right-skewed — a few passengers paid very high fares.")
print("5. Univariate analysis helps identify missing data, skewness, and variable types, guiding data cleaning and modeling steps.")

```




**Output**

<img width="680" height="287" alt="image" src="https://github.com/user-attachments/assets/28db9632-f14d-4d7b-b8b8-3a0807cc87a0" />

<img width="498" height="373" alt="image" src="https://github.com/user-attachments/assets/9fe1c3d2-148b-4cad-8151-af1d26af0f0b" />

<img width="571" height="496" alt="image" src="https://github.com/user-attachments/assets/842dc745-ae48-4a95-b6e7-3a38268b6a57" />

<img width="921" height="508" alt="image" src="https://github.com/user-attachments/assets/d4f42e2e-5630-4c06-83bb-4e2e7ba05143" />

<img width="933" height="565" alt="image" src="https://github.com/user-attachments/assets/36f41673-ba34-4023-8329-0b3cf1b94823" />

<img width="498" height="427" alt="image" src="https://github.com/user-attachments/assets/3ef76db5-e005-48c5-b624-a92fe46cc835" />

<img width="443" height="396" alt="image" src="https://github.com/user-attachments/assets/dfc99c5d-dd41-4222-b34c-14f663cb46a3" />

<img width="835" height="554" alt="image" src="https://github.com/user-attachments/assets/1cc27e1d-5133-4785-98d7-9033a5c88593" />


**Result**
From the univariate analysis:

Majority of passengers were male and in 3rd class.

Around 38% survived, majority being females and higher-class passengers.

Age is right-skewed with most passengers aged 20–40 years.

Fare distribution shows a few high outliers for 1st class passengers.

Thus, univariate analysis helps understand the distribution and spread of each individual feature in the Titanic dataset before moving to bivariate or multivariate analysis.

Visualization:
Plot appropriate charts for each variable using Matplotlib/Seaborn.
