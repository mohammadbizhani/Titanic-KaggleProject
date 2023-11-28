# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 23:15:56 2023

@author: Mohamad
"""

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

# Read the data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


# check the head
train_data.head()


# check the shape
train_data.shape
test_data.shape


# check the missing values
train_data.isnull().sum()
test_data.isnull().sum()


# main check for each column
train_data.describe()




# Pie chart for Sex column
Sex_count = Counter(list(train_data['Sex']))

Sex_name = list(Sex_count.keys())
Sex_value = list(Sex_count.values())

plt.figure('Pie chart')
plt.pie(Sex_value, labels=Sex_name, colors=['r','b'], autopct='%.1f%%')
plt.title("Sex of passengers of the titnanic ship\n",fontweight = "bold")
plt.show()


# Pie chart for Embarked column
Emb_count = Counter(list(train_data['Embarked']))

Emb_name = list(Emb_count.keys())
Emb_value = list(Emb_count.values())

plt.figure('Pie chart for Embarked')
plt.pie(Emb_value, labels=Emb_name, colors=['g','c','y','b'], autopct='%.1f%%')
plt.title('Embarked\n',fontweight = "bold")
plt.show()


# pie chart for Survived/drown people
list_survived= list(train_data.iloc[:,1])

bins = 2

plt.figure('Histogram for Survived people')
plt.hist(list_survived, bins, histtype='bar', rwidth=0.9, color='#09c0ed')
plt.title('Survived People\n',fontweight = "bold")
plt.xlabel('Survived or drown')
plt.ylabel('Distribution')
bin_center = 0.5 * ( plt.hist(list_survived, bins, rwidth=0.9, color='#09c0ed')[1][1:] + plt.hist(list_survived, bins, rwidth=0.9, color='#09c0ed')[1][:-1])
plt.xticks(bin_center, ['Survived', 'drown'])
plt.show()

# Histogram chart for age
list_age = list(train_data.iloc[:,5].dropna())

bins = 10

plt.figure('Histogram for Age', figsize=(8,5))
plt.hist(list_age, bins, histtype='bar',rwidth=0.9, color='#FF5487')
plt.xlabel('Age')
plt.title('Age of passengers(nan values are droped)\n',fontweight = "bold")
plt.ylabel('Distribution')
bin_center = 0.5 * ( plt.hist(list_age, bins, rwidth=0.9, color='#FF5487')[1][1:] + plt.hist(list_age, bins, rwidth=0.9, color='#FF5487')[1][:-1])
plt.xticks(bin_center,['0-9','10-19','20-29','30-39','40-49','50-59','60-69','70-79','80-89','90-99'], rotation=45)
plt.show()


# Histogram chart for Pclass (Ticket Class)
list_pclass= list(train_data.iloc[:,2])

bins = 3

plt.figure('Histogram for Pclass(Ticket Class)')
plt.hist(list_pclass, bins, histtype='bar', rwidth=0.9, color='#7E42F5')
plt.title('Ticket class Distribution for passengers in Titanic\n',fontweight = "bold")
plt.xlabel('Pclass (Ticket Class)')
plt.ylabel('Distribution')
bin_center = 0.5 * ( plt.hist(list_pclass, bins, rwidth=0.9, color='#7E42F5')[1][1:] + plt.hist(list_pclass, bins, rwidth=0.9, color='#7E42F5')[1][:-1])
plt.xticks(bin_center, [1,2,3])
plt.show()


# Histogram chart for sibsp (of siblings / spouses aboard the Titanic)
list_sibsp = list(train_data.iloc[:,6])

bins = 9

plt.figure('Histogram for SibSp')
plt.hist(list_sibsp, bins, rwidth=0.9, color='#f5c842' , histtype='bar')
plt.title('Numbers of siblings/spouses aboard\n the Titanic for each passenger\n',fontweight='bold')
plt.xlabel('Numbers of siblings/spouses')
plt.ylabel('Distribution')
bin_center = 0.5 * ( plt.hist(list_sibsp, bins, rwidth=0.9, color='#f5c842')[1][1:] + plt.hist(list_sibsp, bins, rwidth=0.9, color='#f5c842')[1][:-1])
plt.xticks(bin_center, [0,1,2,3,4,5,6,7,8])
plt.show()

# Histogram chart for Parch (of parents / children aboard the Titanic)
list_parch = list(train_data['Parch'])

bins = 7

plt.figure('Histogram for Parch')
plt.hist(list_parch, bins, rwidth=0.9, color='#db1212' , histtype='bar')
plt.title('Numbers of of parents/children aboard\n the Titanic for each passenger\n',fontweight='bold')
plt.xlabel('Numbers of parents/children')
plt.text(2.8,540,'We have 5 person for 3 parch\n 4 person for 4 parch\n 5 person for 5 parch\n and only 1 person for 6 parch\n',size=12)
plt.ylabel('Distribution')
bin_center = 0.5 * ( plt.hist(list_parch, bins, rwidth=0.9, color='#db1212')[1][1:] + plt.hist(list_parch, bins, rwidth=0.9, color='#db1212')[1][:-1])
plt.xticks(bin_center, [0,1,2,3,4,5,6])
plt.show()


# Chart survived by Gender
plt.figure("Survived people by Gender")
sns.countplot(x='Survived', hue='Sex', data=train_data, palette='Set2')
plt.title('Survival distribution by Gender')
plt.xlabel('Survival status')
plt.ylabel('count')
plt.legend(loc='best', labels=['Male','Female'])
plt.show()


# Chart Survived by Pclass
plt.figure("Survived people by Pclass")
sns.countplot(x='Survived', hue='Pclass', data=train_data)
plt.title('Survival distribution by Plass(Ticket class)')
plt.xlabel('Survival status')
plt.ylabel('count')
plt.show()


# heatmap for train_data
plt.figure("Heatmap for correlations")
sns.heatmap(train_data.corr(), annot=True, cmap='YlGnBu')
plt.tight_layout()
plt.show()


# Pairplot by Pclass (Ticket class)
sns.pairplot(train_data.select_dtypes(['number']), hue='Pclass')
plt.show()





