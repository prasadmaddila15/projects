import numpy as np
import pandas as pd



#we import the data
pima = pd.read_csv("C:/Users/prasa/Downloads/diabetes.csv")

pima.head(10)


pima.shape

pima.info()


# Get the column names 
col_idx = pima.columns
col_idx

# Get row indices 
row_idx = pima.index
print(row_idx)

# Find data type for each attribute 
print("Data type of each attribute:")
pima.dtypes


# Generate statistical summary 
description = pima.describe()
print("Statistical summary of the data:\n")
description

class_counts = pima.groupby('Outcome').size() 
print("Class breakdown of the data:\n")
print(class_counts)

# Compute correlation matrix 
correlations = pima.corr(method = 'pearson') 
print(correlations)

skew = pima.skew() 
print("Skew of attribute distributions in the data:\n") 
print(skew) 



#univariate plots
# Import required package 
from matplotlib import pyplot
# set the figure size
pyplot.rcParams['figure.figsize'] = [20, 10];
# Draw histograms for all attributes 
pima.hist()
pyplot.show()

#density plot
# Density plots for all attributes
pima.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
pyplot.show()


#box and whisper plot
# Draw box and whisker plots for all attributes 
pima.plot(kind= 'box', subplots=True, layout=(3,3), sharex=False, sharey=False)
pyplot.show()

#multi variate plots





#QUESTION 1
#correlation matrix
# import required package 
import numpy as np
# plot correlation matrix
fig = pyplot.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,9,1)
ax.set_xticks(ticks)
ax.set_yticks(ticks)
names = pima.columns
# Rotate x-tick labels by 90 degrees
ax.set_xticklabels(names,rotation=90) 
ax.set_yticklabels(names)
pyplot.show()



import seaborn as sns

#OPTIONAL
# Distribution of features for patients with and without diabetes
sns.boxplot(x='Outcome', y='Glucose', data=pima)
pyplot.show()

sns.boxplot(x='Outcome', y='BMI', data=pima)
pyplot.show()

sns.boxplot(x='Outcome', y='BloodPressure', data=pima)
pyplot.show()



#QUESTION 2
# Correlation between BMI and glucose levels
sns.scatterplot(data=pima, x='BMI', y='Glucose', hue='Outcome')
pyplot.show()

# Correlation between BMI and glucose levels for patients with and without diabetes
sns.scatterplot(data=pima, x='BMI', y='Glucose', hue='Outcome')
pyplot.xlim(20, 60)
pyplot.ylim(50, 200)
pyplot.show()






#QUESTION 3
# Distribution of glucose levels
sns.histplot(data=pima, x='Glucose', kde=True)
pyplot.show()

# Distribution of glucose levels by diabetes status
sns.histplot(data=pima, x='Glucose', hue='Outcome', kde=True)
pyplot.show()



#QUESTION 4
# Differences in average number of pregnancies, BMI, and glucose levels by diabetes status
sns.barplot(data=pima, x='Outcome', y='Pregnancies')
pyplot.show
sns.barplot(data=pima, x='Outcome', y='BMI')
pyplot.show()
sns.barplot(data=pima, x='Outcome', y='Glucose')
pyplot.show()



