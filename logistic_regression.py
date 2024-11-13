import numpy as np #math for handling data
import matplotlib.pyplot as plt # # For plotting data (visualizing data)
from sklearn.model_selection import train_test_split #splitting data into train and test sets
from sklearn.linear_model import LogisticRegression #logistics regression model
from sklearn.metrics import accuracy_score #checking accuracy 

#generating random data from two categories
np.random.seed(0) #Setting a seed reproducibility (the same random data each time)
data_size = 100 #number of data points

#creating features for category 0
category_0 = np.random.normal(2, 0.5, (data_size, 2)) #centered around (2, 2) with a small random noise

#category 1
category_1 = np.random.normal(4, 0.5, (data_size, 2))

#combining data and creatig labels (0 for category_0, 1 for cat 1)

X = np.vstack((category_0, category_1)) #stacking both cats into 1 data set
y = np.hstack((np.zeros(data_size), np.ones(data_size)))

# V= verticl stack H for horizontal

#Plotting the data points
plt.figure(figsize=(8, 6)) #creating a new figure specified size
#Plotting cat - points
plt.scatter(category_0[:, 0], category_0[:, 1], color='blue', label='Category')
#Plotting category 1 points
plt.scatter(category_1[:, 0], category_1[:, 1], color='red', label='Category 1')


#adding labels and legends to the plot
plt.xlabel('Feature 1') #x-axis
plt.ylabel('Feature 2') #y-axis
plt.title('Data Visualization') #title for plot 
plt.legend() #Displaying the legend to identify each category
plt.show() #displaying the plot

#Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# creating the logistic regression model
model = LogisticRegression()
#training the model with the training data
model.fit(X_train, y_train)


#making predictions on the test data
y_pred = model.predict(X_test)

#calculating accuracy of the model
accuracy = accuracy_score(y_test, y_pred)


print("Model Accuracy:", accuracy)

#Creatign a mesh grid to plot the decision boundary
x_min, x_main = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, y_max, 0.1), np.arange(y_min, y_max, 0.1))

#using the model to predict on the grid

Z = model.predict(np.c_[xx.ravel, yy.ravel()]) #predicting categories for each point in the grid
Z = Z.reshape(xx.shape)

#plotting the decision boundary and data points
plt.figure(figsize=(8, 6))
plt.contourf(X[:, 0], X[:, 1], c=y, edgecolor='k', s=50, cmap=plt.cm.RdY1Bu)

#plotting the original data points
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=50, cmap=plt.cm.RdY1Bu)

#labels and title
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.totle('Logistic Regressing Decision Boundary')
plt.show()