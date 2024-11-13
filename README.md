<!DOCTYPE html>
<html>
<head>
</head>
<body>

<h1>Logistic Regression for Binary Classification</h1>

<h2>Project Overview</h2>
<p>This project demonstrates how to implement and visualize a <strong>Logistic Regression model</strong> for binary classification using synthetic data. It covers generating and visualizing two distinct data categories, training a logistic regression model, evaluating its accuracy, and visualizing the decision boundary. Logistic Regression is a fundamental algorithm in machine learning, often used as an introduction to binary classification tasks.</p>

<h2>What I Learned</h2>
<p>In building this project, I learned the following:</p>
<ul>
  <li><strong>Data Generation</strong>: How to generate synthetic data for binary classification using NumPy.</li>
  <li><strong>Data Visualization</strong>: How to plot data points and visualize categories using Matplotlib.</li>
  <li><strong>Model Training</strong>: How to create and train a logistic regression model using Scikit-Learn.</li>
  <li><strong>Accuracy Evaluation</strong>: How to evaluate a model’s accuracy on test data.</li>
  <li><strong>Decision Boundary Visualization</strong>: How to plot the decision boundary to understand the model’s classification regions.</li>
</ul>

<h2>Why This Project is Important</h2>
<p>Understanding Logistic Regression is crucial for anyone starting in machine learning or data science, as it:</p>
<ul>
  <li>Provides a foundation for more complex classification models.</li>
  <li>Helps understand the concept of decision boundaries in data classification.</li>
  <li>Demonstrates how data visualization aids in interpreting model results.</li>
</ul>

<h2>Why This Project is Great for a Machine Learning Portfolio</h2>
<p>Logistic Regression is widely used in various industries for binary classification tasks. By showcasing this project, I can demonstrate my foundational knowledge of machine learning and my ability to apply it to real-world data. This project is a solid base for more advanced machine learning projects.</p>

<h2>Code Walkthrough</h2>

<pre><code># Importing necessary libraries
import numpy as np  # For handling arrays and math operations
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.linear_model import LogisticRegression  # For logistic regression model
from sklearn.metrics import accuracy_score  # For checking model accuracy

# Data Generation
np.random.seed(0)
data_size = 100
category_0 = np.random.normal(2, 0.5, (data_size, 2))
category_1 = np.random.normal(4, 0.5, (data_size, 2))
X = np.vstack((category_0, category_1))
y = np.hstack((np.zeros(data_size), np.ones(data_size)))

# Data Visualization
plt.figure(figsize=(8, 6))
plt.scatter(category_0[:, 0], category_0[:, 1], color='blue', label='Category 0')
plt.scatter(category_1[:, 0], category_1[:, 1], color='red', label='Category 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Data Visualization')
plt.legend()
plt.show()

# Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = LogisticRegression()
model.fit(X_train, y_train)

# Prediction and Accuracy Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Decision Boundary Visualization
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=50, cmap=plt.cm.RdYlBu)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()
</code></pre>

<h2>Follow Me</h2>
<p>Stay connected with my latest projects and insights:</p>
<ul>
  <li><strong>Bluesky</strong>: <a href="https://bsky.app/profile/kelvintechnical.bsky.social">kelvintechnical.bsky.social</a></li>
  <li><strong>X (formerly Twitter)</strong>: <a href="https://x.com/kelvintechnical">kelvintechnical</a></li>
  <li><strong>LinkedIn</strong>: <a href="https://www.linkedin.com/in/kelvin-r-tobias-211949219/">Kelvin R. Tobias</a></li>
</ul>

</body>
</html>
