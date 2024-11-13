<!DOCTYPE html>
<html>
<head>
  <title>Logistic Regression for Binary Classification</title>
</head>
<body>

<h1>Logistic Regression for Binary Classification</h1>

<h2>Project Overview</h2>
<p>This project demonstrates how to implement and visualize a <strong>Logistic Regression model</strong> for binary classification using synthetic data. It covers generating and visualizing two distinct data categories, training a logistic regression model, evaluating its accuracy, and visualizing the decision boundary. Logistic Regression is a fundamental algorithm in machine learning, often used as an introduction to binary classification tasks.</p>

<h2>Code Walkthrough</h2>

<h3>1. Importing Libraries</h3>
<pre><code># Importing necessary libraries
import numpy as np  # For handling arrays and math operations
import matplotlib.pyplot as plt  # For plotting graphs
from sklearn.model_selection import train_test_split  # For splitting data into train and test sets
from sklearn.linear_model import LogisticRegression  # For logistic regression model
from sklearn.metrics import accuracy_score  # For checking model accuracy
</code></pre>
<p><strong>Explanation:</strong> We begin by importing the necessary libraries:</p>
<ul>
  <li><code>numpy</code>: To handle arrays and perform mathematical operations.</li>
  <li><code>matplotlib.pyplot</code>: For plotting data and visualizations.</li>
  <li><code>sklearn.model_selection.train_test_split</code>: To split the data into training and test sets.</li>
  <li><code>sklearn.linear_model.LogisticRegression</code>: To create a logistic regression model.</li>
  <li><code>sklearn.metrics.accuracy_score</code>: To measure the accuracy of our model on test data.</li>
</ul>

<h3>2. Data Generation</h3>
<pre><code># Data Generation
np.random.seed(0)
data_size = 100
category_0 = np.random.normal(2, 0.5, (data_size, 2))
category_1 = np.random.normal(4, 0.5, (data_size, 2))
X = np.vstack((category_0, category_1))
y = np.hstack((np.zeros(data_size), np.ones(data_size)))
</code></pre>
<p><strong>Explanation:</strong> We generate synthetic data for two categories:</p>
<ul>
  <li>We set the random seed to ensure reproducibility.</li>
  <li><code>data_size</code> defines the number of data points in each category.</li>
  <li><code>category_0</code> and <code>category_1</code>: Two clusters of data points generated with different centers (2 and 4), creating two distinct groups.</li>
  <li><code>X</code> combines both categories into one dataset, and <code>y</code> creates labels (0 and 1) for each category.</li>
</ul>

<h3>3. Data Visualization</h3>
<pre><code># Data Visualization
plt.figure(figsize=(8, 6))
plt.scatter(category_0[:, 0], category_0[:, 1], color='blue', label='Category 0')
plt.scatter(category_1[:, 0], category_1[:, 1], color='red', label='Category 1')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Data Visualization')
plt.legend()
plt.show()
</code></pre>
<p><strong>Explanation:</strong> We visualize the generated data to observe the two distinct categories:</p>
<ul>
  <li><code>plt.scatter</code> plots each category with a different color.</li>
  <li>Labels and legends are added for clarity.</li>
  <li><code>plt.show()</code> displays the plot, helping us understand how data points are distributed in each category.</li>
</ul>

<h3>4. Model Training</h3>
<pre><code># Model Training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
model = LogisticRegression()
model.fit(X_train, y_train)
</code></pre>
<p><strong>Explanation:</strong> We split the data into training and test sets and train the logistic regression model:</p>
<ul>
  <li><code>train_test_split</code> divides the dataset (70% training and 30% testing).</li>
  <li><code>LogisticRegression()</code> creates the model, and <code>model.fit()</code> trains it with the training data.</li>
</ul>

<h3>5. Prediction and Accuracy Evaluation</h3>
<pre><code># Prediction and Accuracy Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)
</code></pre>
<p><strong>Explanation:</strong> We make predictions and evaluate the model’s accuracy:</p>
<ul>
  <li><code>model.predict(X_test)</code> generates predictions on the test data.</li>
  <li><code>accuracy_score</code> compares predictions to actual values, and we print the accuracy.</li>
</ul>

<h3>6. Decision Boundary Visualization</h3>
<pre><code># Decision Boundary Visualization
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
<p><strong>Explanation:</strong> We visualize the decision boundary of the logistic regression model:</p>
<ul>
  <li><code>meshgrid</code> creates a grid of points covering the feature space.</li>
  <li>We predict the category for each grid point and reshape the predictions to match the grid.</li>
  <li><code>contourf</code> displays the decision boundary, showing where the model classifies each region.</li>
  <li>The scatter plot overlays the original data points on the decision boundary for comparison.</li>
</ul>

<h2>Follow Me</h2>
<p>Stay connected with my latest projects and insights:</p>
<ul>
  <li><strong>Bluesky</strong>: <a href="https://bsky.app/profile/kelvintechnical.bsky.social">kelvintechnical.bsky.social</a></li>
  <li><strong>X (formerly Twitter)</strong>: <a href="https://x.com/kelvintechnical">kelvintechnical</a></li>
  <li><strong>LinkedIn</strong>: <a href="https://www.linkedin.com/in/kelvin-r-tobias-211949219/">Kelvin R. Tobias</a></li>
</ul>

</body>
</html>
