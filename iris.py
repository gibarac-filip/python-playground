from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_predict, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import randint
import pickle
import joblib
import os

os.getcwd()
os.chdir('Desktop/python-playground')
# Load the Iris dataset
iris = load_iris()

# Access the data, target, feature names, etc.
data = iris.data
target = iris.target
feature_names = iris.feature_names
target_names = iris.target_names
iris_data = pd.DataFrame(data, columns=feature_names)
iris_data.shape
iris_data.size
iris_data.info()
iris_data.describe()

# Check for missing values
iris_data['species'] = target_names[target]
missing_iris = iris_data.isna().any(axis=1)

# Drop rows with missing values
iris_no_missing = iris_data.dropna()

# Drop duplicate rows
iris_no_duplicates = iris_no_missing.drop_duplicates()

# Set up the figure and axes
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
plt.subplots_adjust(hspace=0.5)

# Create histograms for each feature
for i, ax in enumerate(axes.flat):
    sns.histplot(iris_no_duplicates.iloc[:, i], ax=ax, kde=True)
    ax.set_title(f'Histogram of {feature_names[i]}')
    ax.set_xlabel('Value')
    ax.set_ylabel('Frequency')

# Show the histograms
plt.tight_layout()
plt.show()

# Create box plots for each feature
plt.figure(figsize=(10, 6))
sns.boxplot(data=iris_no_duplicates)
plt.title('Box Plot of Features')
plt.ylabel('Value')
plt.xticks(rotation=45)
plt.show()

# Create scatterplots to visualize feature correlations
sns.set(style="ticks")
sns.pairplot(iris_no_duplicates)
plt.suptitle('Scatterplots of Feature Correlations', y=1.02)
plt.show()

# Create a correlation matrix and heatmap
correlation_matrix = iris_no_duplicates.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix')
plt.show()

# Create a FacetGrid for plotting
g = sns.FacetGrid(iris_no_duplicates, col='species', height=4, aspect=1.2)
g.map_dataframe(sns.scatterplot, x='sepal length (cm)', y='sepal width (cm)')
g.set_axis_labels("Sepal Length (cm)", "Sepal Width (cm)")
g.set_titles(col_template="{col_name} species")
g.fig.suptitle("Scatterplots of Sepal Length vs. Sepal Width by Species", y=1.02)
plt.tight_layout()
plt.show()

#revert to original dataset
iris_data = pd.DataFrame(data, columns=feature_names)
iris_data['species'] = target
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=17)

scaler = StandardScaler()
# Fit and transform the training data using the scaler
X_train_standardized = scaler.fit_transform(X_train)

# Transform the testing data using the same scaler
X_test_standardized = scaler.transform(X_test)

knn = KNeighborsClassifier()
logreg = LogisticRegression()
rf = RandomForestClassifier()
svm = SVC()
voting_ensemble = VotingClassifier(estimators=[('logreg', logreg), ('rf', rf), ('svm', svm), ('knn', knn)], voting='hard')
stacked_ensemble = StackingClassifier(estimators=[('logreg', logreg), ('rf', rf), ('svm', svm), ('knn', knn)], final_estimator=LogisticRegression())

# Train models against standardized data
knn.fit(X_train_standardized, y_train)
logreg.fit(X_train_standardized, y_train)
rf.fit(X_train_standardized, y_train)
svm.fit(X_train_standardized, y_train)
voting_ensemble.fit(X_train_standardized, y_train)
stacked_ensemble.fit(X_train_standardized, y_train)

# Predictions against standardized data
knn_std_pred = knn.predict(X_test_standardized)
logreg_std_pred = logreg.predict(X_test_standardized)
rf_std_pred = rf.predict(X_test_standardized)
svm_std_pred = svm.predict(X_test_standardized)
voting_ensemble_std_pred = voting_ensemble.predict(X_test_standardized)
stacked_ensemble_std_pred = stacked_ensemble.predict(X_test_standardized)

# Evaluate models against standardized data
knn_std_accuracy = accuracy_score(y_test, knn_std_pred)
logreg_std_accuracy = accuracy_score(y_test, logreg_std_pred)
rf_std_accuracy = accuracy_score(y_test, rf_std_pred)
svm_std_accuracy = accuracy_score(y_test, svm_std_pred)
voting_ensemble_accuracy = accuracy_score(y_test, voting_ensemble_std_pred)
stacked_ensemble_accuracy = accuracy_score(y_test, stacked_ensemble_std_pred)
print("Current model rankings: ", [(knn, knn_std_accuracy), (logreg, logreg_std_accuracy), (rf, rf_std_accuracy), (svm, svm_std_accuracy)])
print("Voting Classifier Accuracy:", voting_ensemble_accuracy)
print("Stacked Classifier Accuracy:", stacked_ensemble_accuracy)

# Initialize the GridSearchCV object
"""
Method: Exhaustively searches all possible combinations of hyperparameters from a predefined grid.
Pros: Guarantees that the best combination will be found within the specified grid.
Cons: Can be computationally expensive, especially when the hyperparameter space is large or when there are many hyperparameters.
Usage: Best suited when you have a relatively small number of hyperparameters and you want to perform an exhaustive search over all combinations.
"""

# Define the hyperparameter grid for KNN
param_grid_knn = {
    'n_neighbors': [2, 3, 5, 7, 11],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'weights': ['uniform', 'distance'],
    'p': [1, 2, 4]
}

# Initialize the GridSearchCV object for KNN
grid_search_knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid_knn, cv=5, n_jobs=-1)

# Fit the model and find the best parameters
grid_search_knn.fit(X_train_standardized, y_train)

# Print the best parameters and the corresponding accuracy
print("Best Parameters for KNN:", grid_search_knn.best_params_)
print("Best Accuracy for KNN:", grid_search_knn.best_score_)

# Refine the SVC model further
param_grid_svc = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

grid_search_svc = GridSearchCV(estimator=SVC(), param_grid=param_grid_svc, cv=5, n_jobs=-1)

# Fit the model and find the best parameters
grid_search_svc.fit(X_train_standardized, y_train)

# Print the best parameters and the corresponding accuracy
print("Best Parameters for SVC:", grid_search_svc.best_params_)
print("Best Accuracy for SVC:", grid_search_svc.best_score_)

# Define the hyperparameter grid for Logistic Regression
param_grid_logreg = {
    'C': [0.1, 1, 10],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

# Initialize the GridSearchCV object for Logistic Regression
grid_search_logreg = GridSearchCV(estimator=LogisticRegression(max_iter=10000), param_grid=param_grid_logreg, cv=5, n_jobs=-1)

# Fit the model and find the best parameters
grid_search_logreg.fit(X_train_standardized, y_train)

# Print the best parameters and the corresponding accuracy
print("Best Parameters for Logistic Regression:", grid_search_logreg.best_params_)
print("Best Accuracy for Logistic Regression:", grid_search_logreg.best_score_)

# Define the hyperparameter grid
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize the GridSearchCV object
grid_search_rf = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5, n_jobs=-1)

# Fit the model and find the best parameters
grid_search_rf.fit(X_train_standardized, y_train)

# Print the best parameters and the corresponding accuracy
print("Best Parameters:", grid_search_rf.best_params_)
print("Best Accuracy:", grid_search_rf.best_score_)

# Initialize the RandomizedSearchCV object
"""
    Method: Randomly samples a fixed number of combinations from the hyperparameter space according to specified distributions.
    Pros: More efficient in terms of computation time, as it explores only a subset of the hyperparameter space.
    Cons: May not guarantee that the best combination will be found, but it can still provide good results.
    Usage: Suitable when the hyperparameter space is large or when you want to perform a more exploratory search without exhaustively evaluating all combinations.
"""
# Define the hyperparameter grid for KNN
param_random_knn = {
    'n_neighbors': [2, 3, 5, 7, 11],
    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
    'weights': ['uniform', 'distance'],
    'p': [1, 2, 4]
}

# Initialize the GridSearchCV object for KNN
random_search_knn = RandomizedSearchCV(estimator=KNeighborsClassifier(), param_distributions=param_random_knn, cv=5, n_jobs=-1)

# Fit the model and find the best parameters
random_search_knn.fit(X_train_standardized, y_train)

# Print the best parameters and the corresponding accuracy
print("Best Parameters for KNN:", random_search_knn.best_params_)
print("Best Accuracy for KNN:", random_search_knn.best_score_)

# Define the hyperparameter distributions
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [None, 10, 20],
    'min_samples_split': randint(2, 11)
}

random_search_svc = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=param_dist, n_iter=10, cv=5, n_jobs=-1, random_state=17)

# Fit the model and find the best parameters
random_search_svc.fit(X_train_standardized, y_train)

# Print the best parameters and the corresponding accuracy
print("Best Parameters:", random_search_svc.best_params_)
print("Best Accuracy:", random_search_svc.best_score_)

# Define the hyperparameter grid for Logistic Regression
param_random_logreg = {
    'C': [0.1, 1, 10],
    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
}

# Initialize the RandomizedSearchCV object for Logistic Regression
random_search_logreg = RandomizedSearchCV(estimator=LogisticRegression(max_iter=10000), param_distributions=param_random_logreg, cv=5, n_jobs=-1)

# Fit the model and find the best parameters
random_search_logreg.fit(X_train_standardized, y_train)

# Print the best parameters and the corresponding accuracy
print("Best Parameters for Logistic Regression:", random_search_logreg.best_params_)
print("Best Accuracy for Logistic Regression:", random_search_logreg.best_score_)

# Define the hyperparameter grid
param_random_rf = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize the RandomizedSearchCV object
random_search_rf = RandomizedSearchCV(estimator=RandomForestClassifier(), param_distributions=param_random_rf, cv=5, n_jobs=-1)

# Fit the model and find the best parameters
random_search_rf.fit(X_train_standardized, y_train)

# Print the best parameters and the corresponding accuracy
print("Best Parameters:", random_search_rf.best_params_)
print("Best Accuracy:", random_search_rf.best_score_)

# Select the best model based on Grid Search validation results
best_grid_model = max([(knn, grid_search_knn.best_score_), (logreg, grid_search_logreg.best_score_), (rf, grid_search_rf.best_score_), (svm, grid_search_svc.best_score_)],
                 key=lambda x: x[1])[0]

# Select the best model based on Random Search validation results
best_random_model = max([(knn, random_search_knn.best_score_), (logreg, random_search_logreg.best_score_), (rf, random_search_rf.best_score_), (svm, random_search_svc.best_score_)],
                 key=lambda x: x[1])[0]

# Evaluate the best model on the test set
# Best parameters from the hyperparameter tuning
# best_C = grid_search_logreg.best_params_['C'] # Replace with the best value from your results
# best_solver = grid_search_logreg.best_params_['solver']  # Replace with the best value from your results
best_n_neighbors = grid_search_knn.best_params_['n_neighbors']
best_weights = grid_search_knn.best_params_['weights']
best_p = grid_search_knn.best_params_['p']

# Create a new Logistic Regression model with the best parameters
# best_logreg_model = LogisticRegression(C=best_C, solver=best_solver)
best_knn_model = KNeighborsClassifier(n_neighbors=best_n_neighbors, weights=best_weights, p=best_p)

# Train the model on the entire training dataset
# best_logreg_model.fit(X_train_standardized, y_train)
best_knn_model.fit(X_train_standardized, y_train)

# Predictions on the test set using the tuned model
# best_logreg_pred = best_logreg_model.predict(X_test_standardized)
best_knn_pred = best_knn_model.predict(X_test_standardized)

# Evaluate the model's accuracy
# best_logreg_accuracy = accuracy_score(y_test, best_logreg_pred)
# print("Best Logistic Regression Model Accuracy:", best_logreg_accuracy)
best_knn_accuracy = accuracy_score(y_test, best_knn_pred)
# (true_positive + true_negative) / (all results)
print("Best KNN Model Accuracy:", best_knn_accuracy)

# Compute precision
precision = precision_score(y_test, best_knn_pred, average='weighted')  # Can use 'micro', 'macro', etc. instead of 'weighted'
# true_positive / (true_positive + false_positive)
# "Of all instances predicted as positive, how many were actually positive?"
print("Precision:", precision)

# Compute recall
recall = recall_score(y_test, best_knn_pred, average='weighted')  # Can use 'micro', 'macro', etc. instead of 'weighted'
# true_positive / (true_positive + false_negative)
# "Of all actual positive instances, how many were correctly predicted?"
print("Recall:", recall)

# Compute F1-score
f1 = f1_score(y_test, best_knn_pred, average='weighted')  # Can use 'micro', 'macro', etc. instead of 'weighted'
# (2 * precision * recall) / (precision + recall)
# harmonic mean of precision and recall
print("F1-score:", f1)

"""
Why does the accuracy of the model 'dip' from 0.975 to 0.933?

Randomness and Variability: 
    Test set accuracy can vary due to randomness in the data split and the sensitivity of the model to small changes. 
    Ensure that you're using the same random seed for both the hyperparameter tuning and the final evaluation on the test set to minimize this variability.

Data Drift: 
    If the distribution of the test set is significantly different from the training set or the validation set, it can lead to a drop in performance. 
    Ensure that the test data is representative of the training data.

Overfitting: 
    While hyperparameter tuning aims to prevent overfitting, it's still possible that the model has learned noise from the training set. 
    Regularization techniques like L1 or L2 regularization can help mitigate overfitting.

Feature Scaling: 
    Ensure that the feature scaling applied during training is also applied to the test data. 
    You should use the same StandardScaler instance used during training to standardize your test data.

Complexity: 
    Sometimes, hyperparameter tuning might lead to models that are more complex and prone to overfitting. 
    Check whether the final model might benefit from being simpler.

Imbalanced Classes: 
    If the classes are imbalanced, accuracy might not be the best metric to evaluate model performance. 
    Consider using other metrics like precision, recall, or F1-score.

Inherent Limitations: 
    There might be inherent limitations in the dataset that the model can't overcome.
    For instance, if the data is not well-separated, no model can achieve perfect accuracy.

Ensemble Methods: 
    If the differences are small, you might consider using ensemble methods to combine predictions from multiple models, potentially boosting the overall performance.

Remember that a drop in accuracy doesn't necessarily mean the model is ineffective. It's important to consider a variety of evaluation metrics, understand the context of your problem, and perhaps perform further analysis to gain insights into why the difference exists.    
"""
# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, best_knn_pred)

# Visualize the confusion matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Perform 5-fold cross-validation for accuracy
cv_scores = cross_val_score(best_knn_model, iris_data.iloc[0:,0:4], target, cv=5, scoring='accuracy')

# Print the cross-validation scores for each fold
print("Cross-Validation Scores:", cv_scores)

# Calculate the average cross-validation accuracy
average_accuracy = cv_scores.mean()
print("Average Cross-Validation Accuracy:", average_accuracy)

# Obtain cross-validated predictions
cv_predictions = cross_val_predict(best_knn_model, iris_data.iloc[0:,0:4], target, cv=5)

# Calculate accuracy using the cross-validated predictions and true labels
cv_accuracy = accuracy_score(target, cv_predictions)
print("Cross-Validated Accuracy:", cv_accuracy)

# Save the fitted model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(best_knn_model, file)
    
# Load the model from the file
# with open('model.pkl', 'rb') as file:
    # loaded_model = pickle.load(file)
    
# Save the fitted model to a file
dump(best_knn_model, 'model.joblib')

# Load the model from the file
# loaded_model = load('model.joblib')

# Load the trained model
# loaded_model = joblib.load('model.joblib')  # Replace with the path to your saved model

# Load the new data for predictions (e.g., CSV, Excel, etc.)
# new_data = pd.read_csv('new_data.csv')  # Replace with the path to your new data

# Preprocess the new data if needed (same preprocessing as training data)
# ...

# Make predictions on the new data
# predictions = loaded_model.predict(new_data)

# Save the predictions (e.g., as a new column in the DataFrame or in a separate file)
# new_data['predictions'] = predictions
# new_data.to_csv('predictions.csv', index=False)  # Save predictions to CSV

# print("Predictions saved.")


# Function to make predictions
# def make_prediction(new_data):
    # prediction = loaded_model.predict(new_data)
    # return prediction

# Example of using the function for real-time predictions
# new_data_point = ...  # Your new data point
# prediction = make_prediction(new_data_point)

# print("Prediction:", prediction)

# in order to make predictions on newer data, import the predictions into the old code above, run again, save off as a different version
