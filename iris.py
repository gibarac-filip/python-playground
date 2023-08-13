from sklearn.datasets import load_iris
import pandas as pd
import matplotlib as matplot
import seaborn

# Load the Iris dataset
iris = load_iris()

# Access the data, target, feature names, etc.
data = iris.data
target = iris.target
feature_names = iris.feature_names
target_names = iris.target_names
iris_data = pd.DataFrame(data, columns=feature_names)
target_data = pd.DataFrame(target)
iris_data.shape
iris_data.size
iris_data.info()
iris_data.describe()
# Check for missing values
missing_iris = iris_data.isna().any(axis=1)
missing_target = target_data.isna().any(axis=1)

# Drop rows with missing values
iris_no_missing = iris_data.dropna()
target_no_missing = target_data.dropna()

# Drop duplicate rows
iris_no_duplicates = iris_no_missing.drop_duplicates()
target_no_duplicates = target_no_missing.drop_duplicates()