import pandas as pd
import openpyxl as oxl
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression


#url = 'https://raw.githubusercontent.com/guipsamora/pandas_exercises/master/Visualization/Titanic_Desaster/train.csv'
url = 'Titanic.xlsx'
titanic = pd.read_excel(url)
titanic['PassengerId'] = pd.to_numeric(titanic['PassengerId'], errors='coerce')
titanic.dropna(subset=['PassengerId'], inplace=True)

# 1. Set PassengerId as the index.
titanic = titanic.set_index('PassengerId')

# 2. Create a pie chart presenting the male/female as proportions

# Count the occurrences of each gender
gender_counts = titanic['Sex'].value_counts()

# Plotting a pie chart
plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Gender Distribution')
# Equal aspect ratio ensures that pie is drawn as a circle
plt.axis('equal')
plt.show()

# 3. Create a scatterplot with the Fare payed and the Age, differ the plot color by gender

# Define colors for genders
colors = {'male': 'blue', 'female': 'red'}

# Create scatter plot
plt.figure(figsize=(8, 6))
for gender in titanic['Sex'].unique():
    plt.scatter(
        titanic[titanic['Sex'] == gender]['Age'],
        titanic[titanic['Sex'] == gender]['Fare'],
        c=colors[gender],
        label=gender,
        alpha=0.7
    )

# Set labels and title
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Fare Paid vs Age (Colored by Gender)')
plt.legend()
plt.grid(True)
plt.show()

# 4. How many people survived?
sum(titanic['Survived'])
titanic['Survived'].sum() / titanic['Survived'].count()

# 5. Create a histogram with the Fare paid
plt.hist(titanic['Fare'], bins=20, color='skyblue', edgecolor='black')  # Adjust the number of bins as needed
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.title('Histogram of Fare')
plt.show()

# 6. BONUS: Create your own question and answer it.
# H_0 = higher paying male and females are more likely to survive as compare to their lower paying counterparts

# Subsetting data for Male and Female separately
male_survived = titanic[(titanic['Sex'] == 'male') & (titanic['Survived'] == 1)]['Fare']
male_not_survived = titanic[(titanic['Sex'] == 'male') & (titanic['Survived'] == 0)]['Fare']
female_survived = titanic[(titanic['Sex'] == 'female') & (titanic['Survived'] == 1)]['Fare']
female_not_survived = titanic[(titanic['Sex'] == 'female') & (titanic['Survived'] == 0)]['Fare']
survived = titanic[titanic['Survived'] == 1]['Fare']
not_survived = titanic[titanic['Survived'] == 0]['Fare']

# Perform t-test for Male survivors vs non-survivors
tstat_male, pval_male = ttest_ind(male_survived, male_not_survived, equal_var=False)

# Perform t-test for Female survivors vs non-survivors
tstat_female, pval_female = ttest_ind(female_survived, female_not_survived, equal_var=False)

# Perform t-test across all gender survivors vs non-survivors
tstat, pval = ttest_ind(survived, not_survived, equal_var=False)

# Print the p-values
print(f"Male p-value: {pval_male}") #fail to reject H_0
print(f"Female p-value: {pval_female}") #fail to reject H_0
print(f"All p-value: {pval}") #fail to reject H_0

# Survived = Pclass + Sex + Age +SibSp + Parch + Fare + Cabin + Embarked
# Assuming 'X' contains your features and 'y' contains the target variable ('Survived')
# Perform necessary data preprocessing steps before this

# Initialize a RandomForestClassifier
# clf = RandomForestClassifier() - doesn't handle NaN values at all
# Decision Tree doesn't handle NaN values very well, so let's impute the data
clf = DecisionTreeClassifier()

# Make sure the data is acurate for the classifier needs
le = LabelEncoder()
titanic['Sex'] = le.fit_transform(titanic['Sex'])
titanic['Cabin'] = le.fit_transform(titanic['Cabin'])
titanic['Embarked'] = le.fit_transform(titanic['Embarked'])

#how to handle NaN values...
new_titanic = titanic[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked', 'Survived']]

from sklearn.impute import KNNImputer
imputer = KNNImputer()
new_titanic = pd.DataFrame(imputer.fit_transform(new_titanic), columns=new_titanic.columns)

"""
# Example with RandomForestRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='mean')
df[['column_with_missing_values']] = imputer.fit_transform(df[['column_with_missing_values']])


for col in new_titanic.columns:
    regressor = LinearRegression() 
    regressor.fit(new_titanic.loc[:, new_titanic.columns != col], new_titanic[col])  
    new_titanic.loc[new_titanic[col].isnull(), col] = regressor.predict(new_titanic.loc[new_titanic[col].isnull(), titanic.columns != col])

"""

# Fit the model
clf.fit(new_titanic.loc[:, new_titanic.columns != 'Survived'], new_titanic['Survived'])

# Get feature importances
feature_importances = pd.Series(clf.feature_importances_, index=new_titanic.loc[:, new_titanic.columns != 'Survived'].columns)
sorted_importances = feature_importances.sort_values(ascending=False)

# Plot feature importances
sorted_importances.plot(kind='barh')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importances')
plt.show()  # Display the plot

"""
# Select features based on importance
selected_features = feature_importances[feature_importances > .07].index.tolist()

# Prepare X (features) and y (target variable)
X = your_data[selected_features]
y = your_data['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choose a model and train it
model = RandomForestClassifier()  # or any other classifier
model.fit(X_train, y_train)

# Make predictions on test data
predictions = model.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")
"""