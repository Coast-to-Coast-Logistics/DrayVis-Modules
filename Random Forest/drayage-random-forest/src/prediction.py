import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # or RandomForestClassifier
from sklearn.metrics import mean_squared_error  # or accuracy_score for classification

# Step 1: Load the data
data = pd.read_csv('c:\\Users\\gavin\\OneDrive\\Desktop\\DrayVis-Modules\\data\\port_drayage_dummy_data.csv')

# Step 2: Data Cleaning
# Check for missing values
print(data.isnull().sum())

# Step 3: Feature Selection
# Assuming 'rate' is the target variable and the rest are features
X = data.drop('rate', axis=1)  # Features
y = data['rate']  # Target variable

# Step 4: Encoding categorical variables
X = pd.get_dummies(X, drop_first=True)

# Step 5: Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Create and fit the model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # Change to RandomForestClassifier if needed
model.fit(X_train, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Step 9: Feature Importance
importances = model.feature_importances_
feature_importance = pd.DataFrame(importances, index=X.columns, columns=['Importance']).sort_values('Importance', ascending=False)
print(feature_importance)