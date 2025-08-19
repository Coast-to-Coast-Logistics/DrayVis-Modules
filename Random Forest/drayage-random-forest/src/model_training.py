import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor  # or RandomForestClassifier
from sklearn.metrics import mean_squared_error  # or classification metrics

# Step 1: Load the data
data = pd.read_csv('c:\\Users\\gavin\\OneDrive\\Desktop\\DrayVis-Modules\\data\\port_drayage_dummy_data.csv')

# Step 2: Data preparation
# Example: Convert categorical variables to numerical
data = pd.get_dummies(data, columns=['carrier', 'order_type'], drop_first=True)

# Define features and target variable
X = data.drop('rate', axis=1)  # Features
y = data['rate']  # Target variable

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build the Random Forest model
model = RandomForestRegressor(n_estimators=100, random_state=42)  # Use RandomForestClassifier for classification
model.fit(X_train, y_train)

# Step 5: Evaluate the model
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')