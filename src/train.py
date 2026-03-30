import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
# Load dataset
df = pd.read_csv("data/housing.csv")

# Handle missing values (IMPORTANT FIX)
df = df.fillna(df.mean(numeric_only=True))

# Separate features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Convert categorical column
X = pd.get_dummies(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("===== MODEL RESULTS =====")
print(f"Dataset size: {len(df)}")
print(f"RMSE: {rmse}")
print(f"R2 Score: {r2}")