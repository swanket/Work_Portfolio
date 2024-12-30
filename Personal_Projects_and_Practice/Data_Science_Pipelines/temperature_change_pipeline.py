import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

data = pd.read_excel('computerlab_data.xlsx')
print(data.head())

time = data['date (year)']

X = data[['Solar constant (unit= W/m2, SATIRE project and Lean 2000)', 'Global mean stratospheric aerosol optical depth (unit=dimensionless, GISS)', 'Atmospheric CO2 concentration (unit=ppm, Earth Policy Institute/NOAA)',
          'Anthropogenic SO2 emissions (unit=Tg/yr, from Pacific Northwest National Laboratory)', 'El Nino index (unit=dimensionless, NOAA)']]

Y = data['Global mean temperature anomaly (unit=degree celsius, HadCRUT, 1961-1990 seasonal cycle used to calculate anomaly)']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Standardize features
    ('model', RandomForestRegressor(n_estimators=100, random_state=42))  # Predictive model
])

# Step 6: Train the Pipeline
pipeline.fit(X_train, y_train)

# Step 7: Evaluate the Model
# Make predictions
y_pred = pipeline.predict(X_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse:.2f}")

print(f'Length of y_pred = {len(y_pred)}')
print(f'Length of y_test = {len(y_test)}')

# Extract the Random Forest model from the pipeline
rf_model = pipeline.named_steps['model']

# Get feature importance
feature_importance = rf_model.feature_importances_

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

# Display the feature importance
print("Feature Importances:")
print(importance_df)


plt.figure(figsize=(8, 6))
plt.bar(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.title('Feature Importance')
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.show()


# plt.figure(figsize = (10,7))
# plt.plot(y_pred)
# plt.scatter(X_test,y_test)
# plt.show()
