#!/usr/bin/env python
# coding: utf-8

# In[4]:


# Import necessary libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic data for demonstration purposes with two different heart rate ranges
np.random.seed(42)
heart_rate_low = np.random.randint(60, 80, 50)
heart_rate_high = np.random.randint(80, 100, 50)
heart_rate = np.concatenate((heart_rate_low, heart_rate_high))

cpr_amount = 0.5 * heart_rate + np.random.normal(scale=5, size=100)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(heart_rate.reshape(-1, 1), cpr_amount, test_size=0.2, random_state=42)

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
print(f'Mean Squared Error: {mse}')

# Plot the results
plt.scatter(X_test, y_test, color='black')
plt.plot(X_test, predictions, color='blue', linewidth=3)
plt.xlabel('Heart Rate')
plt.ylabel('CPR Amount')
plt.title('Heart Rate vs. CPR Amount Prediction')
plt.show()

# Use the model to predict CPR amount for new heart rates
new_heart_rates = np.array([[180]])  # Replace with your desired heart rates
predicted_cpr = model.predict(new_heart_rates)



# Display predicted CPR amounts for new heart rates
for i in range(len(new_heart_rates)):
    print(f'Predicted CPR Amount for {new_heart_rates[i][0]} heart rate: {predicted_cpr[i]}')


# In[ ]:




