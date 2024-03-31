from keras.datasets import boston_housing


(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

#Train data
train_data_s = train_data
train_data_s -= mean
train_data_s /= std

#Test Data
test_data_s = test_data
test_data_s -= mean
test_data_s /= std


from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu', input_shape=(train_data_s.shape[1],)))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mean_absolute_error'])
    return model

model = build_model()

model.load_weights('model.weights.best.hdf5')

test_mse_score, test_mae_score = model.evaluate(test_data_s, test_targets)
print(f" Neural Network: mse : {test_mse_score} mae: {test_mae_score}")


predictions = model.predict(test_data_s)

for i in range(10):
    print(f"Prediction: {predictions[i][0]:.2f}, Actual: {test_targets[i]}")

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(train_data_s, train_targets)

results = forest.score(test_data_s, test_targets)

print(" Random Forest Regressor -----> ")
print(results)

predictions_forest = forest.predict(test_data_s)


for i in range(10):
    print(f"Prediction: {predictions_forest[i]:.2f}, Actual: {test_targets[i]}")

from sklearn.metrics import mean_absolute_error


mae = mean_absolute_error(test_targets, predictions_forest)
print(f"Random Forest : Mean Absolute Error: {mae}")


stacking = []

for i in range(len(test_targets)):
    stacking.append((predictions[i][0] + predictions_forest[i]) / 2)

stacking_mae = mean_absolute_error(test_targets, stacking)
print(f"Embeded Average Model : Mean Absolute Error: {stacking_mae}")

import numpy as np

def calculate_mape(y_true, y_pred):
    """
    Calculate the Mean Absolute Percentage Error (MAPE)

    Parameters:
    y_true (array-like): True values
    y_pred (array-like): Predicted values

    Returns:
    mape (float): The MAPE between y_true and y_pred
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero_mask = y_true != 0  # Avoid division by zero
    return np.mean(np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])) * 100


mape = calculate_mape(test_targets, stacking)
print(f"MAPE: {mape}%")

from sklearn.metrics import mean_absolute_percentage_error

mape_2 = mean_absolute_percentage_error(test_targets, stacking)
print(f"MAPE: {mape_2}%")



print(" --- Linear Regression Stacking ")

from sklearn.linear_model import LinearRegression

stacking_array = np.array(stacking)

print(stacking_array.shape)
print(train_targets.shape)
print(train_data_s.shape)

combined_test_predictions = np.hstack((predictions.reshape(-1, 1), predictions_forest.reshape(-1, 1)))


linear = LinearRegression()

linear.fit(combined_test_predictions, test_targets)

linear_predictions = linear.predict(combined_test_predictions)
print(linear_predictions)

mape_linear = mean_absolute_percentage_error(test_targets, linear_predictions)
print(f"Mape Linear ? -> {mape_linear}")
