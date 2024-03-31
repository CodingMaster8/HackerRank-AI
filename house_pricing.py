#The purpose of this model will be to get the best MAE for house price prediction using the Boston Dataset.
#State of the Art model will be made using ensemble methods, hyper-optimization, and batch normalization.

from keras.datasets import boston_housing


#Get the data from the dataset
(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()


"""Normalize Data with z-score normalization (Standarization) to prevent outliers to affect results"""
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

""" Build Network """

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


""" K-Cross Validation """

from keras.callbacks import ModelCheckpoint
checkpointer = ModelCheckpoint(filepath='model6.weights.best.hdf5', verbose=1, save_best_only=True)

"""
import numpy as np
k = 4
num_val_samples = len(train_data_s) // k
num_epochs = 100
all_mae_histories = []

for i in range(k):
    print('processing fold #', i)
    val_data = train_data_s[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
    partial_train_data = np.concatenate([train_data_s[:i * num_val_samples], train_data_s[(i + 1) * num_val_samples:]], axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples], train_targets[(i + 1) * num_val_samples:]], axis=0)

    model = build_model()
    model.fit(partial_train_data, partial_train_targets,
                        validation_data=(val_data, val_targets),
                        epochs=num_epochs, batch_size=1, verbose=2, shuffle=True, callbacks=[checkpointer])

test_mse_score, test_mae_score = model.evaluate(test_data_s, test_targets)

print(f"mse : {test_mse_score} mae: {test_mae_score}")

predictions = model.predict(test_data_s)

for i in range(10):
    print(f"Prediction: {predictions[i][0]:.2f}, Actual: {test_targets[i]}")
"""


"""
import matplotlib.pyplot as plt


def smooth_curve(points, factor=0.9):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points


smooth_mae_history = smooth_curve(average_mae_history[10:])
plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()
"""


num_val_samples = len(train_data_s) // 4

val_data = train_data_s[3 * num_val_samples: 4 * num_val_samples]
val_targets = train_targets[3 * num_val_samples: 4 * num_val_samples]

train_data_new = train_data_s[:3 * num_val_samples]
train_targets_new = train_targets[:3 * num_val_samples]

model = build_model()
model.fit(train_data_s, train_targets, epochs=200, batch_size=16, validation_data=(val_data, val_targets), verbose=2, shuffle=True, callbacks=[checkpointer])

test_mse_score, test_mae_score = model.evaluate(test_data_s, test_targets)

print(f"mse : {test_mse_score} mae: {test_mae_score}")

predictions = model.predict(test_data_s)

for i in range(10):
    print(f"Prediction: {predictions[i][0]:.2f}, Actual: {test_targets[i]}")
