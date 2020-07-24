# AGMOModel
This model utilizes probabilistic layers in TensorFlow Probability with Keras, which allows us to quanitfy the uncertantiy 
in the predictions, maximize the probability of our predictions, account for both known unknowns and unknown unknowns, 
and mimimze the negative log-likelyhood loss: 
    -log P(y|x).

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from libdl.dl import dl
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python import tf2
if not tf2.enabled():
  import tensorflow.compat.v2 as tf
  tf.enable_v2_behavior()
  assert tf2.enabled()
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import losses
from tensorflow.python.keras import metrics as metrics_module
from tensorflow.python.keras import optimizers
import tensorflow_probability as tfp
from pprint import pprint
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

%matplotlib inline

tfd = tfp.distributions

print(tf.__version__)

### Training datasets

# Get training data as a dataframe
schema = dl.schemas.get(id="ds_nz_kaggle_files")
train = schema.get(id="training").as_df().set_index(["Organism Code", "purifiedsample_sequoia_identifier"])

# Get the two regression targets
train_y_perf = train['TOM Wash PT Model X'].copy()
train_y_odor = train['TOM Odor PT Model X'].copy()

X = train.drop([train_y_perf.name, train_y_odor.name], axis=1)._get_numeric_data()

# If necessary, replace missing values with column mean values
X = X.fillna(X.mean())

corr_plot = X.corr()
plt.figure(figsize=(10, 6))
sns.set_style('ticks')
sns.heatmap(corr_plot, annot=True)
plt.show()

## Check vif for colinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor
vif = pd.DataFrame()
vif["VIF_Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif

## Drop the suckas that are leaky and/or highly colinear
X.drop(['Calculated_Hydrophobicity','Calculated_Net_charges_PH9_[from_PROpKa]',
        'Calculated_Net_charges_PH7_[from_PROpKa]','Calculated_Net_charges_PH7',
        'Calculated_Net_charges_PH9','OCTET BR4 Rel, Cycle2','OCTET BR4 Rel, Cycle1',
        'OCTET BR4 Raw, Cycle2','OCTET BR4 Raw, Cycle1','OCTET BR2 Rel, Cycle1','OCTET BR2 Rel, Cycle2',
        'OCTET BR3 Raw, Cycle1','OCTET BR3 Raw, Cycle2','OCTET BR3 Rel, Cycle1','OCTET BR3 Rel, Cycle2',
        'OCTET Response_Rel, Cycle1','OCTET Response_Rel, Cycle2','OCTET X=40_Rel, Cycle1','OCTET X=40_Rel, Cycle2',
        'OCTET X=290_Rel, Cycle1','OCTET X=290_Rel, Cycle2','OCTET X=320_Rel, Cycle1','OCTET X=320_Rel, Cycle2',
        'OCTET X=410_Rel, Cycle1','OCTET X=410_Rel, Cycle2','OCTET BR2 Raw, Cycle1','OCTET BR2 Raw, Cycle2',
        'TOM Wash AT Model X','TOM Odor AT Model X','OCTET Response, Cycle1','OCTET Response, Cycle2',
        'OCTET X=290, Cycle1','OCTET X=290, Cycle2','OCTET X=320, Cycle1','OCTET X=320, Cycle2','OCTET X=410, Cycle1',
        'OCTET X=410, Cycle2','TOM Wash PT Persil Universal Powder','TOM Odor PT Persil Universal Powder',
        'TOM Wash AT Persil Universal Powder','TOM Odor AT Persil Universal Powder',
        'A280/A260','PUNE FI-R-A-01-00','PUNE FI-R-A-01-05','PUNE FI-R-A-01-10','PUNE FI-R-A-01-15',
        'PUNE FI-R-A-01-20','PUNE FI-R-A-01-25','PUNE FI-R-A-01-35','PUNE FI-R-A-01-45','PUNE FI-R-A-01-55',
        'PUNE FI-R-D-01-20','PUNE FI-R-D-01-25','PUNE FI-R-D-01-35','PUNE FI-R-D-01-45','PUNE FI-R-D-01-55',
        'OCTET X=40, Cycle1','OCTET X=40, Cycle2','OCTET BR1 Raw, Cycle1','OCTET BR1 Raw, Cycle2',
        'GRAST Humidity Stress','AMSA_across_trial_rp_wash_Persil_Universal_15','A280 Concentration mg/ml',
        'A280','AST Concentration mg/ml','GRAST Hum+Ox Stress','AMSA IWS Degradation Factor detergent[Persil Universal]_dH[15]'], axis=1, inplace=True)

## And check again
vif = pd.DataFrame()
vif["VIF_Factor"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif["features"] = X.columns
vif

corr_plot = X.corr()
plt.figure(figsize=(10, 6))
sns.set_style('ticks')
sns.heatmap(corr_plot, annot=True)
plt.show()


### Performance Dataset

## Get the test data
test_X = schema.get(id="validation_x").as_df().set_index(["Organism Code", "purifiedsample_sequoia_identifier"])

actual_values_wash = test_X.pop('TOM Wash AT Model X')
actual_values_odor = test_X.pop('TOM Odor AT Model X')

## Get test data
#test_X = pd.read_csv("test_X.csv")

## For this simple model get only the numeric features
test_X = test_X._get_numeric_data() 

## If necessary, impute the missing values using column mean values computed from the training set
#test_X = test_X.fillna(X.mean())

test_X.drop(['Calculated_Hydrophobicity','Calculated_Net_charges_PH9_[from_PROpKa]',
        'Calculated_Net_charges_PH7_[from_PROpKa]','Calculated_Net_charges_PH7',
        'Calculated_Net_charges_PH9','OCTET BR4 Rel, Cycle2','OCTET BR4 Rel, Cycle1',
        'OCTET BR4 Raw, Cycle2','OCTET BR4 Raw, Cycle1','OCTET BR2 Rel, Cycle1','OCTET BR2 Rel, Cycle2',
        'OCTET BR3 Raw, Cycle1','OCTET BR3 Raw, Cycle2','OCTET BR3 Rel, Cycle1','OCTET BR3 Rel, Cycle2',
        'OCTET Response_Rel, Cycle1','OCTET Response_Rel, Cycle2','OCTET X=40_Rel, Cycle1','OCTET X=40_Rel, Cycle2',
        'OCTET X=290_Rel, Cycle1','OCTET X=290_Rel, Cycle2','OCTET X=320_Rel, Cycle1','OCTET X=320_Rel, Cycle2',
        'OCTET X=410_Rel, Cycle1','OCTET X=410_Rel, Cycle2','OCTET BR2 Raw, Cycle1','OCTET BR2 Raw, Cycle2',
        'OCTET Response, Cycle1','OCTET Response, Cycle2','TOM Wash PT Persil Universal Powder','TOM Odor PT Persil Universal Powder',
        'TOM Wash AT Persil Universal Powder','TOM Odor AT Persil Universal Powder',
        'OCTET X=290, Cycle1','OCTET X=290, Cycle2','OCTET X=320, Cycle1','OCTET X=320, Cycle2','OCTET X=410, Cycle1',
        'OCTET X=410, Cycle2','A280/A260','PUNE FI-R-A-01-00','PUNE FI-R-A-01-05','PUNE FI-R-A-01-10','PUNE FI-R-A-01-15',
        'PUNE FI-R-A-01-20','PUNE FI-R-A-01-25','PUNE FI-R-A-01-35','PUNE FI-R-A-01-45','PUNE FI-R-A-01-55',
        'PUNE FI-R-D-01-20','PUNE FI-R-D-01-25','PUNE FI-R-D-01-35','PUNE FI-R-D-01-45','PUNE FI-R-D-01-55',
        'OCTET X=40, Cycle1','OCTET X=40, Cycle2','OCTET BR1 Raw, Cycle1','OCTET BR1 Raw, Cycle2',
        'GRAST Humidity Stress','AMSA_across_trial_rp_wash_Persil_Universal_15','A280 Concentration mg/ml',
        'A280','AST Concentration mg/ml','GRAST Hum+Ox Stress','AMSA IWS Degradation Factor detergent[Persil Universal]_dH[15]'], axis=1, inplace=True)

test_X.shape

### The Model

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(X.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

perfmodel = build_model()
perfmodel.summary()

odormodel = build_model()
odormodel.summary()

# Train the Performance Wash Model

# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 2500

perf = perfmodel.fit(
  X, train_y_perf,
  epochs=EPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

hist = pd.DataFrame(perf.history)
hist['epoch'] = perf.epoch
hist.tail()

def plot_history(perf):
  hist = pd.DataFrame(perf.history)
  hist['epoch'] = perf.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,3])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()
  plt.show()


plot_history(perf)

#model = build_model()

# Check for improvement
#early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

#history = model.fit(X, train_y_perf, epochs=EPOCHS,
#                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

#plot_history(history)

preds_perf_test = perfmodel.predict(test_X).flatten()
perf_test_values = pd.DataFrame(preds_perf_test, index=test_X.index) 

# Train the Odor Model

ODOREPOCHS = 2500

odor = odormodel.fit(
  X, train_y_odor,
  epochs=ODOREPOCHS, validation_split = 0.2, verbose=0,
  callbacks=[PrintDot()])

hist = pd.DataFrame(odor.history)
hist['epoch'] = odor.epoch
hist.tail()

def plot_history(odor):
  hist = pd.DataFrame(odor.history)
  hist['epoch'] = odor.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


plot_history(odor)

#odor_model = build_model()

# Check for improvement
#early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

#odor_history = model.fit(X, train_y_odor, epochs=EPOCHS,
#                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

#plot_history(odor_history)

preds_odor_test = odormodel.predict(test_X).flatten()
odor_test_values = pd.DataFrame(preds_odor_test, index=test_X.index) 

# Visualize Model Performance

plt.scatter(actual_values_wash, preds_perf_test)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')

error = preds_perf_test - actual_values_wash
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")

plt.scatter(actual_values_odor, preds_odor_test)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')

error = preds_odor_test - actual_values_odor
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error")
_ = plt.ylabel("Count")

### Generating the submission files

# Create the two submission files
pd.DataFrame({'pred': preds_perf_test}, index=test_X.index).to_csv('perf_test_submission.csv')
pd.DataFrame({'pred': preds_odor_test}, index=test_X.index).to_csv('odor_test_submission.csv')

