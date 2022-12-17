import pandas as pd
import numpy as np
from numpy.random import seed
import datetime as dt
import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt
from collections.abc import Iterable
import json

from pathlib import Path
import shutil
from django.conf import settings
from ..models import *

import mlmodel.helper_functions.features_data as features_data
import mlmodel.helper_functions.transform as transform
import mlmodel.helper_functions.predict as predict

offset_days = [5,15,30]
reportTrainingProgress = False
plotInitialTrainingGraph = False
plotDeltaTrainingGraph = False
daysForUsingAModel = 5
numberOfGoodModelsRequired = 3
requiredAccuracy = 0.875
sparseAccuracy = 0.825
backupAccuracyIfNoModelsShortlisted = 0.8
useActivation = 'sigmoid'
mapper = {
  'invested_amount_new_30' : "invested_amount",
  'loantoinvest_new_30' : "loan_to_invest_ratio", 
  'bucketRatio1_new_30' : "lowriskbucket_ratio", 
  'bucketRatio2_new_30' : "mediumriskbucket_ratio", 
  'bucketRatio3_new_30' : "highriskbucket_ratio"
}

def flatten(x):
    if isinstance(x, Iterable):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]

def doInitialTraining(client_id, FEATURE, x_train_data, y_train_data, minAccuracyNeeded = 0.875, sparseDataAccuracy = 0.825, 
                      backupAccuracy = 0.8, activationType = 'softmax', nEpochs = 300, goodModelsRequired = 3, 
                      sufficientDataLimit = 400, sparseDataLimit = 200, reportProgress = False, plotGraphs = False,
                      optimizerUsed = tensorflow.keras.optimizers.Nadam(learning_rate=0.0085, beta_1=0.9, beta_2=0.98, epsilon=1e-05, name="Nadam")):

  if reportProgress:
    print(f"Starting training for client {client_id} for feature {FEATURE}")

  primeList=[7,13,23,3,5,11,17,19,2,29,31,37,41,43,47,53,59,61,67,71,79,83,89,97,101,103,107]
  trainingModels = []
  backupModels = []
  goodModelsCount = 0
  currentRequiredAccuracy = minAccuracyNeeded
  if (x_train_data.shape[0] < sparseDataLimit):
    currentRequiredAccuracy = sparseDataAccuracy
  elif (x_train_data.shape[0] < sufficientDataLimit):
    currentRequiredAccuracy = (sparseDataAccuracy + minAccuracyNeeded) / 2.

  for i in primeList:
    if reportProgress:
      print(f"  Initialising with seed {i}")
    
    # set seeds
    seed(3*i)
    tensorflow.random.set_seed(9*i)
    
    # Neural Network###########################################
    ann = Sequential()
    ann.add(Input(shape=(10*len(offset_days),), name='Input-Layer')) 
    ann.add(Dense(2*10*len(offset_days), activation=activationType, name='Hidden-Layer1'))
    ann.add(Dense(3*10*len(offset_days), activation=activationType, name='Hidden-Layer2'))
    ann.add(Dense(int(1.8*10*len(offset_days)), activation=activationType, name='Hidden-Layer3'))
    ann.add(Dense(int(1.2*10*len(offset_days)), activation=activationType, name='Hidden-Layer4'))
    ann.add(Dense(int(0.6*10*len(offset_days)), activation=activationType, name='Hidden-Layer5'))
    ann.add(Dense(5, activation=activationType, name='Hidden-Layer6'))
    ann.add(Dense(1, activation='sigmoid', name='Output-Layer'))
    ###############################################################

    ann.compile(optimizer=optimizerUsed, loss='binary_crossentropy', metrics=['Accuracy', 'Precision', 'Recall'])

    # ann.fit doesn't work with either of validation_data input and with shuffle='True'
    seqModel = ann.fit(x_train_data, y_train_data, epochs=nEpochs, verbose = 2 if reportProgress else 0)
    
    # visualizing losses and accuracy
    # print(seqModel.history.keys())
    # print(seqModel.history.values())
    if plotGraphs:
      train_loss = seqModel.history['loss']
      accuracy   = seqModel.history['Accuracy']
      precision  = seqModel.history['precision']
      recall    = seqModel.history['recall']
      xc         = range(nEpochs)

      plt.figure()
      plt.plot(xc, train_loss)
      plt.plot(xc, accuracy)
      plt.plot(xc, precision)
      plt.plot(xc, recall)
      plt.legend(['Loss', 'Accuracy', 'Precision', 'Recall'])
      plt.show()

    endingAccuracy = seqModel.history['Accuracy'][nEpochs - 2]
    if (endingAccuracy >= currentRequiredAccuracy):
      trainingModels.append(ann)
      if reportProgress:
        print("  Adding this model to the shortlisted training models")
      goodModelsCount += 1
    else:
      if (endingAccuracy >= backupAccuracy):
        backupModels.append(ann)
    
    if (goodModelsCount > goodModelsRequired - 1):
      break
  
  if len(trainingModels) == 0:
    trainingModels = backupModels
    goodModelsCount = len(trainingModels)
    
  return trainingModels, goodModelsCount

def doRetraining(trainingModels, x_delta_data, y_delta_data, optimizerUsed = tensorflow.keras.optimizers.Nadam(
  learning_rate=0.0045, beta_1=0.9, beta_2=0.99, epsilon=1e-05, name="Nadam"), nEpochs = 100, reportProgress = False, plotGraphs = False):

  if reportProgress:
    print(f"Starting retraining")

  for trainingModel in trainingModels:
    trainingModel.compile(optimizer=optimizerUsed, loss='binary_crossentropy', metrics=['Accuracy', 'Precision', 'Recall'])
    trainingModel.fit(x_delta_data, y_delta_data, epochs=nEpochs, verbose = 2 if reportProgress else 0)

  return trainingModels

def doPredictionsAndRetraining(client_id, FEATURE, trainingModels, x_test_data, y_test_data, cutoffForPlus1Score = None, 
                               optimizerUsed = tensorflow.keras.optimizers.Nadam(learning_rate=0.0045, beta_1=0.9, beta_2=0.99, epsilon=1e-05, name="Nadam"), 
                               nEpochs = 100, daysForUsingAModel = 5, reportProgress = False, plotGraphs = False):

  daysPredictionNeeded = len(x_test_data)

  print(f"Starting prediction accuracy calculation and retraining for client {client_id} for feature {FEATURE}")
  if reportProgress:
    print(f"Doing {daysPredictionNeeded} predictions and {(daysPredictionNeeded // daysForUsingAModel)} retrainings")
 
  countOfModels = len(trainingModels)
  
  if cutoffForPlus1Score == None:
    cutoffForPlus1Score = 1./countOfModels
  cutoffForPlus1Score = min(1., cutoffForPlus1Score)

  pred_labels_temp = []

  for i in range(0, (daysPredictionNeeded//daysForUsingAModel)):
    sumOfPredictions = None
    for trainingModel in trainingModels:
      currentPrediction = trainingModel.predict(x_test_data.iloc[i*daysForUsingAModel:(i+1)*daysForUsingAModel])
      if sumOfPredictions is None:
        sumOfPredictions = currentPrediction
      else:
        sumOfPredictions = sumOfPredictions + currentPrediction
    sumOfPredictions = sumOfPredictions / (countOfModels * 1.)
    if reportProgress:
      SOPText = list(flatten(sumOfPredictions))
      print(f"Initial calculated predictions for days {i*daysForUsingAModel} to {((i+1)*daysForUsingAModel - 1)} are {SOPText}")
    sumOfPredictions = np.where(sumOfPredictions > cutoffForPlus1Score, 1, 0)
    sumOfPredictions = sumOfPredictions.astype(int)
    
    if reportProgress:
      SOPText = list(flatten(sumOfPredictions))
      print(f"Final calculated predictions for days {i*daysForUsingAModel} to {((i+1)*daysForUsingAModel - 1)} are {SOPText}")

    pred_labels_temp.append(sumOfPredictions)

    trainingModels = doRetraining(trainingModels, x_test_data.iloc[i*daysForUsingAModel:(i+1)*daysForUsingAModel], 
                                  y_test_data.iloc[i*daysForUsingAModel:(i+1)*daysForUsingAModel], 
                                  reportProgress = reportProgress, plotGraphs = plotGraphs)
    

  daysRemaining = daysPredictionNeeded % daysForUsingAModel
  if daysRemaining != 0:
    sumOfPredictions = None
    for trainingModel in trainingModels:
      currentPrediction = trainingModel.predict(x_test_data.iloc[(daysPredictionNeeded - daysRemaining):daysPredictionNeeded])
      if sumOfPredictions is None:
        sumOfPredictions = currentPrediction
      else:
        sumOfPredictions = sumOfPredictions + currentPrediction
    sumOfPredictions = sumOfPredictions / (countOfModels * 1.)

    if reportProgress:
      SOPText = list(flatten(sumOfPredictions))
      print(f"Initial calculated predictions for days {daysPredictionNeeded - daysRemaining} to {daysPredictionNeeded - 1} are {SOPText}")
    sumOfPredictions = np.where(sumOfPredictions >= cutoffForPlus1Score, 1, 0)
    sumOfPredictions = sumOfPredictions.astype(int)

    if reportProgress:
      SOPText = list(flatten(sumOfPredictions))
      print(f"Final predictions for days {daysPredictionNeeded - daysRemaining} to {daysPredictionNeeded - 1} are {SOPText}")

    pred_labels_temp.append(sumOfPredictions)

  flattened_predictions = list(flatten(pred_labels_temp))
  daysPredicted = len(flattened_predictions)
  if reportProgress:
    print(f"Flattened predictions are : {flattened_predictions}")
    print(f"Days of prediction needed = {daysPredictionNeeded}; Days of prediction done = {daysPredicted}")

  return trainingModels, flattened_predictions

"""
Training and saving a model for a particular feature
"""
def train(rm, client_id, FEATURE, input_data, output_data):
  output_data_cp = output_data.copy()
  output_data_cp['label'] = output_data_cp[FEATURE]

  for row in range(len(output_data_cp)):
    if output_data_cp.iloc[row, 0] == -1:
      output_data_cp.iloc[row, 1]= 0
    else:
      output_data_cp.iloc[row, 1]= 1

  del output_data_cp[FEATURE]
  
  size = int(len(input_data)*0.8)
  X_train, X_test, y_train, y_test = input_data.iloc[:size], input_data.iloc[size:], output_data_cp.iloc[:size], output_data_cp.iloc[size:]

  trainingModels, goodModelsCount = doInitialTraining(client_id, FEATURE, X_train, y_train, activationType = useActivation, goodModelsRequired = numberOfGoodModelsRequired, 
                                                      minAccuracyNeeded = requiredAccuracy, sparseDataAccuracy = sparseAccuracy, backupAccuracy = backupAccuracyIfNoModelsShortlisted,
                                                      reportProgress = reportTrainingProgress, plotGraphs = plotInitialTrainingGraph)

  trainingModels, flatten_list = doPredictionsAndRetraining(client_id, FEATURE, trainingModels, X_test, y_test, reportProgress = reportTrainingProgress, plotGraphs = plotDeltaTrainingGraph)


  # for i, model in enumerate(trainingModels):
  #   print(i)
  #   model.save(Path(settings.CLIENT_MODELS_DIRECTORY, rm ,f"client_{client_id}", mapper[FEATURE] ,f"model_{i}.h5"))

  prediction=[]
  prediction.append(flatten_list)

  df= pd.DataFrame(prediction)
  test_prediction= df.mode().loc[0].to_list() 
  
  ans = pd.DataFrame(test_prediction, columns = ["Predicted"])
  ans['Truth'] = y_test.values
  ans['accuracy'] = 0
  ans.loc[ans['Predicted'] == ans['Truth'], ['accuracy']] = 1
  acc = (ans[ans.accuracy == 1].shape[0] / ans.shape[0])*100
  return acc, flatten_list


"""
This function is used to train the model for a particular client.
"""
def perclient(rm, client_id, client_data):
  client_data_cp = client_data.copy()

  # features df
  input_data = features_data.generate(client_data_cp)
  # save input df for 6th graph
  input_data.to_csv(Path(settings.CLIENT_MODELS_DIRECTORY, rm, f"client_{client_id}", "input_data.csv"), index = False)

  # transformed df for input
  transformed_input_data = transform.dataCreation_NN_input(input_data)
  transformed_input_data.report_date = pd.to_datetime(transformed_input_data.report_date).dt.date
  # save transformed input df
  transformed_input_data.to_csv(Path(settings.CLIENT_MODELS_DIRECTORY, rm, f"client_{client_id}", "transformed_input_data.csv"), index = False)
  
  # obtain output df from input
  transformed_output_data = transform.dataCreation_NN_output(transformed_input_data)
  transformed_output_data.report_date = pd.to_datetime(transformed_output_data.report_date).dt.date

  features_acc = {
    'invested_amount_new_30' : 0, 
    'loantoinvest_new_30' : 0, 
    'bucketRatio1_new_30' : 0, 
    'bucketRatio2_new_30' : 0, 
    'bucketRatio3_new_30' : 0
  }

  features_pred = {
    'invested_amount_new_30' : [], 
    'loantoinvest_new_30' : [], 
    'bucketRatio1_new_30' : [], 
    'bucketRatio2_new_30' : [], 
    'bucketRatio3_new_30' : []
  }
  last_days, start_date, last_date = predict.adjust_input_params(transformed_input_data, last_days=int((len(input_data)-30)*0.2), start_date=None, last_date=None)

  transformed_input_data.set_index("report_date", inplace = True)
  transformed_output_data.set_index("report_date", inplace = True)
  for cols in transformed_output_data.columns:
    transformed_output_data[cols] = transformed_output_data[cols].astype('int')
  transformed_input_data = transformed_input_data.iloc[:-30]

 # training and saving model for each target
  for feature in features_acc.keys():
    features_acc[feature], features_pred[feature] = train(rm, client_id, feature, transformed_input_data, transformed_output_data[[feature]])

  pred = {
    "client_id": int(client_id),
    "start_date": {
      "day": int(start_date.strftime("%d")),
      "month": int(start_date.strftime("%m")),
      "year" : int(start_date.strftime("%Y"))
    },
    "end_date": {
      "day": int(last_date.strftime("%d")),
      "month": int(last_date.strftime("%m")),
      "year" : int(last_date.strftime("%Y"))
    },
    "working_days" : int(last_days),
    "t_prediction": {
      "invested_amount": features_pred['invested_amount_new_30'],
      "loan_to_invest_ratio": features_pred['loantoinvest_new_30'],
      "lowriskbucket_ratio": features_pred['bucketRatio1_new_30'],
      "mediumriskbucket_ratio": features_pred['bucketRatio2_new_30'],
      "highriskbucket_ratio": features_pred['bucketRatio3_new_30']
    }
  }
  dic = pred['t_prediction']
  for key in dic.keys():
    for idx, i in enumerate(dic[key]):
      dic[key][idx] = int(i)

  json.dump(pred, open(Path(settings.CLIENT_MODELS_DIRECTORY, rm, f"client_{client_id}", "predictions.json"), 'w'))

  Accuracy = {
    "id": client_id,
    "invested_amount": features_acc['invested_amount_new_30'],
    "loan_to_invest_ratio": features_acc['loantoinvest_new_30'],
    "lowriskbucket_ratio": features_acc['bucketRatio1_new_30'],
    "mediumriskbucket_ratio": features_acc['bucketRatio2_new_30'],
    "highriskbucket_ratio": features_acc['bucketRatio3_new_30']
  }
  return Accuracy


"""
This function returns the data of the client having client id as "client_id" from the whole input data, "data"
"""
def get_client_data(data, client_id):
  client_data = data[data.client_id == client_id]
  client_data = client_data.sort_values(['report_date'], ascending=[True])
  del client_data['client_id']
  return client_data


"""
This function creates the required directory structure for the client having client id as "client_id"
"""
def setup_directory(rm, client_id):
  dirpath = Path(settings.CLIENT_MODELS_DIRECTORY) / rm / f"client_{client_id}"   
  if dirpath.exists() and dirpath.is_dir():
      shutil.rmtree(dirpath)
  dirpath.mkdir(parents=True, exist_ok=True)
  return dirpath


"""
Main execution of the program
"""
def main(relationship_manager, data):
  all_clients = data.client_id.unique().tolist()
  accuracies = []

  for client in all_clients:
    new_client = Client()
    new_client.id = client
    new_client.rm = User.objects.get(username = relationship_manager)
    new_client.dir_loc = setup_directory(relationship_manager, client)

    acc = perclient(relationship_manager, client, get_client_data(data, client))
    accuracies.append(acc)

    new_client.invested_amount_acc = acc['invested_amount']
    new_client.loan_to_invest_ratio_acc = acc['loan_to_invest_ratio']
    new_client.lowriskbucket_ratio_acc = acc['lowriskbucket_ratio']
    new_client.mediumriskbucket_ratio_acc = acc['mediumriskbucket_ratio']
    new_client.highriskbucket_ratio_acc = acc['highriskbucket_ratio']

    if Client.objects.filter(id=client).exists():
      Client.objects.filter(id=client).delete()
    new_client.save()

  return accuracies