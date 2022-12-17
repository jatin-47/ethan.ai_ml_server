import pandas as pd
import numpy as np
import datetime as dt
import os
from pathlib import Path
import tensorflow as tf
import json
import tensorflow

from django.conf import settings
from mlmodel.models import Client

mapper = {
  'invested_amount_new_30' : "invested_amount",
  'loantoinvest_new_30' : "loan_to_invest_ratio", 
  'bucketRatio1_new_30' : "lowriskbucket_ratio", 
  'bucketRatio2_new_30' : "mediumriskbucket_ratio", 
  'bucketRatio3_new_30' : "highriskbucket_ratio"
}

"""
Prediction for a particular feature
"""
def predict(dirpath, FEATURE, input_data, days, start_date, last_date):
  x_predict = input_data[(input_data.report_date >= start_date) & (input_data.report_date <= last_date)]
  x_predict.set_index("report_date", inplace = True)
 
  pred_labels=[]
  no_of_models = len(os.listdir(Path(dirpath,  mapper[FEATURE])))

  for i in range(no_of_models):
    ann = tf.keras.models.load_model(Path(dirpath, mapper[FEATURE] ,f"model_{i}.h5"))
    optimizerUsed = tensorflow.keras.optimizers.Nadam(learning_rate=0.0085, beta_1=0.9, beta_2=0.98, epsilon=1e-05, name="Nadam")
    ann.compile(optimizer=optimizerUsed, loss='binary_crossentropy', metrics=['Accuracy', 'Precision', 'Recall'])

    pred_labels_te = []
    pred_labels_te.append((ann.predict(x_predict.iloc[:]) > 0.5).astype(int))
    pred_labels.append(pred_labels_te)

  prediction=[]
  for i in range(no_of_models):
    flatten_list = [j for sub in pred_labels[i] for j in sub]
    prediction.append(flatten_list)

  df= pd.DataFrame(prediction)
  final_prediction = []
  for col in df.columns:
    l = np.array(df[col].to_list())
    final_prediction.append(np.bitwise_or.reduce(l)) 
  # return [i[0] for i in final_prediction]
  return final_prediction

"""
Adjust the input parameters according to the data and weekdays
"""
def adjust_input_params(input_data, last_days, start_date, last_date):
  if isinstance(start_date, str) == True:
    start_date = dt.date.fromisoformat(start_date)
  if isinstance(last_date, str) == True:
    last_date = dt.date.fromisoformat(last_date)

  today = input_data.iloc[-1]['report_date']

  #start_date <= today
  if start_date != None and start_date > today:
    start_date = today
  #last_date <= today
  if last_date != None and last_date > today:
    last_date = today

  #when last_date < start_date
  if start_date != None and last_date != None and last_date < start_date: 
    last_date = start_date

  if last_days == None and start_date == None and last_date == None:
    last_days, start_date, last_date = 1,today,today
  elif last_days == None and start_date == None:
    last_days, start_date = 1, last_date
  elif last_days == None and last_date == None:
    last_days, last_date = 1, start_date
  elif start_date == None and last_date == None:
    start_date, last_date = today - dt.timedelta(days=(last_days-1)), today
  elif last_days == None:
    last_days = abs(last_date - start_date).days + 1
  elif start_date == None:
    start_date = last_date - dt.timedelta(days=(last_days-1))
  elif last_date == None:
    last_date = start_date + dt.timedelta(days=(last_days-1))

  #if start_date is a saturday or a sunday change it to previous friday
  if start_date != None and start_date.weekday() in [5,6]:
    if start_date.weekday() == 5:
      start_date = start_date - dt.timedelta(days=1)
    elif start_date.weekday() == 6:
      start_date = start_date - dt.timedelta(days=2)
  
  #if last_date is a saturday or a sunday change it to next monday
  if last_date != None and last_date.weekday() in [5,6]:
    if last_date.weekday() == 5:
      last_date = last_date + dt.timedelta(days=2)
    elif last_date.weekday() == 6:
      last_date = last_date + dt.timedelta(days=1)

  last_days = len(input_data[(input_data.report_date >= start_date) & (input_data.report_date <= last_date)])

  return last_days, start_date, last_date

"""
This function is used to make predictions from the model for a particular client.
"""
def perclient(client_id, last_days, start_date, last_date):
  dirpath = Client.objects.get(id=client_id).dir_loc
  input_data = pd.read_csv(Path(dirpath, "transformed_input_data.csv"))
  input_data.report_date = pd.to_datetime(input_data.report_date).dt.date
  
  last_days, start_date, last_date = adjust_input_params(input_data, last_days, start_date, last_date)

  features_pred = {
    'invested_amount_new_30' : [], 
    'loantoinvest_new_30' : [], 
    'bucketRatio1_new_30' : [], 
    'bucketRatio2_new_30' : [], 
    'bucketRatio3_new_30' : []
  }

  for feature in features_pred.keys():
    features_pred[feature] = predict(dirpath, feature, input_data, last_days, start_date, last_date)

  response = {
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
  dic = response['t_prediction']
  for key in dic.keys():
    for idx, i in enumerate(dic[key]):
      dic[key][idx] = int(i)

  return response


"""
Main execution of the program
"""
def main(rm, client_id = None, last_days = None, start_date = None, last_date = None):
  all_clients = Client.objects.all().values_list('id', flat=True)

  if client_id == None:
    client_id = all_clients
  elif isinstance(client_id, list) == False:
    client_id = [client_id]

  available_clients = []
  for client in client_id:
    if client in all_clients:
      available_clients.append(client)

  if len(available_clients) == 0:
    return False

  response = []
  for client in available_clients:
    pred_path = Path(settings.CLIENT_MODELS_DIRECTORY, rm, f"client_{client}", "predictions.json")

    if pred_path.exists() and pred_path.is_file():
      data = json.load(open(Path(settings.CLIENT_MODELS_DIRECTORY, rm, f"client_{client}", "predictions.json")))
      response.append(data)
    else: 
      response.append(perclient(client, last_days=last_days, start_date=start_date, last_date=last_date))
  return response