## Libraries
import pandas as pd
from numpy.random import seed
import datetime as dt

import psycopg2 as pg
import pandas.io.sql as psql

import tensorflow
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense

from django.conf import settings
"""## Fetching data

#### From the SQL database for a particular client
"""

# Connecting to the postgreSQL DB to fetch latest position history data as csv
def fetching_data_from_db(client_id):
  #####################################
  #Store these in environment variables in production
  DB_HOST = "ethan-202101-rpt.cppb6lkzjjge.ap-southeast-1.rds.amazonaws.com"
  DB_NAME = "vibgyor202101rpt"
  DB_USER = "saiml_rpta_user"
  DB_PASS = "saiml@2020"
  #####################################
  connection = pg.connect(host=DB_HOST, database=DB_NAME, user=DB_USER, password=DB_PASS, port=5455)
  dataframe = psql.read_sql(f'SELECT * FROM trades WHERE client_id={client_id}', connection)
  dataframe = dataframe.sort_values(['trade_date'], ascending=[True])
  # dataframe.trade_date = pd.to_datetime(dataframe.trade_date)
  return dataframe

"""#### From the locally stored csv in google drive for a particular client"""

def fetching_data_from_GD(client_id): 
  data = pd.read_csv(f"{settings.BASE_DIR}\processed_Portfolio.csv")
  data = data[data.client_id == client_id]
  data = data.sort_values(['report_date'], ascending=[True])
  return data

def get_all_client_ids():
  data = pd.read_csv(f"{settings.BASE_DIR}\processed_Portfolio.csv")
  return data.client_id.unique().tolist()

"""#### Fetching data for Bonds, Market and World Indices"""

def fetching_bonds_data():
  bonds = pd.read_csv(f"{settings.BASE_DIR}\\bonds_data.csv")
  return bonds

def fetching_market_data():
  market = pd.read_csv(f"{settings.BASE_DIR}\World_Indices_Data.csv")
  return market

def fetching_world_data():
  world = pd.read_csv(f"{settings.BASE_DIR}\World Indices.csv")
  return world

"""## Creating 10 features for a *particular* client

#### Creating investment amount as a feature for a client
"""

def invested_amount(data_cp):
  data = data_cp.copy()
  client = data.iloc[0]['client_id']
  data = data[['report_date','client_id', 'base_ccy_invested_amount']]
  data = data.sort_values(['report_date'], ascending=[True])

  store_data = {
    "client_id":[],
    "report_date":[],
    "invested_amount":[]
  }
 
  dates = data.report_date.unique().tolist()
  for date in dates:
    total_invest = abs(data[(data.report_date == date)]['base_ccy_invested_amount'].sum())
    store_data["client_id"].append(client)
    store_data["report_date"].append(date)
    store_data["invested_amount"].append(total_invest)
  df = pd.DataFrame(store_data)
  return df

"""#### Creating 3 bucket ratios & 3 win loss ratios as features for a particular client"""

def three_bucket_ratio_and_three_win_loss_ratios(data_cp):
  data = data_cp.copy()
  client = data.iloc[0]['client_id']
  data = data[['report_date', 'client_id','asset_class','mtm_base_ccy','base_ccy_invested_amount','unrealisedpl']]
  data = data.sort_values(['report_date'], ascending=[True])

  risk_Wt_dic = {
    "Equity" :  4,
    "Bond" : 1,                                                  
    "Derivative" : 5,                                             
    "Fixed Income Fund"  : 1,           
    "Equity Structured Product" : 5,
    "Equity structured product" : 5,
    "Equity Fund"  : 3,                 
    "Alternatives"  : 3,                
    "Fixed Income Funds" : 1,            
    "Fund" : 2,
    'FX Structured Products' : 5,
    'Loan' : 5, 
  }
  asset_classes = list(risk_Wt_dic.keys())

  # bucketRatio1 is for fund,fixed income funds
  # bucketRatio2 is for bond
  # bucketRatio3 is for assets having weights>=3
  store_data = {
      "client_id":[],
      "date":[],
      "bucketRatio1":[],
      "bucketRatio2":[],
      "bucketRatio3":[],
      "win_loss_ratio_1":[],
      "win_loss_ratio_2":[],
      "win_loss_ratio_3":[]
  }

  dates = data.report_date.unique().tolist()
  for date in dates:
    bucket1 = 1
    bucket2 = 1
    bucket3=1
    bucket1_upl = 0
    bucket2_upl = 0
    bucket3_upl=0

    perDayData = data[data.report_date==date]
    for asset_class in asset_classes:
      assetInvest = perDayData[perDayData.asset_class==asset_class]['base_ccy_invested_amount'].sum()
      assetupl = perDayData[perDayData.asset_class==asset_class]['unrealisedpl'].sum()

      if(risk_Wt_dic[asset_class]<3 and asset_class!="Bond") :
        bucket1 = bucket1 + assetInvest
        bucket1_upl = bucket1_upl + assetupl
      elif(asset_class=="Bond"):
        bucket2 = bucket2 + assetInvest
        bucket2_upl = bucket2_upl + assetupl
      else:
        bucket3 = bucket3 + assetInvest
        bucket3_upl = bucket3_upl + assetupl
    
    total_invest = perDayData['base_ccy_invested_amount'].sum()
    ratio1 = bucket1/total_invest
    ratio2 = bucket2/total_invest
    ratio3= bucket3/total_invest
    win_loss_ratio_1 = bucket1_upl/bucket1
    win_loss_ratio_2 = bucket2_upl/bucket2
    win_loss_ratio_3 = bucket3_upl/bucket3
    
    store_data["client_id"].append(client)
    store_data["date"].append(date)
    store_data["bucketRatio1"].append(ratio1)
    store_data["bucketRatio2"].append(ratio2)
    store_data["bucketRatio3"].append(ratio3)
    store_data["win_loss_ratio_1"].append(win_loss_ratio_1)
    store_data["win_loss_ratio_2"].append(win_loss_ratio_2)
    store_data["win_loss_ratio_3"].append(win_loss_ratio_3)
  df = pd.DataFrame(store_data)
  return df

"""#### Creating loan to invest ratio as a feature for a client"""

def loan_to_invest_ratio(data_cp):
  data = data_cp.copy()
  client = data.iloc[0]['client_id']
  data = data[['report_date','client_id','asset_class','mtm_base_ccy','base_ccy_invested_amount']]
  data = data.sort_values(['report_date'], ascending=[True])

  store_data = {
    "client_id":[],
    "date":[],
    "loantoinvest":[]
  }
 
  dates = data.report_date.unique().tolist()
  for date in dates:
    loan_invest = abs(data[(data.report_date == date) & (data.asset_class=='Loan')]['mtm_base_ccy'].sum())
    total_invest = abs(data[(data.report_date == date)]['base_ccy_invested_amount'].sum())
    store_data["client_id"].append(client)
    store_data["date"].append(date)
    store_data["loantoinvest"].append(loan_invest/total_invest)
  df = pd.DataFrame(store_data)
  return df

"""#### Creating weighted bond yield as a feature for a client"""

def wt_bond_yield(data_cp, bonds_cp):
  data = data_cp.copy()
  bonds = bonds_cp.copy()

  # bonds = pd.read_csv('/content/drive/MyDrive/2yr Data/Bonds Data/bonds_data.csv')
  if 'Unnamed: 0' in bonds.columns:
    bonds.drop(['Unnamed: 0'], axis = 1, inplace = True) 
    
  bonds.sort_values('Coupon', inplace=True, ascending=True) 
  bonds.reset_index(drop=True, inplace=True)
  bonds_list = bonds.Coupon.tolist()
  num = 0
  for row in range(len(bonds)):
    if bonds.loc[row, 'Coupon'] != -1:
      num = row
      break
  avg = sum(bonds_list[num:])/len(bonds_list[num:])
  bonds.loc[bonds['Coupon'] == -1, 'Coupon'] = avg
  bonds = bonds.set_index('Names').T.to_dict('list')
  bonds = {k:v[0] for (k,v) in bonds.items()}

  client = data.iloc[0]['client_id']
  data = data[['report_date','client_id', 'security_id','asset_class','base_ccy_invested_amount']]
  data = data[data.asset_class == 'Bond']
  data['Bond_Coupon'] = data['security_id'].map(bonds)
  data.sort_values(["report_date"],inplace=True)
  
  data.loc[data.asset_class == 'Bond'] = data.loc[data.asset_class == 'Bond'].fillna(avg)
  
  storeDataDic = {
    "report_date":[],
    "client_id":[],
    "wt_bond_yeild":[]
  }

  dates = data.report_date.unique().tolist()
  for date in dates:
    tempData = data[(data.report_date == date)]
    num=0
    total_amt = tempData.base_ccy_invested_amount.sum() + 1
    for index, row in tempData.iterrows():
      amt = row['base_ccy_invested_amount']/total_amt
      coupon = row['Bond_Coupon']
      num = num + amt*coupon
    storeDataDic["report_date"].append(date)
    storeDataDic["client_id"].append(client)
    storeDataDic["wt_bond_yeild"].append(num)
  df = pd.DataFrame(storeDataDic)
  return df

"""#### Creating weighted stock index as a feature for a client"""

def wt_stock_index(data_cp, market_cp, world_cp):
  data = data_cp.copy()
  market = market_cp.copy()
  world = world_cp.copy()

  # market = pd.read_csv('/content/drive/MyDrive/2yr Data/World Market Indices/World_Indices_Data.csv')
  # world = pd.read_csv('/content/drive/MyDrive/2yr Data/World Market Indices/World Indices.csv')
  client = data.iloc[0]['client_id']

  if 'Unnamed: 0' in market.columns:
    market.drop('Unnamed: 0', axis=1, inplace=True)
    
  data = data[['report_date','client_id','asset_class','base_ccy_invested_amount','country']]

  # Taking only equity
  data = data[data.asset_class == 'Equity']

  # Drop euro countries
  data = data.drop(data[data.country == "Euro Member Countries"].index)
  data.sort_values(["report_date"],inplace=True)

  world = world.set_index('Country').T.to_dict('list')
  world = {k.strip():v[0] for (k,v) in world.items()}

  market.set_index(['Date'], inplace=True)
  perMarket = market.pct_change(periods = 60)
  perMarket['Date'] = perMarket.index

  storeDataDic = {
    "report_date":[],
    "client_id":[],
    "wtStockInd":[]
  }

  dates = data.report_date.unique().tolist()
  for date in dates:
    tempData = data[(data.report_date == date)]
    num=0
    total_amt = tempData.base_ccy_invested_amount.sum()
    for index, row in tempData.iterrows():
      country = row['country']
      amt = row['base_ccy_invested_amount']/total_amt
      ind = world[country]
      # if(ind == "^SET.BK"):
      #   ind = " SET"
      indChange = perMarket[perMarket.Date == date][ind][0]
      num = num + amt*indChange
    storeDataDic["report_date"].append(date)
    storeDataDic["client_id"].append(client)
    storeDataDic["wtStockInd"].append(num*100)
  df = pd.DataFrame(storeDataDic)
  return df

"""## Merging all features into one dataframe"""

def data_creation(invested_amount, three_bucket_ratio_and_three_win_loss_ratios, loan_to_invest_ratio, wt_bond_yield, wt_stock_index):
  invested_amount_cp = invested_amount.copy()
  three_bucket_ratio_and_three_win_loss_ratios_cp = three_bucket_ratio_and_three_win_loss_ratios.copy()
  loan_to_invest_ratio_cp = loan_to_invest_ratio.copy()
  wt_bond_yield_cp = wt_bond_yield.copy()
  wt_stock_index_cp = wt_stock_index.copy()

  # Merge the different features
  df1  = pd.merge(left=three_bucket_ratio_and_three_win_loss_ratios_cp, right=loan_to_invest_ratio_cp, on=['client_id', 'date'], how='outer')
  df1.rename(columns={'date': 'report_date'},inplace=True, errors='raise')

  df2 = pd.merge(left=df1, right=wt_bond_yield_cp, on=['client_id', 'report_date'], how='outer')
  df3 = pd.merge(left=df2, right=wt_stock_index_cp, on=['client_id', 'report_date'], how='outer')
  df4 = pd.merge(left=df3, right=invested_amount_cp, on=['client_id', 'report_date'], how='outer')

  df4.sort_values(["client_id","report_date"], inplace = True)
  df4.fillna(method='ffill', inplace=True)
  return df4



"""## Input & Target data creation

#### Input Data creation
"""

def dataCreation_NN_input(all_features):
  client = all_features.iloc[0]['client_id']

  df = all_features.copy(deep=True)
  # df.drop(['Unnamed: 0'], axis = 1, inplace = True) 

  df_array = pd.DataFrame()

  df.reset_index(drop=True, inplace=True)
  columns = ['bucketRatio1','bucketRatio2','bucketRatio3','win_loss_ratio_1','win_loss_ratio_2','win_loss_ratio_3','loantoinvest','wt_bond_yeild','wtStockInd', 'invested_amount']
  for col in columns:
    new_col = f"{col}_new"
    df[new_col] = 0
    for idx, row in df.iterrows():
      if(idx<30): continue
      if  df.loc[idx,col] <= df.loc[idx-30,col]:
        df.loc[idx,new_col] = -1
      else:
        df.loc[idx,new_col] = 1
  df.drop(df.loc[0:29].index, inplace=True)
  df.drop(columns, axis = 1, inplace = True) 
  df_array = df_array.append(df)
  df_array.reset_index(drop=True, inplace=True)
    
  return df_array

"""#### Target Data creation"""

def dataCreation_NN_output(NN_input):
  df = NN_input.copy(deep=True)
 
  df_array = pd.DataFrame()
  columns = df.columns.tolist() 
  columns.remove('client_id')
  columns.remove('report_date')

  df.reset_index(drop=True, inplace=True)
  for col in columns:
    df[col] = df[col].shift(-30)
  df_array = df_array.append(df)
  df_array.dropna(inplace=True)
  return df_array

"""## Model Training and prediction for a particular feature"""

def predict_days(FEATURE, input_data, output_data, days, start_date, last_date):
  input_data_cp = input_data.copy()
  output_data_cp = output_data.copy()
  
  client = input_data_cp.iloc[0]['client_id']

  del input_data_cp['client_id']
  del output_data_cp['client_id']

  x_predict = input_data_cp[(input_data_cp.report_date >= start_date) & (input_data_cp.report_date <= last_date)]

  input_data_cp.set_index("report_date", inplace = True)
  output_data_cp.set_index("report_date", inplace = True)
  x_predict.set_index("report_date", inplace = True)

  output_data_cp = output_data_cp.astype({'invested_amount_new': 'int', 'bucketRatio1_new': 'int', 'bucketRatio2_new': 'int', 'bucketRatio3_new': 'int', 'win_loss_ratio_1_new': 'int', 'win_loss_ratio_2_new': 'int', 'win_loss_ratio_3_new': 'int', 'loantoinvest_new': 'int', 'wt_bond_yeild_new': 'int', 'wtStockInd_new': 'int'})
  
  input_data_cp = input_data_cp.iloc[:-30]
 
  output_data_cp = output_data_cp[[FEATURE]]
  output_data_cp['label'] = output_data_cp[FEATURE]

  for row in range(len(output_data_cp)):
    if output_data_cp.iloc[row, 0] == -1:
      output_data_cp.iloc[row, 1]= 0
    else:
      output_data_cp.iloc[row, 1]= 1

  del output_data_cp[FEATURE]
  
  prime=[2,7,23,53,71,89]
  pred_labels=[]

  for i in prime:
    seed(3*i)
    tensorflow.random.set_seed(9*i)
    # Neural Network ##############################################
    ann = Sequential()
    ann.add(Input(shape=(10,), name='Input-Layer')) 
    ann.add(Dense(20, activation='relu', name='Hidden-Layer1'))
    ann.add(Dense(30, activation='relu', name='Hidden-Layer2'))
    ann.add(Dense(20, activation='relu', name='Hidden-Layer3'))
    ann.add(Dense(10, activation='relu', name='Hidden-Layer4'))
    ann.add(Dense(5, activation='relu', name='Hidden-Layer5'))
    ann.add(Dense(1, activation='sigmoid', name='Output-Layer'))
    ###############################################################

    ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['Accuracy', 'Precision', 'Recall'])
    ann.fit(input_data_cp, output_data_cp, epochs=150, verbose=0) 
    pred_labels_te = []
    pred_labels_te.append((ann.predict(x_predict.iloc[:]) > 0.5).astype(int))
    pred_labels.append(pred_labels_te)

  prediction=[]
  for i in range(0,len(prime)):
    flatten_list = [j for sub in pred_labels[i] for j in sub]
    prediction.append(flatten_list)
  df= pd.DataFrame(prediction)
  final_prediction= df.mode().loc[0].to_list() 

  train_prediction = [i[0] for i in (ann.predict(input_data_cp.iloc[:]) > 0.5).astype(int)]
  ans = pd.DataFrame(train_prediction, columns = ["Predicted"])
  ans['Truth'] = output_data_cp.values
  ans['accuracy'] = 0
  ans.loc[ans['Predicted'] == ans['Truth'], ['accuracy']] = 1
  acc = (ans[ans.accuracy == 1].shape[0] / ans.shape[0])*100

  return {"lst" : [i[0] for i in final_prediction], "acc" : acc}

"""## PerClient Function"""

def perclient(client_id, last_days, start_date, last_date):
  #fetching data for a client_id
  fetched_data = fetching_data_from_GD(client_id)
  fetched_data_cp = fetched_data.copy()

  # getting features dataframe
  f1 = invested_amount(fetched_data_cp)
  f2to7 = three_bucket_ratio_and_three_win_loss_ratios(fetched_data_cp)
  f8 = loan_to_invest_ratio(fetched_data_cp)

  #fetching bonds data 
  bonds = fetching_bonds_data()
  f9 = wt_bond_yield(fetched_data_cp, bonds.copy())

  #fetching market and world data
  market = fetching_market_data()
  world = fetching_world_data()
  f10 = wt_stock_index(fetched_data_cp, market.copy(), world.copy())

  #merging all features
  merged_features = data_creation(f1, f2to7, f8, f9, f10)

  #input data
  input_data = dataCreation_NN_input(merged_features)
  #output data
  output_data = dataCreation_NN_output(input_data.copy())

  input_data.report_date = pd.to_datetime(input_data.report_date).dt.date
  output_data.report_date = pd.to_datetime(output_data.report_date).dt.date

  #####################################################################
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
  #####################################################################

  #targets to be predicted
  features_pred = {
      'invested_amount_new' : [], 
      'loantoinvest_new' : [], 
      'bucketRatio1_new' : [], 
      'bucketRatio2_new' : [], 
      'bucketRatio3_new' : []
  }

  features_acc = {
      'invested_amount_new' : 0, 
      'loantoinvest_new' : 0, 
      'bucketRatio1_new' : 0, 
      'bucketRatio2_new' : 0, 
      'bucketRatio3_new' : 0
  }

  # training model for each target
  for feature in features_pred.keys():
    pred = predict_days(feature, input_data, output_data, last_days, start_date, last_date)
    features_pred[feature] = pred["lst"]
    features_acc[feature] = pred["acc"]

  # making the response
  response = {
    "client_id": int(client_id),
    "Description" : "The prediction is for 30 days ahead of the given date.",
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
    "Accuracy" : {
      "invested_amount": features_acc['invested_amount_new'],
      "loan_to_invest_ratio": features_acc['loantoinvest_new'],
      "lowriskbucket_ratio": features_acc['bucketRatio1_new'],
      "mediumriskbucket_ratio": features_acc['bucketRatio2_new'],
      "highriskbucket_ratio": features_acc['bucketRatio3_new']
    },
    "t_prediction": {
      "invested_amount": features_pred['invested_amount_new'],
      "loan_to_invest_ratio": features_pred['loantoinvest_new'],
      "lowriskbucket_ratio": features_pred['bucketRatio1_new'],
      "mediumriskbucket_ratio": features_pred['bucketRatio2_new'],
      "highriskbucket_ratio": features_pred['bucketRatio3_new']
    }
  }
  dic = response['t_prediction']
  for key in dic.keys():
    for idx, i in enumerate(dic[key]):
      dic[key][idx] = int(i)

  return response

"""## Main Function"""

def main(client_id = None, last_days = None, start_date = None, last_date = None):
  all_clients = get_all_client_ids()

  if client_id == None:
    client_id = all_clients
  elif isinstance(client_id, list) == False:
    client_id = [client_id]

  available_clients = []
  for client in client_id:
    if client in all_clients:
      available_clients.append(client)

  if len(available_clients) == 0:
    print("No client found")
    return {"messeage" : "No client found"}

  response = []

  for client in available_clients:
    response.append(perclient(client, last_days=last_days, start_date=start_date, last_date=last_date))

  return response