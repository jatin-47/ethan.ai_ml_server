import os
import pandas as pd
from django.conf import settings

"""
Creating investment amount as a feature for a client
"""
def invested_amount(data_cp):
  data = data_cp.copy()
  data = data[['report_date', 'base_ccy_invested_amount']]
  data = data.sort_values(['report_date'], ascending=[True])

  store_data = {
    "report_date":[],
    "invested_amount":[]
  }
 
  dates = data.report_date.unique().tolist()
  for date in dates:
    total_invest = abs(data[(data.report_date == date)]['base_ccy_invested_amount'].sum())
    store_data["report_date"].append(date)
    store_data["invested_amount"].append(total_invest)
  df = pd.DataFrame(store_data)
  return df


"""
Creating 3 bucket ratios & 3 win loss ratios as features for a particular client
"""
def three_bucket_ratio_and_three_win_loss_ratios(data_cp):
  data = data_cp.copy()
  data = data[['report_date','asset_class','mtm_base_ccy','base_ccy_invested_amount','unrealisedpl']]
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
      "report_date":[],
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
    
    store_data["report_date"].append(date)
    store_data["bucketRatio1"].append(ratio1)
    store_data["bucketRatio2"].append(ratio2)
    store_data["bucketRatio3"].append(ratio3)
    store_data["win_loss_ratio_1"].append(win_loss_ratio_1)
    store_data["win_loss_ratio_2"].append(win_loss_ratio_2)
    store_data["win_loss_ratio_3"].append(win_loss_ratio_3)
  df = pd.DataFrame(store_data)
  return df


"""
Creating loan to invest ratio as a feature for a client
"""
def loan_to_invest_ratio(data_cp):
  data = data_cp.copy()
  data = data[['report_date','asset_class','mtm_base_ccy','base_ccy_invested_amount']]
  data = data.sort_values(['report_date'], ascending=[True])

  store_data = {
    "report_date":[],
    "loantoinvest":[]
  }
 
  dates = data.report_date.unique().tolist()
  for date in dates:
    loan_invest = abs(data[(data.report_date == date) & (data.asset_class=='Loan')]['mtm_base_ccy'].sum())
    total_invest = abs(data[(data.report_date == date)]['base_ccy_invested_amount'].sum())
    store_data["report_date"].append(date)
    store_data["loantoinvest"].append(loan_invest/total_invest)
  df = pd.DataFrame(store_data)
  return df


"""
Creating weighted bond yield as a feature for a client
"""
def wt_bond_yield(data_cp):
  data = data_cp.copy()
  data = data[['report_date','security_id','asset_class','base_ccy_invested_amount','Bond_Coupon']]
  data = data[data.asset_class == 'Bond']
  data.sort_values(["report_date"],inplace=True)
    
  storeDataDic = {
    "report_date":[],
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
    storeDataDic["wt_bond_yeild"].append(num)
  df = pd.DataFrame(storeDataDic)
  return df


"""
Creating weighted stock index as a feature for a client
"""
def wt_stock_index(data_cp, market, world):
  data = data_cp.copy()

  if 'Unnamed: 0' in market.columns:
    market.drop('Unnamed: 0', axis=1, inplace=True)
    
  data = data[['report_date','asset_class','base_ccy_invested_amount','country']]
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
    storeDataDic["wtStockInd"].append(num*100)
  df = pd.DataFrame(storeDataDic)
  return df


"""
Generating a dataframe containing all the features for a client
"""
def generate(client_data):
  f1 = invested_amount(client_data)
  f2to7 = three_bucket_ratio_and_three_win_loss_ratios(client_data)
  f8 = loan_to_invest_ratio(client_data)
  f9 = wt_bond_yield(client_data)

  market = pd.read_csv(os.path.join(settings.BASE_DIR, "World_Indices_Data.csv"))
  world = pd.read_csv(os.path.join(settings.BASE_DIR, "World Indices.csv"))
  f10 = wt_stock_index(client_data, market.copy(), world.copy())

  # Merge the different features
  df1  = pd.merge(left=f2to7, right=f8, on=['report_date'], how='outer')
  df2 = pd.merge(left=df1, right=f9, on=['report_date'], how='outer')
  df3 = pd.merge(left=df2, right=f10, on=['report_date'], how='outer')
  df4 = pd.merge(left=df3, right=f1, on=['report_date'], how='outer')

  df4.sort_values(["report_date"], inplace = True)
  df4.fillna(method='ffill', inplace=True)
  return df4
