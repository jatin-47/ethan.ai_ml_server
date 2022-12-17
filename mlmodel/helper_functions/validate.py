import pandas as pd

def alternate_bfill_ffill(df,column):
  df[f'{column}'].fillna(-1,inplace=True)
  # default=-1.0
  start=-1
  end=-1
  for index, row in df.iterrows():
    if(row[f'{column}']==-1):
      if(start==-1):
        start=index
        end=index
      else:
        end=index
    else:
      if(start==-1):
        continue
      # case when starting rows are NULL
      if(start==0):
        end = end+1
        for i in range(start,end):
          df.at[i,f'{column}'] = df.at[end,f'{column}']
      # when start and end are in between
      else:  
        start = start-1
        end = end+1
        mid=(start+end)//2
        for i in range(start,mid):
          df.at[i,f'{column}'] = df.at[start,f'{column}']
        for i in range(mid,end):
          df.at[i,f'{column}'] = df.at[end,f'{column}']
      start=-1
      end=-1
  if start!=-1:
    df.loc[df[f'{column}']==-1,f'{column}'] = df.at[start-1,f'{column}']
  return df



def validate(df):
  ################################ Basic checks ############################################
  error_message = []
  error_code = 0
  null_rows  = []
  # check column names
  expected_columns = ['report_date', 'client_id', 'asset_class', 'mtm_base_ccy', 'country','security_id', 'base_ccy_invested_amount', 'unrealisedpl', 'Bond_Coupon']
  columns_present = df.columns.tolist()
  for col in expected_columns:
    if col not in columns_present:
      error_message.append(f'{col} Column is not present')
      error_code = 1
  if error_code == 1:
    return (error_message,error_code,null_rows,df)    

  # null values
  required_columns = ['report_date', 'client_id', 'asset_class','unrealisedpl']
  null_columns = df.columns[df.isna().any()].tolist()
  for col in null_columns:
    if col in required_columns:
      error_message.append(f'{col} column contains null values')
      error_code = 1
    
  # print rows which have null values in essential columns
  df_req = df[required_columns]
  df_req = df_req[df_req.isnull().any(axis=1)]
  null_rows = df_req.to_dict('records')

  if error_code==1:
    return (error_message,error_code,null_rows,df)

  # if asset_class is bond in no coupon rate
  if df[ (df['Bond_Coupon'].isnull()) & (df['asset_class']=='Bond')].shape[0]>0:
    error_message.append("Bonds data has missing coupon rate")
  

  ############################## Filling missing values ###########################################
  # Make negative mtm_base values to NaN and do alternate ffiil and bfill till all data is filled

  df.loc[df['mtm_base_ccy'] < 0, 'mtm_base_ccy'] = None
  
  # alternate bfill and ffill
  df = alternate_bfill_ffill(df,'mtm_base_ccy')

  # For a client & a security make the country with max count as the country for all values in country col
  client_list = df.client_id.unique().tolist()
  for client in client_list:
    security_id_list = df[df.client_id==client].security_id.unique().tolist()
    for security_id in security_id_list:
      countries = df[(df.client_id == client) & (df.security_id == security_id)].country.tolist()
      country_mode = max(set(countries), key=countries.count)
      df.loc[(df.client_id == client) & (df.security_id == security_id), 'country'] = country_mode

  # For a particular security_id in asset_class == Bonds, take the max occurring coupon rate and fill the values of “Bond_Coupon” column with that.
  df['Bond_Coupon'].fillna((df['Bond_Coupon'].mean()), inplace=True)

  # For base_ccy_invested_amount do alternate ffiil and bfill till all data is filled
  df = alternate_bfill_ffill(df,'base_ccy_invested_amount')

  return (error_message,error_code,null_rows,df)