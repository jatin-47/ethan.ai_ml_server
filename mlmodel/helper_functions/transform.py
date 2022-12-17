import pandas as pd
offset_days = [5,15,30]

def dataCreation_NN_input(df):
  print("Creating NN input features")
  df_array = pd.DataFrame()
  df.reset_index(drop=True, inplace=True)

  columns = ['bucketRatio1','bucketRatio2','bucketRatio3','win_loss_ratio_1','win_loss_ratio_2','win_loss_ratio_3','loantoinvest','wt_bond_yeild','wtStockInd', 'invested_amount']
  max_offset = max(offset_days)
  for col in columns:
    for j in offset_days:
      new_col = f"{col}_new_{j}"
      df[new_col] = 0
      for idx, row in df.iterrows():
        if(idx<max_offset): continue
        if  df.loc[idx,col] <= df.loc[idx-j,col]:
          df.loc[idx,new_col] = -1
        else:
          df.loc[idx,new_col] = 1
  df.drop(df.loc[0:(max_offset-1)].index, inplace=True)
  df.drop(columns, axis = 1, inplace = True) 
  # df_array = df_array.append(df)
  df_array = pd.concat([df_array, df])
  df_array.reset_index(drop=True, inplace=True)
  return df_array


def dataCreation_NN_output(NN_input):
  df = NN_input.copy(deep=True)
 
  df_array = pd.DataFrame()
  columns = df.columns.tolist() 
  columns.remove('report_date')

  df.reset_index(drop=True, inplace=True)
  for col in columns:
    df[col] = df[col].shift(-30)
  df_array = pd.concat([df_array, df])
  df_array.dropna(inplace=True)
  return df_array
