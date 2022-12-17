import requests
import pandas as pd
import json

#test an api call
def test_api_call(counter, payload):
    url = 'http://127.0.0.1:8000/riskprediction/'
    headers = {'content-type': 'application/json'}
    response = requests.get(url, data=json.dumps(payload), headers=headers)
    print(json.dumps(payload))
    with open(f'Responses/{counter}.json', 'w+') as json_file:
        json_file.write(response.text)

def test(input_file):
    input = pd.read_csv(input_file)
    input['client_id'] = input['client_id'].astype(str)
    input['last_days'] = input['last_days'].astype(str)
    input['start_date'] = input['start_date'].astype(str)
    input['last_date'] = input['last_date'].astype(str)
    for i in range(len(input)):
        for cols in input.columns:
            if input.loc[i, cols] == 'None':
                input.loc[i, cols] = None
        if input.loc[i, 'client_id'] != None:    
            temp_client = [int(ele) for ele in input.loc[i, 'client_id'].split(' ')]
        if input.loc[i, 'last_days'] != None:
            input.loc[i, 'last_days'] = int(input.loc[i, 'last_days'])

        dic = {'client_id': temp_client, 'last_days':  input.loc[i, 'last_days'], 'start_date': input.loc[i, 'start_date'], 'last_date': input.loc[i, 'last_date']}
        payload = {}
        for key in dic.keys():
            if dic[key] != None:
                payload[key] = dic[key]
        test_api_call(i, payload )


if __name__ == '__main__':
    test("input.csv")
    # dic = {'client_id': None, 'last_days':  14, 'start_date': None, 'last_date': None}
    # test_api_call(0,dic)