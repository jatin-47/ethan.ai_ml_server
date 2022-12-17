from matplotlib import transforms
from rest_framework.views import APIView
from rest_framework.parsers import JSONParser
from django.shortcuts import redirect, render
from django.views import View
from django.http import JsonResponse
import os
import numpy as np
from django.core.mail import send_mail
import pandas as pd
import timeit
import time
from datetime import datetime
from pathlib import Path
import shutil
import plotly.express as px
from plotly.offline import plot

from django.conf import settings
import mlmodel.helper_functions.Plot as plot3D
import mlmodel.helper_functions.validate as validate
import mlmodel.helper_functions.train as train
import mlmodel.helper_functions.predict as predict
import mlmodel.helper_functions.transform as transform

from .models import *
from django.contrib.auth import authenticate, login, logout
from django.db import IntegrityError

class Ping(APIView):
    def get(self, request):
        return JsonResponse({'message': 'pong'})

class Login(View):
    def get(self, request):
        return render(request, "login.html")

    def post(self, request):
        # Attempt to sign user in
        username = request.POST.get("username")
        password = request.POST.get("password")
        user = authenticate(request, username=username, password=password)

        # Check if authentication successful
        if user is not None:
            login(request, user)
            if "next" in request.POST:
                return redirect(request.POST.get('next'))
            else:
                return redirect("dashboard")
        else:
            return render(request, "login.html", {
                "message": "*Invalid username and/or password."
            })

class Logout(View):
    def get(self, request):
        logout(request)
        return render(request, "login.html", {
            "message": "*You have successfully logged out. 👍"
        })

class Register(View):
    def get(self, request):
        return render(request, "register.html")

    def post(self, request):
        username = request.POST["username"]
        email = request.POST["email"]

        # Ensure password matches confirmation
        password = request.POST["password"]
        confirmation = request.POST["confirmation"]
        if password != confirmation:
            return render(request, "register.html", {
                "message": "*Passwords must match."
            })

        # Attempt to create new user
        try:
            user = User.objects.create_user(username, email, password)
            user.save()
        except IntegrityError:
            return render(request, "register.html", {
                "message": "*Username already taken."
            })
        login(request, user)
        return redirect("dashboard")
    
############################################################################################

class Dashboard(View):
    def get(self, request):
        return render(request, "dashboard.html")

class UploadData(View):
    def get(self, request):
        return render(request, 'upload.html')
    
    def post(self, request):
        csv_file = request.FILES.get('pos_csv')
        if not csv_file:
            return render(request, 'upload.html', {'error': 'Upload a file!'})
        data = pd.read_csv(csv_file)  
        error_msg, validation, null_rows ,modified_df = validate.validate(data)
        if validation == 1:
            validation = False
        else:
            validation = True

        if not validation:
            print(null_rows)
            return render(request, 'upload.html', {'error': error_msg, 'null_rows': null_rows})
        data = modified_df
        
        client_ids = data.client_id.unique().tolist()

        temp_dirpath = Path(settings.TEMP) / "temp_data"  
        if temp_dirpath.exists() and temp_dirpath.is_dir():
            shutil.rmtree(temp_dirpath)
        temp_dirpath.mkdir(parents=True, exist_ok=True)
        data.to_csv(Path(temp_dirpath, "uploaded_data.csv"), index = False)
        
        return render(request, 'preview.html', {'client_options': client_ids, 'error' : error_msg})


class TrainModel(View):
    def post(self, request):
        start = timeit.default_timer()
        
        filepath = Path(settings.TEMP) / "temp_data" / "uploaded_data.csv"
        if filepath.is_file():
            data = pd.read_csv(filepath)
        else:
            return render(request, 'upload.html', {'error': 'Upload a file!'})

        clients = [int(id) for id in request.POST.getlist('client_id')]
        if len(clients) != 0:
            data = data[data.client_id.isin(clients)]
        
        relationship_manager = request.user
        accuracies = train.main(str(relationship_manager), data)

        stop = timeit.default_timer()
        
        # subject = 'Training Completed'
        # message = f"Hi {request.user.username}, Training is completed (time taken = {time.strftime('%H:%M:%S', time.gmtime(stop - start))}), login to your dashboard to see the predictions. See the accuracies below:\n\n{accuracies}"
        # email_from = settings.EMAIL_HOST_USER
        # recipient_list = [request.user.email, ]
        # send_mail( subject, message, email_from, recipient_list )

        return render(request, 'train.html', {'clients': accuracies, "time": time.strftime('%H:%M:%S', time.gmtime(stop - start))})

def reject_outliers(data, m = 5.):
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d/mdev if mdev else 0.
    newdata = data[s<m]
    return newdata

def makePlotly3DGraph(inputDataframe, identifierColumnName, xAxisFeatureName, yAxisFeatureName, zAxisFeatureName, sizeFeatureName,
                      symbolFeatureName, saveFileName, numberOfElementsNeeded = None, elementsFilterFeature = None, xlim_lower = None, xlim_upper= None,
                      ylim_lower = None, ylim_upper = None, zlim_lower = None, zlim_upper = None, removeOutliers = False):

    inbuiltPalette = ['aqua', 'black', 'blue',
            'blueviolet', 'brown', 'cadetblue',
            'cornflowerblue', 'cyan', 'darkblue', 'darkcyan',
            'darkgoldenrod', 'darkgreen',
            'darkkhaki', 'darkmagenta', 'darkolivegreen', 'darkorange',
            'darkorchid', 'darkred', 'darksalmon', 'darkseagreen',
            'darkslateblue', 'darkslategray', 'darkslategrey',
            'darkturquoise', 'darkviolet', 'deeppink', 'deepskyblue',
            'dimgray', 'dimgrey', 'dodgerblue', 'firebrick',
            'floralwhite', 'forestgreen', 'fuchsia', 'gainsboro',
            'ghostwhite', 'gold', 'goldenrod', 'gray', 'grey', 'green',
            'greenyellow', 'honeydew', 'hotpink', 'indianred', 'indigo',
            'ivory', 'khaki', 'lavender', 'lavenderblush', 'lawngreen',
            'lemonchiffon', 'lightblue', 'lightcoral', 'lightcyan',
            'lightgoldenrodyellow', 'lightgray', 'lightgrey',
            'lightgreen', 'lightpink', 'lightsalmon', 'lightseagreen',
            'lightskyblue', 'lightslategray', 'lightslategrey',
            'lightsteelblue', 'lightyellow', 'lime', 'limegreen',
            'linen', 'magenta', 'maroon', 'mediumaquamarine',
            'mediumblue', 'mediumorchid', 'mediumpurple',
            'mediumseagreen', 'mediumslateblue', 'mediumspringgreen',
            'mediumturquoise', 'mediumvioletred', 'midnightblue',
            'mintcream', 'mistyrose', 'moccasin', 'navajowhite', 'navy',
            'oldlace', 'olive', 'olivedrab', 'orange', 'orangered',
            'orchid', 'palegoldenrod', 'palegreen', 'paleturquoise',
            'palevioletred', 'papayawhip', 'peachpuff', 'peru', 'pink',
            'plum', 'powderblue', 'purple', 'red', 'rosybrown',
            'royalblue', 'rebeccapurple', 'saddlebrown', 'salmon',
            'sandybrown', 'seagreen', 'seashell', 'sienna', 'silver',
            'skyblue', 'slateblue', 'slategray', 'slategrey', 'snow',
            'springgreen', 'steelblue', 'tan', 'teal', 'thistle', 'tomato',
            'turquoise', 'violet', 'wheat', 'white', 'whitesmoke',
            'yellow', 'yellowgreen', 'blanchedalmond', 'azure',
            'beige', 'bisque', 'aliceblue', 'antiquewhite', 'cornsilk',
            'burlywood', 'aquamarine', 'chartreuse', 'crimson', 'chocolate',
            'darkgray', 'darkgrey', 'coral']

    fixed_symbols = ['circle', 'circle-open', 'cross', 'diamond','diamond-open', 'square', 'square-open', 'x']

    xlim_l = xlim_lower
    xlim_u = xlim_upper
    ylim_l = ylim_lower
    ylim_u = ylim_upper
    zlim_l = zlim_lower
    zlim_u = zlim_upper

    filterFeatureValues = None

    if numberOfElementsNeeded != None:
        if (elementsFilterFeature == None):
            elementsFilterFeature = xAxisFeatureName
        filterFeatureValues = inputDataframe[elementsFilterFeature].unique()
        filterFeatureValues.sort()
        filterFeatureSortedValues = filterFeatureValues[-numberOfElementsNeeded:]
      
        inputDataframe = inputDataframe[inputDataframe[elementsFilterFeature].isin(filterFeatureSortedValues)]
       

    xRange = inputDataframe[xAxisFeatureName]
  
    if removeOutliers:
        xRange = reject_outliers(xRange, 6.)
   
    yRange = inputDataframe[yAxisFeatureName]
   
    if removeOutliers:
        yRange = reject_outliers(yRange, 6.)
    
    zRange = inputDataframe[zAxisFeatureName]
   
    if removeOutliers:
        zRange = reject_outliers(zRange, 6.)
    
    if xlim_lower == None:
        minValue = min(xRange)
        maxValue = max(xRange)
        xlim_l = minValue - max((maxValue - minValue) * 0.075, 0.1)
    if ylim_lower == None:
        minValue = min(yRange)
        maxValue = max(yRange)
        ylim_l = minValue - max((maxValue - minValue) * 0.075, 0.1)
    if zlim_lower == None:
        minValue = min(zRange)
        maxValue = max(zRange)
        zlim_l = minValue - max((maxValue - minValue) * 0.075, 0.1)
    if xlim_upper == None:
        minValue = min(xRange)
        maxValue = max(xRange)
        xlim_u = maxValue + max((maxValue - minValue) * 0.075, 0.1)
    if ylim_upper == None:
        minValue = min(yRange)
        maxValue = max(yRange)
        ylim_u = maxValue + max((maxValue - minValue) * 0.075, 0.1)
    if zlim_upper == None:
        minValue = min(zRange)
        maxValue = max(zRange)
        zlim_u = maxValue + max((maxValue - minValue) * 0.075, 0.1)

    fig = px.scatter_3d(inputDataframe, x=xAxisFeatureName, y=yAxisFeatureName, z=zAxisFeatureName,
                        color=identifierColumnName, size=sizeFeatureName, size_max=3,
                        symbol=symbolFeatureName, opacity=0.7, color_discrete_sequence=inbuiltPalette,
                        symbol_sequence=fixed_symbols)

    fig.update_traces(marker=dict(size=3,
                                  line=dict(width=0,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    fig.update_layout(
        scene=dict(
            xaxis=dict(nticks=4, range=[xlim_l, xlim_u], ),
            yaxis=dict(nticks=4, range=[ylim_l, ylim_u], ),
            zaxis=dict(nticks=4, range=[zlim_l, zlim_u], ),
            xaxis_showspikes=False,
            yaxis_showspikes=False
        ),
        scene_aspectmode='cube',
        width=1100,
        margin=dict(r=20, l=10, b=10, t=10))

    if saveFileName is not None:
        split_tup = os.path.splitext(os.path.abspath(saveFileName))
        pngFile = split_tup[0] + ".png"
        fig.write_image(pngFile)
    else:
        try:
            return fig
        except:
            print("Error encountered ")

class PredictModel(View):
    def get(self, request):
        start = timeit.default_timer()

        show_pred = True
        client_options = Client.objects.all().values_list('id', flat=True)
        if len(request.GET) == 0:
            show_pred = False
            return render(request, 'predict.html', {'client_options': client_options,'show_pred': show_pred})

        input = {"client_id": None, "last_days": None, "start_date": None, "last_date": None}

        input['client_id'] = [int(id) for id in request.GET.getlist('client_id')]
        input['last_days'] = request.GET.get('last_days')
        input['start_date'] = request.GET.get('start_date')
        input['last_date'] = request.GET.get('last_date')

        if len(request.GET.getlist('client_id')) == 0:
            input['client_id'] = None
        if request.GET.get('last_days') == '':
            input['last_days'] = None
        if request.GET.get('start_date') == '':
            input['start_date'] = None
        if request.GET.get('last_date') == '':
            input['last_date'] = None

        if input['last_days'] is not None:
            input['last_days'] = int(input['last_days'])
        
        relationship_manager = request.user
        predictions = predict.main(str(relationship_manager), input['client_id'], input['last_days'], input['start_date'], input['last_date'])
        if predictions == False:
            return render(request, 'predict.html', {'client_options': client_options,'show_pred': show_pred,"time": time.strftime('%H:%M:%S', time.gmtime(stop - start))})

        clients = []
        plot_divs = []
        for client in predictions:
            clients.append(client['client_id'])

            dirpath = Client.objects.get(id=client['client_id']).dir_loc
            input_data = pd.read_csv(Path(dirpath, "input_data.csv"))
            transformed_input_data = pd.read_csv(Path(dirpath, "transformed_input_data.csv"))

            input_data.report_date = pd.to_datetime(input_data.report_date).dt.date   
            transformed_input_data.report_date = pd.to_datetime(transformed_input_data.report_date).dt.date

            transformed_output_data = transform.dataCreation_NN_output(transformed_input_data)
            transformed_output_data.report_date = pd.to_datetime(transformed_output_data.report_date).dt.date

            start_date = datetime.strptime(f"{client['start_date']['year']}-{client['start_date']['month']}-{client['start_date']['day']}", '%Y-%m-%d').date()
            last_date = datetime.strptime(f"{client['end_date']['year']}-{client['end_date']['month']}-{client['end_date']['day']}", '%Y-%m-%d').date()
            x_predict = input_data[(input_data.report_date >= start_date) & (input_data.report_date <= last_date)]
            
            six_plots = []
            mapper = {           
                'invested_amount' : 'invested_amount_new_30',
                'loan_to_invest_ratio' : 'loantoinvest_new_30',
                'lowriskbucket_ratio' : 'bucketRatio1_new_30',
                'mediumriskbucket_ratio' : 'bucketRatio2_new_30',
                'highriskbucket_ratio' : 'bucketRatio3_new_30'
            }
            for feature in client['t_prediction'].keys():
                feature_data = client['t_prediction'][feature]
                pred = [-1 if x==0 else 1 for x in feature_data]
                tl = transformed_output_data.iloc[-len(pred):][mapper[feature]].to_list()
                acc = []
                for i in range(len(tl)):
                    if(tl[i]==pred[i]):
                        acc.append('1')
                    else:
                        acc.append('0')
                df = pd.DataFrame({feature: tl, 'Accuracy': acc})
                fig = px.bar(df, y=feature, range_y=[-1,1], color='Accuracy',color_discrete_map={'1':'green', '0':'red'},labels={'index': "days"})
                
                # pred_s = ["DOWN" if x==-1 else "UP" for x in pred]
                # df = pd.DataFrame({feature: pred, 'Legend': pred_s})
                # fig = px.bar(df, y=feature, range_y=[-1,1], color='Legend', color_discrete_map={'UP':'green', 'DOWN':'red'}, labels={'index': "days"})
                fig.update_layout(width=500, height=300)
                six_plots.append(plot(fig, output_type='div', include_plotlyjs=False, show_link=False, link_text="")) 
            # last graph
            invested_amount = x_predict['invested_amount'].tolist()
            x_predict['wt_risk_port'] = (1*x_predict['bucketRatio1'] + 3*x_predict['bucketRatio2']+ 9*x_predict['bucketRatio3'])/13
            wt_risk_port = x_predict['wt_risk_port'].tolist()
            days = list(range(1, len(invested_amount)+1))
            df = pd.DataFrame({'invested_amount': invested_amount, 'wt_risk_port': wt_risk_port,'days':days})
            
            fig = makePlotly3DGraph(inputDataframe=df, identifierColumnName=None,
                            xAxisFeatureName='wt_risk_port', yAxisFeatureName='days',
                            zAxisFeatureName='invested_amount', sizeFeatureName=None, symbolFeatureName=None,
                            saveFileName=None, xlim_lower=None, xlim_upper=None, ylim_lower=None, ylim_upper=None,
                            zlim_lower=None, zlim_upper=None, numberOfElementsNeeded=None, elementsFilterFeature = None,
                            removeOutliers = False)
            # fig = px.line(df, x="invested_amount", y="wt_risk_port", text="days")
            # fig.update_traces(textposition="bottom right")
            fig.update_layout(width=600, height=400)
            last_graph = plot(fig, output_type='div',include_plotlyjs=False,show_link=False, link_text="")

            six_plots.append(last_graph)
            plot_divs.append((six_plots, client['client_id'], client['start_date'], client['end_date'], client['working_days']))
        stop = timeit.default_timer()
        return render(request, 'predict.html', context={'clients': clients,'plot_divs': plot_divs,'client_options': client_options,'show_pred': show_pred,"time": time.strftime('%H:%M:%S', time.gmtime(stop - start))})

class RiskPredictionPlot(View):
    def get(self, request):
        plot_div = plot3D.main()
        # soup = BeautifulSoup(plot_div, 'html.parser')
        # soup.div['style'] = 'height:95vh;' 
        return render(request, "plot.html", context={'plot_div': plot_div})

##########################################################################

class EfficientFrontier(APIView):
    def get(self, request):
        #get date,equity_list,bonds_list,cryptocurrency_list,currency_list from body
        requestData = JSONParser().parse(request)
        date=requestData['date']
        equity_list=requestData['equity_list']
        bonds_list=requestData['bonds_list']
        cryptocurrency_list=requestData['cryptocurrency_list']
        currency_list=requestData['currency_list']
        print(date,equity_list,bonds_list,cryptocurrency_list,currency_list)
        return JsonResponse({'message': 'efficient frontier'})

class Recommendation(APIView):
    def get(self,request):
        requestData = JSONParser().parse(request)
        date=requestData['date']
        equity_list=requestData['equity_list']
        bonds_list=requestData['bonds_list']
        cryptocurrency_list=requestData['cryptocurrency_list']
        currency_list=requestData['currency_list']
        risk=requestData['risk']
        returns=requestData['returns']
        print(date,equity_list,bonds_list,cryptocurrency_list,currency_list,risk,returns)
        return JsonResponse({'message': 'recommendation'})

class Backtests(APIView):
    def get(self,request):
        requestData = JSONParser().parse(request)
        starting_date=requestData['starting_date']
        ending_date=requestData['ending_date']
        equity_list=requestData['equity_list']
        bonds_list=requestData['bonds_list']
        cryptocurrency_list=requestData['cryptocurrency_list']
        currency_list=requestData['currency_list']
        print(starting_date,ending_date,equity_list,bonds_list,cryptocurrency_list,currency_list)
        return JsonResponse({'message': 'backtests'})

class AssetStatics(APIView):
    def get(self,request):
        requestData = JSONParser().parse(request)
        starting_date=requestData['starting_date']
        ending_date=requestData['ending_date']
        list=requestData['list']
        print(starting_date,ending_date,list)
        return JsonResponse({'message': 'asset statics'})
