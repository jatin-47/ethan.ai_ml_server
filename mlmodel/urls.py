from django.urls import path
from mlmodel import views
from django.contrib.auth.decorators import login_required

urlpatterns = [
    path('ping/', views.Ping.as_view()),
    path('efficient/frontier/', views.EfficientFrontier.as_view()),
    path('recommendation/', views.Recommendation.as_view()),
    path('backtests/', views.Backtests.as_view()),
    path('asset/statics/', views.AssetStatics.as_view()),
    ##########################################################################
    path('login', views.Login.as_view(), name='login'),
    path('logout', views.Logout.as_view(), name='logout'),
    path('register', views.Register.as_view(), name='register'),
    path('dashboard', login_required(views.Dashboard.as_view()), name='dashboard'),
    path('riskprediction/upload', login_required(views.UploadData.as_view(), login_url='login'), name='upload'),
    path('riskprediction/train', login_required(views.TrainModel.as_view(), login_url='login'), name='model_train'),
    path('riskprediction/predict', login_required(views.PredictModel.as_view(), login_url='login'), name='model_predict'),

    path('riskprediction/plot', views.RiskPredictionPlot.as_view())
]