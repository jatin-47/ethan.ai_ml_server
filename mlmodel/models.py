from http import client
from django.conf import settings
from django.contrib.auth.models import AbstractUser
from django.db import models

class User(AbstractUser):
    pass

class Client(models.Model): 
    id = models.BigAutoField(primary_key=True)
    trained_on = models.DateField(auto_now=True)
    dir_loc = models.FilePathField(path=settings.CLIENT_MODELS_DIRECTORY)
    rm = models.ForeignKey(User, on_delete=models.CASCADE, related_name="relationship_manager") #one-many relationship

    invested_amount_acc = models.FloatField(default=0)
    loan_to_invest_ratio_acc = models.FloatField(default=0)
    lowriskbucket_ratio_acc = models.FloatField(default=0)
    mediumriskbucket_ratio_acc = models.FloatField(default=0)
    highriskbucket_ratio_acc = models.FloatField(default=0)

