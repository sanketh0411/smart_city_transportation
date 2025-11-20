from django.db import models

# Create your models here.
from django.db.models import CASCADE


class ClientRegister_Model(models.Model):
    username = models.CharField(max_length=30)
    email = models.EmailField(max_length=30)
    password = models.CharField(max_length=10)
    phoneno = models.CharField(max_length=10)
    country = models.CharField(max_length=30)
    state = models.CharField(max_length=30)
    city = models.CharField(max_length=30)
    address= models.CharField(max_length=300)
    gender= models.CharField(max_length=30)


class traffic_accident_detection(models.Model):

    Fid= models.CharField(max_length=3000)
    Accident_Index= models.CharField(max_length=3000)
    Longitude= models.CharField(max_length=3000)
    Latitude= models.CharField(max_length=3000)
    Police_Force= models.CharField(max_length=3000)
    Accident_Severity= models.CharField(max_length=3000)
    Number_of_Vehicles= models.CharField(max_length=3000)
    Number_of_Casualties= models.CharField(max_length=3000)
    ADate= models.CharField(max_length=3000)
    Day_of_Week= models.CharField(max_length=3000)
    ATime= models.CharField(max_length=3000)
    first_Road_Class= models.CharField(max_length=3000)
    first_Road_Number= models.CharField(max_length=3000)
    Road_Type= models.CharField(max_length=3000)
    Speed_limit= models.CharField(max_length=3000)
    Junction_Control= models.CharField(max_length=3000)
    second_Road_Class= models.CharField(max_length=3000)
    second_Road_Number= models.CharField(max_length=3000)
    Light_Conditions= models.CharField(max_length=3000)
    Weather_Conditions= models.CharField(max_length=3000)
    Road_Surface_Conditions= models.CharField(max_length=3000)
    Prediction= models.CharField(max_length=3000)

class detection_accuracy(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)

class detection_ratio(models.Model):

    names = models.CharField(max_length=300)
    ratio = models.CharField(max_length=300)



