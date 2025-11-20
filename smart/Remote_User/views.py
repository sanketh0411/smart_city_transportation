from django.db.models import Count
from django.db.models import Q
from django.shortcuts import render, redirect, get_object_or_404
import datetime
import openpyxl
import numpy as np

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
# Create your views here.
from Remote_User.models import ClientRegister_Model,traffic_accident_detection,detection_ratio,detection_accuracy

def login(request):


    if request.method == "POST" and 'submit1' in request.POST:

        username = request.POST.get('username')
        password = request.POST.get('password')
        try:
            enter = ClientRegister_Model.objects.get(username=username,password=password)
            request.session["userid"] = enter.id

            return redirect('ViewYourProfile')
        except:
            pass

    return render(request,'RUser/login.html')

def Add_DataSet_Details(request):

    return render(request, 'RUser/Add_DataSet_Details.html', {"excel_data": ''})


def Register1(request):
    if request.method == "POST":
        username = request.POST.get('username')
        email = request.POST.get('email')
        password = request.POST.get('password')
        phoneno = request.POST.get('phoneno')
        country = request.POST.get('country')
        state = request.POST.get('state')
        city = request.POST.get('city')
        address = request.POST.get('address')
        gender = request.POST.get('gender')
        ClientRegister_Model.objects.create(username=username, email=email, password=password, phoneno=phoneno,
                                            country=country, state=state, city=city, address=address, gender=gender)
        obj = "Registered Successfully"
        return render(request, 'RUser/Register1.html', {'object': obj})
    else:
        return render(request,'RUser/Register1.html')

def ViewYourProfile(request):
    userid = request.session['userid']
    obj = ClientRegister_Model.objects.get(id= userid)
    return render(request,'RUser/ViewYourProfile.html',{'object':obj})


def Detect_Traffic_Accident_Type(request):
    if request.method == "POST":

        if request.method == "POST":

            Fid= request.POST.get('Fid')
            Accident_Index= request.POST.get('Accident_Index')
            Longitude= request.POST.get('Longitude')
            Latitude= request.POST.get('Latitude')
            Police_Force= request.POST.get('Police_Force')
            Accident_Severity= request.POST.get('Accident_Severity')
            Number_of_Vehicles= request.POST.get('Number_of_Vehicles')
            Number_of_Casualties= request.POST.get('Number_of_Casualties')
            ADate= request.POST.get('ADate')
            Day_of_Week= request.POST.get('Day_of_Week')
            ATime= request.POST.get('ATime')
            first_Road_Class= request.POST.get('first_Road_Class')
            first_Road_Number= request.POST.get('first_Road_Number')
            Road_Type= request.POST.get('Road_Type')
            Speed_limit= request.POST.get('Speed_limit')
            Junction_Control= request.POST.get('Junction_Control')
            second_Road_Class= request.POST.get('second_Road_Class')
            second_Road_Number= request.POST.get('second_Road_Number')
            Light_Conditions= request.POST.get('Light_Conditions')
            Weather_Conditions= request.POST.get('Weather_Conditions')
            Road_Surface_Conditions= request.POST.get('Road_Surface_Conditions')



        data = pd.read_csv("Datasets.csv", encoding='latin-1')

        def apply_results(label):
            if (label == 0):
                return 0
            elif (label == 1):
                return 1

        data['Results'] = data['Label'].apply(apply_results)
        x = data['Fid'].apply(str)
        y = data['Results']


        cv = CountVectorizer()
        x = cv.fit_transform(x)


        models = []
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20)
        X_train.shape, X_test.shape, y_train.shape

        print("Random Forest Classifier")
        from sklearn.ensemble import RandomForestClassifier
        rf_clf = RandomForestClassifier()
        rf_clf.fit(X_train, y_train)
        rfpredict = rf_clf.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, rfpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, rfpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, rfpredict))
        models.append(('RandomForestClassifier', rf_clf))

        # SVM Model
        print("SVM")
        from sklearn import svm

        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, y_train)
        predict_svm = lin_clf.predict(X_test)
        svm_acc = accuracy_score(y_test, predict_svm) * 100
        print(svm_acc)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, predict_svm))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, predict_svm))
        models.append(('svm', lin_clf))

        print("Recurrent Neural Network (RNN)")

        from sklearn.neural_network import MLPClassifier
        mlpc = MLPClassifier().fit(X_train, y_train)
        y_pred = mlpc.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, y_pred) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, y_pred))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, y_pred))
        models.append(('MLPClassifier', mlpc))

        from sklearn.ensemble import GradientBoostingClassifier
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(
            X_train,
            y_train)
        clfpredict = clf.predict(X_test)
        print("ACCURACY")
        print(accuracy_score(y_test, clfpredict) * 100)
        print("CLASSIFICATION REPORT")
        print(classification_report(y_test, clfpredict))
        print("CONFUSION MATRIX")
        print(confusion_matrix(y_test, clfpredict))
        models.append(('GradientBoostingClassifier', clf))



        classifier = VotingClassifier(models)
        classifier.fit(X_train, y_train)
        y_pred = lin_clf.predict(X_test)

        Fid1 = [Fid]
        vector1 = cv.transform(Fid1).toarray()
        predict_text = classifier.predict(vector1)

        pred = str(predict_text).replace("[", "")
        pred1 = str(pred.replace("]", ""))

        prediction = int(pred1)


        if prediction == 0:
                val = 'Not Severe'
        elif prediction == 1:
                val = 'Severe'

        print(prediction)
        print(val)

        traffic_accident_detection.objects.create(
        Fid=Fid,
        Accident_Index=Accident_Index,
        Longitude=Longitude,
        Latitude=Latitude,
        Police_Force=Police_Force,
        Accident_Severity=Accident_Severity,
        Number_of_Vehicles=Number_of_Vehicles,
        Number_of_Casualties=Number_of_Casualties,
        ADate=ADate,
        Day_of_Week=Day_of_Week,
        ATime=ATime,
        first_Road_Class=first_Road_Class,
        first_Road_Number=first_Road_Number,
        Road_Type=Road_Type,
        Speed_limit=Speed_limit,
        Junction_Control=Junction_Control,
        second_Road_Class=second_Road_Class,
        second_Road_Number=second_Road_Number,
        Light_Conditions=Light_Conditions,
        Weather_Conditions=Weather_Conditions,
        Road_Surface_Conditions=Road_Surface_Conditions,
        Prediction=val)

        return render(request, 'RUser/Detect_Traffic_Accident_Type.html',{'objs': val})
    return render(request, 'RUser/Detect_Traffic_Accident_Type.html')



