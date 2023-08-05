from urllib import request
from django.shortcuts import render

# Create your views here.
from pip import main
from .Detector import detector
import os
from django.shortcuts import redirect, render

 

 
def detect_obj(request):
    # try:
        if request.method=="POST":
            # videopath=0

            configPath=os.path.join(r"C:/Users/Mathi\djangopro/ObjectDetection_FlaskDeployment-master/ObjectDetection_FlaskDeployment-master/objj\detect\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt")
            modelPath=os.path.join(r"C:/Users/Mathi\djangopro/ObjectDetection_FlaskDeployment-master/ObjectDetection_FlaskDeployment-master/objj\detect/frozen_inference_graph.pb")
            classesPath=os.path.join(r"C:/Users/Mathi\djangopro/ObjectDetection_FlaskDeployment-master/ObjectDetection_FlaskDeployment-master/objj\detect/coco.names")

            
            # detector = Detector(configPath,classesPath,modelPath)
            # detector.onVideo(request)
            detector(classesPath,configPath,modelPath)
            return render(request,"stop.html")
    # except: 

    #     return render(request,"stop.html")
        
        if request.method=="GET":
            return render(request,'stop.html')
    #     from django.http import HttpResponse
    #     return HttpResponse("stopped")
        # return redirect('/')
    # get_image(filepath, filename)


def formm(request):
    return render(request,'home.html')    
def stop(request):
    return render(request,'stop.html')    

