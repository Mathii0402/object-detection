
from cgitb import text
from pickle import TRUE
import cv2 as cv
import numpy as np
import time
from gtts import gTTS
import os
# from django.http import HttpResponse
from django.shortcuts import render
class Detector:
    def __init__(self,videopath,configPath,modelPath,classesPath):
        self.videopath=videopath
        self.configPath=configPath
        self.modelPath=modelPath
        self.classesPath=classesPath
        


        self.net=cv.dnn_DetectionModel(self.modelPath,self.configPath)
        self.net.setInputSize(320,320)
        self.net.setInputScale(1.0/127.5)
        self.net.setInputMean((127.5,127.5,127.5))
        self.net.setInputSwapRB(True)

        self.readClasses()

    def readClasses(self):
        with open(self.classesPath,'r') as f:
            self.classesList=f.read().splitlines()

        
        self.colorList=np.random.uniform(low=0,high=255,size=(len(self.classesList),3) )
        print(self.classesList)

   

    def onVideo(self,request):
        language="en"
        
        cap=cv.VideoCapture(self.videopath, cv.CAP_DSHOW)
    
      
        if(cap.isOpened()==False):
            print('ERROR IN OPENING VIDEO')
            return render(request,"stop.html")
        
        (success,image)=cap.read()
        
        startTime=0
        while success:
            currentTime=time.time()
            fps=1/(currentTime-startTime)
            startTime=currentTime 
            classLabelIDs,confidences,bbox= self.net.detect(image,confThreshold=0.5)

            bboxs=list(bbox)
            confidences=list(np.array(confidences).reshape(1,-1)[0])
            confidences=list(map(float,confidences))
        

            bboxIdx=cv.dnn.NMSBoxes(bboxs,confidences,score_threshold=0.5,nms_threshold=0.2)
          
            
                
            try:
                if len(bboxIdx!=0):
                   
                    for i in range(0,len(bboxIdx)):
                        bbox=bboxs[np.squeeze(bboxIdx[i])]
                        classConfidence=confidences[np.squeeze(bboxIdx[i])]
                        classLabelID=np.squeeze(classLabelIDs[np.squeeze(bboxIdx[i])])
                        classLabel=self.classesList[classLabelID]
                        classColor=[int(c) for c in self.colorList[classLabelID]]


                        displayText="{}:{:.2f}".format(classLabel,classConfidence)
                        speech=classLabel
                        x,y,w,h=bbox

                        cv.rectangle(image,(x,y),(x+w,y+h),color=classColor,thickness=1)
                        cv.putText(image,speech,(x,y-10),cv.FONT_HERSHEY_PLAIN,1,classColor,2)
                    ###################################################
                        linewidth=min(int(w*0.3),int(h*0.3))

                        cv.line(image,(x,y),(x+linewidth,y),classColor,thickness=5)
                        cv.line(image,(x,y),(x,y+linewidth),classColor,thickness=5)


                        cv.line(image,(x+w,y),(x+w-linewidth,y),classColor,thickness=5)
                        cv.line(image,(x+w,y),(x+w,y+linewidth),classColor,thickness=5)

    #####################################################

                        cv.line(image,(x,y+h),(x+linewidth,y+h),classColor,thickness=5)
                        cv.line(image,(x,y+h),(x,y+h-linewidth),classColor,thickness=5)


                        cv.line(image,(x+w,y+h),(x+w-linewidth,y+h),classColor,thickness=5)
                        cv.line(image,(x+w,y+h),(x+w,y+h-linewidth),classColor,thickness=5)
    
                        print(currentTime)
                else:
                    print("no obj")
                    return render(request,"stop.html")
                    # cap.release()
            except:
                return render(request,"stop.html")
                
                
            ct=0
            if(int(currentTime)%5==0):
                ct=ct+1
                myobj=gTTS(text=speech,lang=language, slow=False)
                myobj.save("saved.mp3")
                os.system("saved.mp3")

        
            cv.putText(image,"FPS:  "+str(int(fps)),(20,70),cv.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
            cv.imshow("Result ",image)
            # return render(request,"stop.html")
            key=cv.waitKey(5)& 0xFF
            if key == ord("q"):
                break
            (success,image)= cap.read()
        cv.destroyAllWindows(0)
        

        return render(request,"home.html")
        