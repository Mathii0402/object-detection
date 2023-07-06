import time
import os
from django.shortcuts import render
import cv2
from gtts import gTTS
def detector(classFile,configPath,weighsPath):
    

    thres = 0.6
    cap = cv2.VideoCapture(0)
    cap.set(3, 648)
    cap.set(4, 480)

    classNames=[]

    with open(classFile,'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')


    net = cv2.dnn_DetectionModel(weighsPath, configPath)   
    net.setInputSize(328,328)
    net.setInputScale(1.8/127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    # while True:
    #     cur_time = time.time()
    #     success, img = cap.read()

    #     classIds, confs, bbox = net.detect(img, confThreshold = thres)
    #     # print(classIds,bbox)        #bbox - bounding box
    #     if len(classIds) !=0:
    #         for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox): 
    #             deetcted_obj = classNames[classId]      
    #             cv2.rectangle(img,box,color=(255,0,0),thickness=2)
    #             cv2.putText(img,classNames[classId],(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
    #             cv2.putText(img,str(round(confidence*100,2)),(box[0] + 350, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0),2)
    #         if int(cur_time)%5==0:
    #             audio = gTTS(text=deetcted_obj,lang="en",slow=False)
    #             audio.save("object.mp3")
    #             os.system("object.mp3") 
    #     cv2.imshow("Output", img)
    while True:  
        cur_time = time.time()
        success, img = cap.read()  
        classIds, confs, bbox = net.detect(img, confThreshold=0.5)
        # print(classIds, bbox)
        
        if len(classIds) != 0:   
            for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                cv2.putText(img, classNames[classId].upper(), (box[0]+10, box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0), 2)
            
            if int(cur_time)%5==0:
                audio = gTTS(text=classNames[classId],lang="en",slow=False)
                audio.save("object.mp3")
                os.system("object.mp3") 
        
        
        cv2.imshow('output', img)    
        if cv2.waitKey(5) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        