import cv2
import numpy as np
from model import SN_Net, EI_Net
import torch
import torchvision.transforms as transforms
import streamlit as st

def detect_and_resize(gray):
    face_detection_option = {'haarcascade_frontalface_default.xml', 'haarcascade_frontalface_alt.xml', 'haarcascade_frontalface_alt2.xml', 'haarcascade_frontalface_alt_tree.xml'}
    face_detected = 0
    for j in face_detection_option:
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + j)
        face_detected = face_cascade.detectMultiScale(gray, 1.1, 5)
        if len(face_detected) > 0:
            break
    if len(face_detected) == 0:
        #print("No faces detected")
        pass 
    (x,y,w,h) = face_detected[0]

    eyes_detection_option = {'haarcascade_eye.xml', 'haarcascade_eye_tree_eyeglasses.xml'}
    eyes_detected = 0
    for j in eyes_detection_option:
        eyes_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + j)
        eyes_detected = eyes_cascade.detectMultiScale(gray, 1.1, 5)
        if len(eyes_detected) > 0:
            break
    if len(eyes_detected) == 0:
        #print("No eyes detected")
        pass

    angle = 0
    # calculate the angle between the eyes
    if len(eyes_detected) >= 2:
        #print("Eyes detected")
        (ex,ey,ew,eh) = eyes_detected[0]
        (ex2,ey2,ew2,eh2) = eyes_detected[1]
        if ex > ex2:
            temp = ex
            ex = ex2
            ex2 = temp
            temp = ey
            ey = ey2
            ey2 = temp
            temp = ew
            ew = ew2
            ew2 = temp
            temp = eh
            eh = eh2
            eh2 = temp
        ey = ey + eh/2
        ey2 = ey2 + eh2/2
        ex = ex + ew/2
        ex2 = ex2 + ew2/2
        dy = (ey2 - ey)
        dx = (ex2 - ex)
        angle = np.arctan(dy/dx)
        angle = (angle * 180) / np.pi
        #print(angle)
    else:
        #print("Eyes not detected") 
        pass
    # face straightening
    rows, cols = gray.shape[:2]
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    img_rotated = cv2.warpAffine(gray, M, (cols, rows))
    # crop the face
    img_cropped = img_rotated[y:y+h, x:x+w]
    # resize the image
    width = 64
    height = 64
    img_resized = cv2.resize(img_cropped, (width, height), interpolation = cv2.INTER_LINEAR)
    
    return img_resized

def preprocess(image):
    # cv2img = cv2.imshow('temp',image)
    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    resized_img = detect_and_resize(gray_img)
    return resized_img

def pred_and_normalize(net,model_pth,image,mean,var):
    model = net((image.shape[1], image.shape[2]))
    model.load_state_dict(torch.load(model_pth))
    model.eval()
    with torch.no_grad():
        y_pred = model(image.unsqueeze(0))
    pred_value = (y_pred - mean) / np.sqrt(var)
    return pred_value

def predict(image):
    #cv2 전처리 pipeline 거치기
    image2 = preprocess(image)
    transform = transforms.ToTensor()
    image_tensor = transform(image2)
    #i or e
    train_pred_mean = 0.22778183559017
    train_pred_var = 1.0022041672585026e-10
    pred_value = pred_and_normalize(EI_Net,'results/cnn_ei.pth',image_tensor,train_pred_mean,train_pred_var)
    threshold = 0.5481400437636762
    first = 'e' if pred_value>=threshold else 'i'
    #s or n
    train_pred_mean = 0.0003726313971853155
    train_pred_var = 4.695636660473933e-16
    pred_value = pred_and_normalize(SN_Net,'results/cnn_sn.pth',image_tensor,train_pred_mean,train_pred_var)
    threshold = 0.5303776683087028
    second = 's' if pred_value>=threshold else 'n'
    #f or t
    third = 'f'
    #j or p
    forth = 'j'
    #image2,ypred for test
    return first+second+third+forth, image2, pred_value