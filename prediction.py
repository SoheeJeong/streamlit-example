import cv2
import numpy as np
from model import SN_Net

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
    gray_img = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    resized_img = detect_and_resize(gray_img)
    return resized_img

def predict(image):
    #cv2 전처리 pipeline 거치기
    image2 = preprocess(image)
    clf = load(weight.file)
    #i or e
    first = 'i'
    #s or n
    sn_model = SN_Net((image2.shape[1], image2.shape[2]))
    sn_model.load_state_dict(torch.load('sn_model.pth'))
    sn_model.eval()
    with torch.no_grad():
        y_pred = sn_model(image2.unsqueeze(0))
    second = 's' if ypred==1 else 'n'
    #f or t
    third = 'f'
    #j or p
    forth = 'j'
    return first+second+third+forth