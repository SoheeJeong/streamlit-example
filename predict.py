def predict(image):
    #cv2 전처리 pipeline 거치기
    clf = load(weight.file)
    return clf.predict(image)