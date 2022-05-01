import numpy as np
import streamlit as st
import cv2
from PIL import Image
import torch
import os

RESIZE_W = 640
RESIZE_H = 640

def draw(image:str, labelVals:list, thresh:float):
    img = Image.open(image)
    img = np.array(img)
    img = cv2.resize(img, (RESIZE_W, RESIZE_H))


    for i in labelVals:
        # checking confidence threshold
        if i[-1] < thresh:
            continue
        cell = i[0]
        xc = float(i[1]) * RESIZE_W
        yc = float(i[2]) * RESIZE_H
        w = float(i[3]) * RESIZE_W
        h = float(i[4]) * RESIZE_H
    
        img = cv2.rectangle(img, (int(xc - w/2), int(yc - h/2)), (int(xc + w/2), int(yc + h/2)), (0,0,255), 1)
        img = cv2.putText(img, f"{cell} {round(i[-1], 2)}", (int(xc - w/2), int(yc - h/2 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 1)
        
    return img

@st.cache
def loadModel():
    model = torch.hub.load("yolov5", "custom", path="data/weights/best.pt", source="local")
    print("loaded model")
    return model

model = loadModel()

file = st.sidebar.file_uploader('Upload an Image', type=['png', 'jpg', 'jpeg'], accept_multiple_files=False)

imgs = os.listdir("data/images")

imgs = sorted(imgs, key=lambda x: int(x.split("-")[-1].split(".")[0]))
imgs = [i.split(".")[0] for i in imgs]

if not file:
    file = st.sidebar.selectbox("Select an Image", imgs)
    file = f"data/images/{file}.png"

threshold = st.sidebar.slider("Confidence lower bound", min_value=0.01, max_value=1.0, value=0.7)


originalImage = Image.open(file)
fileDimensions = originalImage.size
W = fileDimensions[0]
H = fileDimensions[1]
originalImage = np.array(originalImage)



res = model(originalImage)

# results processing
values = []
for i in res.pandas().xyxy[0].values:
    temp = [i[-1], ((i[0] + i[2])/2) / W, ((i[1] + i[3])/2) / H, (i[2]-i[0]) / W, (i[3] - i[1]) / H, i[-3]]
    values.append(temp)

# display image
outImage = draw(file, values, threshold)

st.write("Output")
st.image(outImage, use_column_width=True)


