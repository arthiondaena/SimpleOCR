import tkinter as tk
from tkinter import *
import PIL
from PIL import Image, ImageDraw
import PIL.ImageOps
import tensorflow as tf
from tensorflow import keras
import numpy as np
import warnings

warnings.filterwarnings('ignore')

model = keras.models.load_model('OCR_Resnet1.h5')
labelNames = "0123456789"
labelNames += "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
labelNames = [l for l in labelNames]

def predict():
    image = image1.resize((32, 32)) 
    image = np.asarray(image)[None, ...]
    image = np.array(image, dtype="float32")
    image /=255.0
    prediction = labelNames[(model.predict(image).argmax(axis=1)[0])]
    print(prediction)
    T.insert(tk.END, prediction)

def reset():
    img=ImageDraw.Draw(image1)
    img.rectangle([(0, 0), (280, 280)], fill="black")
    cv.delete("all")
    T.delete(1.0, 'end')

def activate_paint(e):
    global lastx, lasty
    cv.bind('<B1-Motion>', paint)
    lastx, lasty = e.x, e.y


def paint(e):
    global lastx, lasty
    x, y = e.x, e.y
    cv.create_line((lastx, lasty, x, y), width=10, fill='white')
    #  --- PIL
    draw.line((lastx, lasty, x, y), fill='white', width=15)
    lastx, lasty = x, y


root = Tk()

lastx, lasty = None, None
image_number = 0

cv = Canvas(root, width=280, height=280, bg='black')
# --- PIL
image1 = PIL.Image.new('L', (280, 280), 'black')
draw = ImageDraw.Draw(image1)
cv.bind('<1>', activate_paint)
cv.pack(expand=YES, fill=BOTH)

T= Text(root, height=1, width = 5, font= ('Arial', 16, 'bold'))
l = Label(root, text = "Prediction")
l.config(font=("Courier", 14))

btn_predict = Button(text="predict", command=predict)
btn_reset = Button(text="reset", command=reset)

l.pack()
T.pack()
btn_predict.pack()
btn_reset.pack()
root.mainloop()