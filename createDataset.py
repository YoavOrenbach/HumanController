# import PIL
# from PIL import Image, ImageTk
# from tkinter import *
from cv2 import cv2
import os
import time

#useful link:
# https://www.analyticsvidhya.com/blog/2021/05/create-your-own-image-dataset-using-opencv-in-machine-learning/

camera_port = 1
if not os.path.exists("dataset"):
    os.mkdir("dataset")
key_input = input("please enter the key you want to train or type exit if you are finished: ")
while key_input != "exit":
    cv2.namedWindow("preview")
    camera = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
    if not camera.isOpened():
        print("The Camera is not Opened....Exiting")
        exit()
    if not os.path.exists("dataset/" + key_input):
        os.mkdir("dataset/" + key_input)
    TIMER = int(5)
    prev = time.time()
    while TIMER >= 0:
        ret, img = camera.read()
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(TIMER),
                    (200, 250), font,
                    7, (0, 255, 255),
                    4, cv2.LINE_AA)
        cv2.imshow("preview", img)
        cv2.waitKey(1)
        cur = time.time()
        if cur-prev >= 1:
            prev = cur
            TIMER = TIMER-1
    count = 0
    while count < 100:
        status, frame = camera.read()
        cv2.imshow("preview", frame)
        frame = cv2.resize(frame, (160, 160))
        cv2.imwrite('dataset/'+key_input+'/img'+str(count)+'.png', frame)
        cv2.waitKey(1)
        count = count + 1
    camera.release()
    cv2.destroyAllWindows()
    key_input = input("please enter the key you want to train or type exit if you are finished: ")


# show cv2 video with tkinter:
"""
width, height = 800, 600
camera_port = 1
camera = cv2.VideoCapture(camera_port, cv2.CAP_DSHOW)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = Tk()
root.bind('<Escape>', lambda e: root.quit())
lmain = Label(root)
lmain.pack()

def show_frame():
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

show_frame()
root.mainloop()
"""