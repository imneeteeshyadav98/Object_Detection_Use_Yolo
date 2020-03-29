import cv2
import numpy as np
from darkflow.net.build import TFNet
import time
import tensorflow as tf
config=tf.ConfigProto(log_device_placement=False)
config.gpu_option.allow_growth=False
with tf.Session(config=config) as sess:
    option={
            "models":'./cfg/yolo.cfg',
            "load":'./yolov3.weights'}
    tfnet=TFNet(option)
capture=cv2.VideoCapture(0)
while True:
    ret,frame=capture.read()
    if ret:
        results=tfnet.return_predict(frame)
        image=displayResults(results,frame)
        cv2.imshow("Video",image)
        if cv2.waitKey(1)==13:
            break
capture.release()
cv2.destroyAllWindows()