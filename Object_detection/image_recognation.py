import cv2
import numpy as np


# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

img = cv2.imread("6.jpg")
#cv2.imshow("Image", img)
#cv2.waitKey(0)

img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
