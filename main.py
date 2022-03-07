import cv2
from predictor import predict_asl

cap = cv2.VideoCapture(1)

if __name__ == '__main__':
    predict_asl(cap)


