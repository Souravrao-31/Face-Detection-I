from Detector import *

detector = Detector(use_cuda=True)

detector.processImage("input_data/harry.jpg")
#detector.processVideo(0)