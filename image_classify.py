import os
import sys
import warnings
import os
import time
import psutil
import platform
import subprocess

import argparse
import cv2
import pickle
import numpy as np
import xml.etree.ElementTree
from collections import OrderedDict
from distutils import spawn
import glob
import logging

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from mobilenet_v2 import MobileNetv2
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR) # disable Tensorflow warnings for this tutorial
import warnings
warnings.simplefilter("ignore") # disable Keras warnings for this tutorial

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)




TEXT_LOC = (30,30)
FONT_SIZE = 0.8

THESHOLD = 0.2


#model = load_model('model_data/mobilenetv2_class20.h5')
#with open('model_data/class20.pickle','rb') as f:
#    class20 = pickle.load(f)

print("Loading model...\n")

model = MobileNetv2((224, 224, 3), 2)
#model.load_weights('model_data/trained_classify_final.h5',by_name=True)
model.load_weights('model_data/trained_classifymodel.h5')
#model = load_model('model_data/trained_classifymodel.h5')
class20 = ["bad","football","other"]


def extract(elem, tag, drop_s):
    text = elem.find(tag).text
    if drop_s not in text: raise Exception(text)
    text = text.replace(drop_s, "")
    try:
        return int(text)
    except ValueError:
        return float(text)


def gpustatus():
    if platform.system() == "Windows":
        # If the platform is Windows and nvidia-smi
        # could not be found from the environment path,
        # try to find it from system drive with default installation path
        nvidia_smi = spawn.find_executable('nvidia-smi')
        if nvidia_smi is None:
            nvidia_smi = "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe" % os.environ['systemdrive']
    else:
        nvidia_smi = "nvidia-smi"

    d = OrderedDict()
    d["time"] = time.time()

    cmd_out = subprocess.check_output([nvidia_smi, "-q", "-x"])

    gpu = xml.etree.ElementTree.fromstring(cmd_out).find("gpu")
    util = gpu.find("utilization")
    d["util"] = extract(util, "gpu_util", "%")

    d["mem_used"] = extract(gpu.find("fb_memory_usage"), "used", "MiB")
    totalmem = extract(gpu.find("fb_memory_usage"), "total", "MiB")
    d["mem_used_per"] = d["mem_used"] * 100 / totalmem

    return d


def cpustatus():
    d = OrderedDict()
    d["time"] = time.time()
    util = psutil.cpu_percent()
    d["util"] = util
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2. ** 30 * 1024  # MiB
    d["mem_used"] = memoryUse
    d["mem_used_per"] = py.memory_percent()
    # mem = psutil.virtual_memory()
    # d["mem_used_per"] =  memoryUse / mem.total * 100.0
    # from psutil import virtual_memory
    # mem = virtual_memory()
    # mem.total  # total physical memory available
    return d

def classifyvideo():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH,640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT,480)

    (major_ver,minor_ver,subminor_ver) = cv2.__version__.split('.')

    while True:
        ret, frame = capture.read()

        if int(major_ver) < 3:
            fps = capture.get(cv2.cv.CV_CAP_PROP_FPS)
        else:
            fps = capture.get(cv2.CAP_PROP_FPS)



        img = cv2.resize(frame,(224,224),fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        img = img / 255

        expanded_img = np.expand_dims(img,axis=0)

        pred = model.predict(expanded_img)

        _cls = class20[np.argmax(pred)]
        _prob = np.max(pred)

        font_color = (0,0,0)
        label = 'Prediction : ...'

        if _prob > THESHOLD:

            font_color = (0,0,255)
            label = 'Prediction : {} ({:.2f}%)'.format(_cls.upper(),_prob * 100)

        cv2.putText(frame,label,TEXT_LOC,cv2.FONT_HERSHEY_SIMPLEX,FONT_SIZE,font_color,2)
        cv2.putText(frame,'{} FRAMES'.format(fps),(450,30),cv2.FONT_HERSHEY_SIMPLEX,FONT_SIZE,(0,0,0),2)

        cv2.imshow('test',frame)

        if cv2.waitKey(1) == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

def classifyimage(path):

    frame = cv2.read(path)

    img = cv2.resize(frame, (224, 224), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255

    expanded_img = np.expand_dims(img, axis=0)

    pred = model.predict(expanded_img)

    _cls = class20[np.argmax(pred)]
    _prob = np.max(pred)

    font_color = (0, 0, 0)
    label = 'Prediction : ...'

    if _prob > THESHOLD:
        font_color = (0, 0, 255)
        label = 'Prediction : {} ({:.2f}%)'.format(_cls.upper(), _prob * 100)

    cv2.putText(frame, label, TEXT_LOC, cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, font_color, 2)
    #cv2.putText(frame, '{} FRAMES'.format(fps), (450, 30), cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, (0, 0, 0), 2)

    cv2.imshow('test', frame)
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser()
    # Required arguments.
    # parser.add_argument("--dir",default="testclassify",help="The number of classes of dataset.")
    args = parser.parse_args()
    # testdir = args.dir
    testdir = "testclassify"

    totaltime = 0
    totalgusage = 0.0
    totalgram = 0.0
    totalgper = 0.0
    totalcusage = 0.0
    totalcram = 0.0
    totalcper = 0.0

    fileset = [file for file in glob.glob(testdir + "**/*.jpg", recursive=True)]
    count = len(fileset)

    for file in fileset:

        fname = os.path.basename(file)
        print(fname + ": ")

        frame = cv2.imread(file)

        img = cv2.resize(frame, (224, 224), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        # img = cv2.resize(frame, (224, 224), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
        # img = cv2.resize(frame, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img / 255
        expanded_img = np.expand_dims(img, axis=0)

        start_time = time.time()

        pred = model.predict(expanded_img)
        _cls = class20[np.argmax(pred)]
        _prob = np.max(pred)

        font_color = (0, 0, 0)
        label = 'Prediction : ...'

        if _prob > THESHOLD:
            font_color = (0, 0, 255)
            label = 'Prediction : {} ({:.2f}%)'.format(_cls.upper(), _prob * 100)
        else:
            label = 'Prediction : {} ({:.2f}%)'.format("Unknown", 0)

        #cv2.putText(frame, label, TEXT_LOC, cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, font_color, 2)

        print(label)

        totaltime += (time.time() - start_time)
        gds = gpustatus()
        cds = cpustatus()
        totalgusage += gds["util"]
        totalgram += gds["mem_used"]
        totalgper += gds["mem_used_per"]

        totalcusage += cds["util"]
        totalcram += cds["mem_used"]
        totalcper += cds["mem_used_per"]

    fps = 1.0 / (totaltime / count)

    totalgusage /= count
    totalgram /= count
    totalgper /= count

    totalcusage /= count
    totalcram /= count
    totalcper /= count

    print("\n===================================\n")
    laststr = "FPS:{0:.2f} GPU:{1:.2f}% CPU:{2:.2f}% GPU RAM:{3:.2f} MiB {4:.2f}% CPU RAM:{5:.2f} MiB {6:.2f}%".format(
        fps, totalgusage, totalcusage, totalgram, totalgper, totalcram, totalcper)
    print(laststr)
    print("\n===================================\n")



if __name__== "__main__":
    main()
