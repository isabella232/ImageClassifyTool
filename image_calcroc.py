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

import matplotlib.pyplot as pp
import sklearn.metrics as metrics
from sklearn.metrics import accuracy_score,precision_score, recall_score

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

nload = 1
print("Loading model...\n")

if nload > 0:
    model = MobileNetv2((224, 224, 3), 2)
    model.load_weights('model_data/trained_classifymodel.h5')


#model.load_weights('model_training/logclassify/trained_weights_final.h5')
#model.load_weights('model_data/trained_classify_final.h5',by_name=True)
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

sets=[('dataclassify/validation/0', '1'), ('dataclassify/validation/1', '0')]
label = []
values = []

def calcpredict():
    for workdir, labelval in sets:

        fileset = [file for file in glob.glob(workdir + "**/*.jpg", recursive=True)]

        for file in fileset:
            fname = os.path.basename(file)
            print(fname + ": ")
            start_time = time.time()
            frame = cv2.imread(file)

            img = cv2.resize(frame, (224, 224), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
            # img = cv2.resize(frame, (224, 224), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)
            # img = cv2.resize(frame, (224, 224))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255
            expanded_img = np.expand_dims(img, axis=0)

            # start_time = time.time()

            pred = model.predict(expanded_img)

            label.append(int(labelval))
            values.append(pred[0][0])

    alable = np.array(label)
    ascores = np.array(values)

    np.save('label.npy', alable)
    np.save('scores.npy', ascores)

def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_hat)):
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1

    return(TP, FP, TN, FN)

def main():
    parser = argparse.ArgumentParser()
    # Required arguments.
    # parser.add_argument("--dir",default="testclassify",help="The number of classes of dataset.")
    args = parser.parse_args()
    # testdir = args.dir
    testdirtrue = "dataclassify/validation/0"
    testdirfalse = "dataclassify/validation/1"

    # prepare positive and negative array
    calcpredict()

    alable = np.load('label.npy')
    ascores = np.load('scores.npy')

    afpr, atpr, thresholds = metrics.roc_curve(alable, ascores)

    arocauc = metrics.auc(afpr, atpr)

    pp.title("Receiver Operating Characteristic")
    pp.xlabel("False Positive Rate(1 - Specificity)")
    pp.ylabel("True Positive Rate(Sensitivity)")
    pp.plot(afpr, atpr, "b", label="(AUC = %0.2f)" % arocauc)
    pp.plot([0, 1], [1, 1], "y--")
    pp.plot([0, 1], [0, 1], "r--")
    pp.legend(loc="lower right")

    #ax2 = pp.gca().twinx()
    #ax2.plot(afpr, thresholds, markeredgecolor='r', linestyle='dashed', color='r')
    #ax2.set_ylabel('Threshold', color='r')
    #ax2.set_ylim([thresholds[-1], thresholds[0]])
    #ax2.set_xlim([afpr[0], afpr[-1]])

    pp.show()

    pp.figure()
    pp.plot(1.0 - atpr, thresholds, marker='*', label='tpr')
    pp.plot(afpr, thresholds, marker='o', label='fpr')
    pp.legend()
    pp.xlim([0, 1])
    pp.ylim([0, 1])
    pp.xlabel('thresh')
    pp.ylabel('far/fpr')
    pp.title(' thresh - far/fpr')
    pp.show()

    for threval in np.arange(0.5, 1.0, 0.05):
        predictval = (ascores > threval)

        TP, FP, TN, FN = perf_measure(alable, predictval)

        FACC = accuracy_score(alable, predictval)
        FFAR = FP / float(FP + TN)
        FFRR = FN / float(TP + FN)

        #ftar =  precision_score(alable, predictval)
        #ffar = recall_score(alable, predictval)

        print("Threshold: %.2f Accuracy : %.2f  FAR: %.2f FRR: %.2f" %(threval ,FACC, FFAR,FFRR))

    print("task complete!")

if __name__== "__main__":
    main()
