import colorsys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import glob
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#import tensorflow as tf
#if type(tf.contrib) != type(tf): tf.contrib._warning = None

import logging
import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR) # disable Tensorflow warnings for this tutorial
import warnings
warnings.simplefilter("ignore") # disable Keras warnings for this tutorial

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=config)

import numpy as np
from keras import backend as K

from keras.models import load_model
from keras.layers import Input

from yolo3.model import yolo_eval, yolo_body, tiny_yolo_body
from yolo3.utils import image_preporcess

import os
import time
import psutil
import platform
import subprocess
import argparse

import xml.etree.ElementTree
from collections import OrderedDict
from distutils import spawn
#from time import sleep

#from tensorflow.python.util import deprecation
#deprecation._PRINT_DEPRECATION_WARNINGS = False

class YOLO(object):
    _defaults = {
        "model_path": 'model_data/yolo_weights.h5',
        "anchors_path": 'model_data/yolo_anchors.txt',
        "classes_path": 'model_data/yolo_classes.txt',
        #"model_path": 'model_data/yolo_trained_weights_final.h5',
        #"anchors_path": 'model_data/yolo_anchors.txt',
        #"classes_path": 'model_data/yolo_trained_class.name',
        "score" : 0.3,#0.3
        "iou" : 0.45,#0.45
        "model_image_size" : (416, 416),
        "text_size" : 3,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults) # set up default values
        self.__dict__.update(kwargs) # and update with user overrides
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = K.get_session()
        #self.sess = K.compat.v1.Session()

        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def generate(self):
        print("Loading model...\n")

        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        # Load model, or construct model and load weights.
        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)
        is_tiny_version = num_anchors==6 # default setting
        try:
            self.yolo_model = load_model(model_path, compile=False)
        except:
            self.yolo_model = tiny_yolo_body(Input(shape=(None,None,3)), num_anchors//2, num_classes) \
                if is_tiny_version else yolo_body(Input(shape=(None,None,3)), num_anchors//3, num_classes)
            self.yolo_model.load_weights(self.model_path) # make sure model, anchors and classes match
        else:
            assert self.yolo_model.layers[-1].output_shape[-1] == \
                num_anchors/len(self.yolo_model.output) * (num_classes + 5), \
                'Mismatch between model and given anchor and class sizes'

        #print('{} model, anchors, and classes loaded.'.format(model_path))


        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        if self.model_image_size != (None, None):
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = image_preporcess(np.copy(image), tuple(reversed(self.model_image_size)))
            image_data = boxed_image

        classes = ["person", "car", "bicycle", "dog", "cat"]

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.shape[0], image.shape[1]],#[image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        #print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        thickness = (image.shape[0] + image.shape[1]) // 600
        fontScale=1
        ObjectsList = []
        
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            if predicted_class not in classes:
                continue

            label = '{} {:.2f}'.format(predicted_class, score)
            #label = '{}'.format(predicted_class)
            scores = '{:.2f}'.format(score)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))

            mid_h = (bottom-top)/2+top
            mid_v = (right-left)/2+left

            # put object rectangle
            cv2.rectangle(image, (left, top), (right, bottom), self.colors[c], thickness)

            # get text size
            (test_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, 1)

            # put text rectangle
            cv2.rectangle(image, (left, top), (left + test_width, top - text_height - baseline), self.colors[c], thickness=cv2.FILLED)

            # put text above rectangle
            cv2.putText(image, label, (left, top-2), cv2.FONT_HERSHEY_SIMPLEX, thickness/self.text_size, (0, 0, 0), 1)

            # add everything to list
            ObjectsList.append([top, left, bottom, right, mid_v, mid_h, label, scores])

        return image, ObjectsList

    def close_session(self):
        self.sess.close()

    def detect_img(self, image):
        image = cv2.imread(image, cv2.IMREAD_COLOR)
        original_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_image_color = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        r_image, ObjectsList = self.detect_image(original_image_color)
        return r_image, ObjectsList


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

def detectwithUI(yolo):

    for i in range(1, 6):
        image = "detecttest" + str(i) + ".jpg"
        r_image, ObjectsList = yolo.detect_img(image)
        print(ObjectsList)
        #cv2.imshow(image, r_image)

        dsize = (600, 600)
        # resize image
        output = cv2.resize(r_image, dsize)
        cv2.imshow(image, output)

        #cv2.resizeWindow(image, 600, 600)
        cv2.waitKey(0)

        input("Press Enter to continue...")

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    # Required arguments.
    #parser.add_argument("--dir",default="testdetect",help="The number of classes of dataset.")
    args = parser.parse_args()
    #testdir = args.dir
    testdir = "testdetect"

    yolo = YOLO()

    totaltime = 0
    totalgusage = 0.0
    totalgram = 0.0
    totalgper = 0.0
    totalcusage = 0.0
    totalcram = 0.0
    totalcper = 0.0

    fileset = [file for file in glob.glob(testdir + "**/*.jpg", recursive=True)]
    count = len(fileset) - 1
    binit = False

    for file in fileset:

        start_time = time.time()
        fname = os.path.basename(file)
        print(fname + ": ")
        r_image, ObjectsList = yolo.detect_img(file)
        print(ObjectsList)

        if binit == False:
            binit = True
            continue

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

    yolo.close_session()
