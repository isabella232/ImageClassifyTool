import urllib.request
import os
import time
import sys
import errno
import calendar
import argparse
import json
from shutil import copyfile
import cv2

import glob

def progress(count, total, status=''):
    bar_len = 60
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    sys.stdout.write('[%s] %s%s    %s\r' % (bar, percents, '%', status))
    sys.stdout.flush()

#python imagesplitetrain.py --infolder class01 --classname 0
#python imagesplitetrain.py --infolder class02 --classname 1
def main():
    parser = argparse.ArgumentParser(description='image splite to train and validate for training ')
    parser.add_argument('--infolder', type=str, help='image folder path ')
    parser.add_argument('--classname', type=str, help='splite class name')

    args = parser.parse_args()

    infolder = args.infolder
    classname = args.classname


    traindir = "train/" + classname + "/"
    validationdir = "validation/" + classname + "/"

    try:
        os.makedirs(traindir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    try:
        os.makedirs(validationdir)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

    fileset = [file for file in glob.glob(infolder + "**/*.jpg", recursive=True)]

    count = len(fileset)
    i = 0
    for file in fileset:
        print(file)
        fname = os.path.basename(file)
        if i % 10 == 0: #test
            destination_file = validationdir + fname
        else:#train
            destination_file = traindir + fname

        try:
            srcimg = cv2.imread(file)
            dstimg = cv2.resize(srcimg, (224, 224), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(destination_file, dstimg)
            #copyfile(file, destination_file)

            i = i + 1

        except:
            pass

        progress(i, 10000, status='processing')

        if i == 10000:
            break

    print("\nTask Completed")


if __name__== "__main__":
    main()