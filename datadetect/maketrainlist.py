import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join
import glob

sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]

#classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

classes = ["person", "car", "bicycle", "dog", "cat"]

#workdir  = getcwd()
workdir  = "datadetect"
trainfname = "detecttrain.txt"
testfname = "detecttest.txt"


list_trainfile = open(trainfname, 'w')
#list_testfile = open(testfname, 'w')

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text), float(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')

def getobjectpos(year, image_id):
    in_file = open('VOCdevkit/VOC%s/Annotations/%s.xml'%(year, image_id))
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)
    strpos = ""

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')

        strtmp = xmlbox.find('xmin').text + "," + xmlbox.find('ymin').text + "," + \
                 xmlbox.find('xmax').text + "," + xmlbox.find('ymax').text + "," + str(cls_id)

        strpos = strpos +  " " +  strtmp

    return strpos

def test():
    testdir = "data"
    width = 1920
    height = 1080

    list_trainfile = open(trainfname, 'w')

    fileset = [file for file in glob.glob(testdir + "**/*.jpg", recursive=True)]
    count = len(fileset)
    binit = False

    for file in fileset:
        fname = os.path.basename(file)
        print(fname + ": ")

        title = fname.split(".")[0]
        posdata = testdir + "/" + title + ".txt"

        fpos = open(posdata, 'r')
        sline = fpos.readline()
        fpos.close()

        b = (
        float(sline.split(" ")[1]), float(sline.split(" ")[2]), float(sline.split(" ")[3]), float(sline.split(" ")[4]))
        bb = convert((width, height), b)
        list_trainfile.write(file + " ".join([str(a) for a in bb]) + "0 \n")

        # start_time = time.time()
    list_trainfile.close()

def main():
    for year, image_set in sets:
        if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
            os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
        image_ids = open('VOCdevkit/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()

        btrain = True
        #if year == '2007' and  image_set == 'test': #make test dataset
        #    btrain = False

        for image_id in image_ids:
            fpath = '%s/VOCdevkit/VOC%s/JPEGImages/%s.jpg'%(workdir, year, image_id)
            objpos = getobjectpos(year, image_id)
            if len(objpos) > 0 and btrain == True:
                list_trainfile.write(fpath+objpos+"\n")
            #elif len(objpos) > 0 and btrain == False:
            #    list_testfile.write(fpath + objpos + "\n")

    list_trainfile.close()
    #list_testfile.close()

if __name__== "__main__":
    main()