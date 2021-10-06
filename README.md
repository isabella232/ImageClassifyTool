# ImageClassifyTool

All steps to build the training and test environment have been tested on ubuntu18.04. Here recommend python anaconda environment.

**First install python 3.6 or later.**

```
	sudo apt install python3.6
	sudo apt install -y python3-pip
```
	
**Install keras and tensorflow 1.15 gpu version**

```
	pip3 install keras
	pip3 install tensorflow-gpu==1.15
```

**Install python opencv**

```	
	apt install python3-opencv
```
Now enter the ImageClassifyTool directory.

```
	cd ImageClassifyTool
```
 The folder structure and file function is as follows
 
```
	dataclassify: image dataset directory for train classification model 
	datadetect: image dataset directory for train yolo detect model 
	model_data: trained model directory for test and training
	model_training: working directory while train model
	testclassify: test directory for image classification
	testdetect: test directory for object detect

	trainclassify.py: training python script for classification model
	traindetect.py: training python script for yolo model
	trainclassify.py: benchmark python script for classification
	trainclassify.py: benchmark python script for detection
```

# 1. Train Image classification


**Data collection**

First, we need image data for training. As mentioned above there are several ways for scrap image data set. Here wrote download python script for download from URL list.
Enter “dataclassify” directory and run following python script.

```
	 python3 imagedownloder.py --urls=urls_1.txt --dir=class01
	 python3 imagedownloder.py --urls=urls_2.txt --dir=class02
```
 
This script download to class02 directory all images from URL list.
And add images which downloaded from google image search and grabbed from video.
At least 10k images are recommended for training.

**Data Preprocessing**

Once the image is ready, we have to split two set. One is train other one is validation set.
In addition, for training we have to convert images to 224X224 pixel images. 
Run following python script.

```
	python3 imagesplitetrain.py --infolder class01 --classname 0
	python3 imagesplitetrain.py --infolder class02 --classname 1
```

After run this script we can confirm “train” and “validation” directories.

**Training the classifier**

Now we can start training. Enter the root directory for working.

```
	python3 trainclassify.py --classes=2 --size=224 --batch=64 --epochs=100 --weights=False --tclasses=0
```

*Parameter explanation*

```
	--classes, The number of classes of dataset.
	--size, The image size of train sample.
	--batch, The number of train samples per batch.
	--epochs, The number of train iterations.
	--weights, Fine tune with other weights.
	--tclasses, The number of classes of pre-trained model.
```	

Here –weights and –tclasses parameter is False and 0, because don’t use pre-trained model.
If the training is completed then we can find trained model file in “model_training/logclassify” directory and 
the file name is “trained_classifymodel.h5”.
For test we have to copy this file to “model_data” directory.

```
	cp model_training/logclassify/trained_classifymodel.h5 model_data/trained_classifymodel.h5
```

**Fine-Tuning**

If you want to do fine-tune the trained model to improve accuracy or add other class then you can run the following command.
Before do fine-tune, you need to check the accuracy of the training data and the number of classifications and  it should be noted that the size of the input image should be consistent with the original model. You can download a pre-trained model to classify adult, soccer and other [here](https://drive.google.com/file/d/1_DLrSE2ebeexrgOEds-3Od45Nq_KLQlz/view?usp=sharing).

```
	python3 trainclassify.py --classes=3 --size=224 --batch=64 --epochs=100 --weights=trained_model.h5 --tclasses=2
```

**Test and Result**

Place the images you want to test in the "testclassify" directory and run following python script.

```
	python3 image_classify.py
```

**Calculate ROC and Threshold for classifier**

Need to install dependent python packages

```
	pip install scikit-learn
	python3 image_calcroc.py
```


# 2. Yolo Training


**Data collection**

As same like classification we can scrap image data set. 
In this project used public data set called “VOC”. The Pascal VOC challenge is a very popular dataset for building and evaluating algorithms for image classification, object detection, and segmentation.

Enter “datadetect” directory and download data using following command script.

```
	cd datadetect
	wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar
	wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar
	wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar
	tar xf VOCtrainval_11-May-2012.tar
	tar xf VOCtrainval_06-Nov-2007.tar
	tar xf VOCtest_06-Nov-2007.tar
```

There will be a VOCdevkit/ subdirectory with all the VOC training data in it.
Using following command script we can make training image list.

```
 	python3 maketrainlist.py
```

This script filter only five (person, car, bicycle, dog, cat) objects. 
we can confirm “detecttrain.txt” file for training.
If we use image data from scrap in internet, in this case it needs to be fed with labeled training data in order for our detector to learn to detect objects in images, such as cat and dog in pictures. 
To label images, general using image annotation tool like "labelImg".
Here we don't discuss about this. 

**Training**

Now we can start training. Enter work root directory and run following script.
```
	python3 traindetect.py
```

If completed training then we can find trained model file in  “model_training/logdetect” directory.This file name is “trained_weights_final.h5”. 

For test we copy this file to “model_data” directory.

```
	cp model_training/logdetect/trained_weights_final.h5 model_data/yolo_trained_weights_final.h5
```

**Test**

Place the images you want to test in the "testdetect" directory and run following python script.

```
	python3 image_detect.py
```

# 3. Trouble Shooting

**cannot import cv2**
```
	pip install opencv-python
```
**cannot import pandas**
```
	pip install pandas
```
**cannot import scikit**
```
	pip install scikit-learn
```
**cannot import matplotlib**
```
	pip install matplotlib
```


