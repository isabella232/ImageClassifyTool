# ImageClassifyTool
Object Detection Prototype Spec

Build Train and Test environment
Bellow steps all tested on ubuntu18.04.
In here recommend python anaconda environment.

-	First install python 3.6 or later.

	sudo apt install python3.6
	
	sudo apt install -y python3-pip
	
-	Install keras and tensorflow 1.15 gpu version

pip3 install keras

pip3 install tensorflow-gpu==1.15

-	Install python opencv

apt install python3-opencv
Now enter our ImageClassifyTool directory.
	cd ImageClassifyTool
 
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


1. Train Image classification


 	Data collection

First, we need image data for training. As mentioned above there are several ways for scrap image data set. Here wrote download python script for download from URL list.
Enter “dataclassify” directory.
 cd dataclassify
Execute following python script.

 python3 imagedoenloder.py --urls=urls_2.txt --dir=class02
 
This script download to class02 directory all images from URL list.
And add images which downloaded from google image search and grabbed from video.
At least 10k images are recommended for training.

 	Data Preprocessing

Once the image is ready, we have to split two set. One is train other one is validation set.
In addition, for training we have to convert images to 224X224 pixel images. 
Execute following python script.

python3 imagesplitetrain.py --infolder class02 --classname 1

After execute this script we can confirm “train” and “validation” directories.

 	Training the classifier

Now we can start training. Enter work root directory.

python3 trainclassify.py --classes=2 --size=224 --batch=64 --epochs=100 --weights=False --tclasses=0

--classes, The number of classes of dataset.

--size, The image size of train sample.

--batch, The number of train samples per batch.

--epochs, The number of train iterations.

--weights, Fine tune with other weights.

--tclasses, The number of classes of pre-trained model.

In our case –weights and –tclasses parameter is False and 0, because don’t use pre-trained model.
If completed training then we can find trained model file in  “model_training/logclassify” directory.
This file name is “trained_classifymodel.h5”. 
For test we copy this file to “model_data” directory.

 
cp model_training/logclassify/ trained_classifymodel.h5 directory model_data/trained_classifymodel.h5

 	Test and Result

Place the images you want to examine in the "testclassify" directory.

Execute following python script.

python3 image_classify.py

 	Calculate ROC and Threshold for classifier

Need to install dependent python packages

pip install scikit-learn

python3 image_calcroc.py



2. Yolo Training


 	Data collection

As same like classification we can scrap image data set. 
In this project used public data set called “VOC”. The Pascal VOC challenge is a very popular dataset for building and evaluating algorithms for image classification, object detection, and segmentation.

Enter “datadetect” directory.

 cd datadetect

Download data using following command script. 

wget https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar

wget https://pjreddie.com/media/files/VOCtrainval_06-Nov-2007.tar

wget https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar

tar xf VOCtrainval_11-May-2012.tar

tar xf VOCtrainval_06-Nov-2007.tar

tar xf VOCtest_06-Nov-2007.tar


There will now be a VOCdevkit/ subdirectory with all the VOC training data in it.
Using following command script we make training images list.

 python3 maketrainlist.py

This script filter only five (person, car, bicycle, dog, cat) objects. 
we can confirm “detecttrain.txt” file for training.
If we use image data from scrap in internet, in this case it needs to be fed with labeled training data in order for our detector to learn to detect objects in images, such as cat and dog in pictures. 
To label images, general using image annotation tool like "labelImg".
Here we don't discuss about this. 

 	Training 

Now we can start training. Enter work root directory.

python3 traindetect.py

If completed training then we can find trained model file in  “model_training/logdetect” directory.This file name is “trained_weights_final.h5”. 

For test we copy this file to “model_data” directory.
 
cp model_training/ logdetect/ trained_weights_final.h5 model_data/ yolo_trained_weights_final.h5

 	Test

Place the images you want to examine in the "testdetect" directory.

Execute following python script.

python3 image_detect.py




