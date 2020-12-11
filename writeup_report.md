# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/1968/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with:
- a lambda applied to each image in the batch to normalize them *(model.py Line: 79)*
- a cropping 2D layer to remove 50px from top and 20px from bottom of the image, as suggested in the lessons to remove useless and possibly misleading part of the images *(model.py Line: 81)*
- first convultional layer with 3x3 filter and depth 64 padding same, followed by a max pooling layer with size 2x2 and a droput with 10% probability. The activation it's relu *(model.py Line: 88 - 91)*
- second and third convultional layer with 5x5 filter and depth 128 padding same, followed by a max pooling layer with size 2x2 and a droput with 10% probability. The activation it's relu *(model.py Line: 96 - 104)*
- fourth and fifth convultional layer with 5x5 filter and depth 64 padding same, followed by a max pooling layer with size 2x2 and a droput with 10% probability. The activation it's relu *(model.py Line: 109 - 117)*
- the output from the convulation layers it's flattened and passed through 3 fully connected layer with size 128 each with activation relu *(model.py Line: 121 - 127)*
- output layer ( Dense(1) ) *(model.py Line: 129)*

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting *(model.py Lines: 90, 98, 103, 111, 116)*. 

The model was trained and validated on different data sets to ensure that the model was not overfitting *(model.py Lines: 134 - 143)*. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually *(model.py Line: 138)*.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. Itried to drive on all the valid space trying not to stay only on the perfect center.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a basic infrastructure - i.e. only normalization, cropping and a conlutional layer - and then build up from there.

Then I included the generator function from the lessons and modified it to return for each row from the csv the center, left and right images both normal and flipped, same for the measurements. the left and right angle are modified by a factor of -0.2 and +0.2 respectively. 

I've started adding more convolutional block - i.e. Convolution, max pooling and dropout with acrivation relu - and playing with depths and filter sizes. 

I've used the simulator many times to validate how well the car was driving around track one, adding depth or modifying filter sizes. The major improvement I've seen was given by adding the padding "same" to the convolutions. Probably having added too many layers at the end there was too least data to work with for later layers.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Creation of the Training Set & Training Process

To capture good driving behavior, I tried to drive not focusing too much on perfect center driving. I recorded to loop for each direction.

I've not used track 2 mainly because I wasn't able to complete it.

Preprocessing, normalization and cropping, has been included in the first two layer of the model.

Data has been divided in training and validation split using sklearn train_test_split with an 80-20 ratio.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. the model has been trained for 10 epochs with an adam optimizer.