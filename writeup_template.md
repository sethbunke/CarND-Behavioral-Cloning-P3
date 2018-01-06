# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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

[valid1]: ./images/valid1.png 
[valid2]: ./images/valid2.png 
[valid3]: ./images/valid3.png 
[valid4]: ./images/valid4.png 
[valid5]: ./images/valid5.png 
[valid6]: ./images/valid6.png 

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

See "Final Model Architecture" section below. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 31). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 40-45). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Additionally, multiple datasets and models were used and tested to determine their accuacy and whether or not they were overfitting on the training set by visually examining the mean squared error. 

 ![alt text][valid1]
 ![alt text][valid2]
 ![alt text][valid3]
 ![alt text][valid4]
 ![alt text][valid5]
 ![alt text][valid6]

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 118).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and driving in both directions on the first track. 

Multiple data sets were generated and tested for their accuracy. Generating the data was one of the most difficult parts of this project as the keyboard controls would tend to "lag" precipitating my over-compensating in the driving and when the "lag" would catch up the vehicle to go off the road. 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The model architecture is based largely on that of the NVIDIA design while also incorporating the cropping of the image in the model to reduce irrlevant data being being trained up on by the model and adding a dropout layer to reduce overfitting. 

See images above for details on the results of the different data model/data set/parameter combinations. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

1. Input image size was 160x320x3
2. Lambda layer for normalising the image to be floats centered around 0.
3. A cropping layer to remove the top 70 pixels from the top and 25 pixels from the hood.
4. Convolution layer with kernel size of 5x5 and stride 2x2 and depth 24 with a relu activation
5. Convolution layer with kernel size of 5x5 and stride 2x2 and depth 36 with a relu activation
6. Convolution layer with kernel size of 5x5 and stride 2x2 and depth 48 with a relu activation
7. Convolution layer with kernel size of 3x3 and stride 1x1 and depth 64 with a relu activation
8. Convolution layer with kernel size of 3x3 and stride 1x1 and depth 64 with a relu activation
9. 20% dropout
10. Fully connected layer with 100 nodes
11. Fully connected layer with 50 nodes
12. Fully connected layer with 10 nodes
13. Fully connected layer with 1 node which predicts the steering angle.

#### 3. Creation of the Training Set & Training Process

I recorded approximately 2.5 laps on track one using center lane driving, then another lap going in the opposite direction, and added recover maneuvers. Additionally, for performance/memory managment reasons my code uses a generator (lines 85 - 106) which provides to the model left, center, and right images withe appropriate steering angle as well as a "flipped" versions of all of these images (lines 55 - 80). 

All samples read from the CSV file are shuffled before batches are created and the images are retrieved and the steering angles are calculated. 

Validation sets are established via the use of train_test_split (line 83) with 20% going to the validation set. These sets are passed to the generator mentioned above. 

The number of epochs and batch sizes were "tuned" to determine the optimal combination (lines 114 - 116). The accuracy was verified by viewing the graphed "history objects" (lines 123-136) that were returned from the fit_generator function (line 119).

