**Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/center.jpg "Center driving image"
[image2]: ./examples/recovery_right.jpg "Recovery from right side image"
[image3]: ./examples/recovery_left.jpg "Recovery from left side image"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
Files Submitted & Code Quality

1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

3. Submission code is usable and readable

The model.py and model.ipyn files contains the code for training and saving the convolution neural network.
The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

Model Architecture and Training Strategy

1. An appropriate model architecture has been employed
I used the nVidia model as shown in class, without taking the 'left' and 'right images into consideration (with the proper offset change)
as I managed to do without.
I added drop out layers between the Conv layers and before the first FC layer to avoid overfitting.
The model includes Conv layers with RELU activations for nonlinearity. The preprocessing of the data amounts to normalizing the data
and cropping it. This is done within the model using Keras Layers.

2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. 
At first I used the model without and saw that from epoch 3 I started getting signs of overfitting (training error goes down but validation goes up),
training with two epochs that model wasn't brining good results too.
After adding the dropout layers I managed to train for more epochs and eventually chose 4, which the model didn't show signs of overfitting.

I trained the model with different data sets, some of fully driving the lane, some of partially getting out of specific regions 
and once on the different layer. All this to try and avoid the model of memoizing the lane, and instead actually learning how to get out
of the sides into the middle of the road, which is what essentially we tried to teach the model here.

3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.
I did tune with the dropout hyperparameter, found that 0.1 worked well.
Once I had that tuned, I could increase the epoch number to 4.

4. Appropriate training data

The training consists of the original data given by you, and incrementally I added my own.
I tried my best to try and teach the model to drive at the center of the lane, and when not at center to learn how to get to center.

I added myself driving the track in a forward manner twice - this intorduces a 'left' bias
So I added driving the track backward to try and avoid that bias.
I saw that at some turns the model has trouble, and also when it gets off center it has trouble to get back to the middle.
So I added few smaller training sets which are basically recovering from the sides of the roads into the middle.

Model Architecture and Training Strategy

1. Solution Design Approach

I started with the nVidia model as seen in class (the later videos) as it's obviously a suitable model for this task.
I added dropout to combat overfitting and decided to avoid the strategy of using the 'left' and 'right' images, and instead add
my own data.
I evaluated how well the model works and visualized the training vs validation loss to avoid overfitting.

Again, after adding the 'simple' data of driving forward and backward at the lane, I run the simulator and saw the shortcomings.
The shortcomings were basically how to recover from side driving, so I added few specialized training sets for that purpose.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

Started with two Keras Layers for preprocessing: Lambda layer that does preprocessing and Cropping2D layer that crops the data.
Following that I used:
1. ConvLayer, 5x5 filter and 24 depth, with relu activation
2. Dropout with 0.1 drop parameter
3. ConvLayer, 5x5 filter and 36 depth, with relu activation
4. Dropout with 0.1 drop parameter
5. ConvLayer, 5x5 filter and 48 depth, with relu activation
6. Dropout with 0.1 drop parameter
7. ConvLayer, 3x3 filter and 64 depth, with relu activation
8. ConvLayer, 3x3 filter and 64 depth, with relu activation
9. FC Layer of size 100 
10. FC Layer of size 50 
11. FC Layer of size 10 
12. FC Layer of size 1 



3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![center][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to 
reocover into the center of the road when drifting a bit to the sides. These images show what a recovery looks like :

![right recovery][image2]
![left recovery][image3]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would help with the driving-left bias and 
basically simulate as if we're driving the lap the opposite direction. 


After the collection process, I had 17372 number of data points. I then preprocessed this data by normalizing it and cropping it (so only the road will be visible)

I finally randomly shuffled the data set and put 0.2 of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 4 as evidenced by looking at the training vs validation loss graph,
I used an adam optimizer so that manually training the learning rate wasn't necessary.
