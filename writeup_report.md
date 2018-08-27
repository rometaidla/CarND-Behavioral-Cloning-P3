# **Behavioral Cloning** 

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

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

The _'Behavioural cloning.ipynb'_ jupyter notebook contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64. All convolutional layers use RELU activation functions to introduce linearity.

#### 2. Attempts to reduce overfitting in the model

##### Dropout
Dropout layers with probability 0.5 is added to first 2 fully connected layers. Dropout was not added to convolutional layers as weights are shared between spatial positions so there shouldn't be huge number of parameters to overfit. 

##### Regularization
Didn't try regularization as model already performed quite well without it.

##### Training

The model was trained and validated on different data sets (80/20% split) to ensure that the model was not overfitting. The model was tested iteratively by running it through the simulator after every change and ensuring that given change actually improved how vehicle is staying on the track.

#### 3. Model parameter tuning

##### Optimizer

The model used an adam optimizer, so the learning rate was not tuned manually. 

##### Hyperparameters

Hyperparameters from NVIDIA blog was used, didn't need to fine tune them.

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

##### Normalization
Keras lambda layer is used to normalize data around mean 0, which improves vanishing gradients and also increases convergence.

##### Data augmentation

Was not needed as network already performed with current dataset, but if bigger dataset is needed, Keras generators must be used as memory requirements were already limiting factor on _AWS EC2 g2.2xlarge_ instance.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image   						| 
| Convolution 5x5     	| 24 filters, 5x5 kernel, 2x2 strides	        |
| Relu					|												|
| Convolution 5x5     	| 36 filters, 5x5 kernel, 2x2 strides  	        |
| Relu					|												|
| Convolution 5x5     	| 48 filters, 5x5 kernel, 2x2 strides 	        |
| Relu					|												|
| Convolution 3x3     	| 64 filters, 3x3 kernel, no strides 	        |
| Relu					|												|
| Convolution 3x3     	| 64 filters, 3x3 kernel, no strides 	        |
| Flatten	            | Flattens from 2D to 1D                        |
| Dropout               | Dropout with 0.5 probability					|
| Fully connected		| Input 1164, outputs 100   					|
| Dropout               | Dropout with 0.5 probability					|
| Fully connected		| Input 100, outputs 50      					|
| Fully connected		| Input 50, outputs 10       					|
| Fully connected		| Input 10, outputs 1       					|

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior I iteratively did small changes or added new training data and tested how change affected driving. These were the steps:

##### 1. Flatten layer

_Change_: added just one fully connected layer and tested it with provided test data.

_Result_: car drives just circles, so behaves quite badly.

##### 2. Normalization

_Change_: added data normalization

_Result_: validation loss decreased dramatically, but car behavior didn't improve, it still kept circling

##### 3. Network architecture

_Change_: implemented network architecture from NVIDIA blog: https://devblogs.nvidia.com/deep-learning-self-driving-cars/

_Result_: car is already driving on road, but still wanders on lines and off road.

##### 4. Recorded training data

_Change_: as architecture was already quite complex and car behavior still quite rudimental, I decided to record new training data. 

I recorded around 2 laps both direction of track. Tried to stay in middle of road and be as smooth as possible.

_Result_: car drives quite well, but for some corners goes over the lines, loses track and don't know how to get back on track. This is probably caused from training data as while recording I only drove in middle of road.


##### 5. Additional training data

_Change_: to help car to find way back to middle of road I recorded the vehicle recovering from the left side and right sides of the road back to center.

This was done in everywhere track had different line markings.

_Result_: vehicle mostly stays in middle of road, but in some places like tight bends it still went off the track

##### 6. Cropping

_Change_: seemed too much work and overfitting to specific track to record more cases where vehicle recovers to center of track decided to try other solutions. Added cropping top and bottom of the image.

_Result_: model worked better, but it still struggled in bends and seemed like steering angle was limited to quite small values

##### 7. Dropout

_Change_: added dropouts to 2 fully connected layers to avoid overfitting

_Result_: model manage now to take first tight bend, but still was not able to take second tight bend

##### 8. Multiple camera angles

_Changes_: added usage of left and right cameras with steering correction of _0.2_

_Result_: car drove very well, staying in middle of road and easily took even the tight corners. 

This is the final model I submitted. I also played around with second track, but didn't find a way how handle tight ascents, so the architecture in current state does not work on second track. To better understand what change is needed, would probably need to add visualizing internal CNN states.