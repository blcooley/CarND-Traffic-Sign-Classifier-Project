#**Traffic Sign Recognition**

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./training_set_image_percents.png "Visualization"
[image4]: ./traffic-sign-tests/German traffic signs in real world-1-0.jpg "Traffic Sign 1"
[image5]: ./traffic-sign-tests/German traffic signs in real world-1-2.jpg "Traffic Sign 2"
[image6]: ./traffic-sign-tests/German traffic signs in real world-2-0.jpg "Traffic Sign 3"
[image7]: ./traffic-sign-tests/German traffic signs in real world-2-2.jpg "Traffic Sign 4"
[image8]: ./traffic-sign-tests/German traffic signs in real world-3-3.jpg "Traffic Sign 5"
[image9]: ./traffic-sign-tests/German traffic signs in real world-4-0.jpg "Traffic Sign 6"
[image10]: ./traffic-sign-tests/German traffic signs in real world-4-2.jpg "Traffic Sign 7"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/blcooley/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the len and set functions and the shape member variable to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the percentage of each image type in the data set. The average image should occur in about 2.3% of the images. Note that some images appear nearly 6% of the time while others appear around 0.5% of the time. However, it appears that all image types have a representative same in the training set.

![Percentage occurrence of image in dataset][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I did a quick normalization of the data by subtracting 128 from each RGB element of each image and dividing the result by 128. This normalizes the data set to lie between 1 and -1 with (128, 128, 128) as a arbitrary centerpoint. It turned out that this worked well enough to meet the project requirements, so this is the only preprocessing step I did.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x3 RGB image   							|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, vaild padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten               | outputs 400x1                                 |
| Fully connected		| outputs 120x1                                 |
| RELU					|												|
| Dropout               | keep probability = 0.6						|
| Fully connected		| outputs 84x1                                  |
| RELU					|												|
| Dropout               | keep probability = 0.6						|
| Fully connected		| outputs 43x1                                  |

This was essentially LeNet supplemented with a couple of dropout layers to reduce overfitting.

####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I played with the number of epochs, the batch size, the learning rate, and the keep_prob (probability for the two dropout layers). The biggest issue I had turned out to be getting the normalization calls correct. Once I had that correct, the base LeNet model had around an 89% accuracy on the validation set.

To train the network with the new dropout layers, I simply ran it many times. I started out by adjusting the keep_prob parameter from 0.5 up to 0.8 in increments of 0.1. 0.7 seemed to work better than 0.8. At that point, I adjusted the learning rate from 0.001 to 0.0001 and the epochs to 50. This seemed to work ok, but the best validation seemed to get stuck below 0.93, usually around 0.91. I dropped the batch size figuring that by having a worse approximation of the gradient, this might add some "innovations" to the SGD, and get out of local minima. I tried halving the batch size to 64, then again to 32, at which point I achieved validation above 0.93. Many of the epochs after 25 weren't productive, so I dropped the epochs to 25. I also tried increasing the learning rate back to 0.0005 before settling on 0.0002.

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.996
* validation set accuracy of 0.940
* test set accuracy of 0.930

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

I started with the LeNet architecture, mainly because the material suggested that it would be a good starting point. It makes sense that it would be because the image size we use is the same size as the LeNet images. Also, traffic signs are composed of many simple shapes like the lines and curves that make up written digits. I left color as a distinguishing characteristic of the data, which I think is important because German traffic signs have distinct colors like white, red, black, and blue. Noticing the color can immediately rule out many of the classification classes.

What I noticed with the LeNet model expanded to 3 color channels is that it fit the training data well, but the validation accuracy would peak around 89%. I decided that this meant the model was overfitting, so I added a couple of dropout layers.

I then iterated on the hyperparameters (epochs, batch size, learning rate, and keep_prob for the dropout layers) as described previously. In a relatively short period manual searching, I was able to achieve the project requirements. At that point I stopped tuning. However, further tuning of the combination of learning rate, epochs, and batch size should be able to increase the validation and test accuracy as my search was not exhaustive by any means. L2 regularization or more advanced preprocessing steps may help as well.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are seven German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6]
![alt text][image7] ![alt text][image8] ![alt text][image9]
![alt text][image10]

I chose these seven images because the associated signs were in a section of the earlier visualization of training set frequencies containing a smaller than average percentage of training samples. Also, these are real world signs that have lighting variations and occlusions.

Image 1 is a "road narrows on right" sign. The picture is taken against a background of sky that has a similar lighting to the sign itself. The sign is also slightly rotated.

Image 2 is a "turn right ahead" sign. This sign is relatively clear. The image is rotated, and it was among the least seen signs in the data set, around 0.5% of training images.

Image 3 is a "road work" sign. The image contains bright and dark sections and the sign is at at angle.

Image 4 is a "turn left ahead" sign. This sign is clear, but only about 1% of the test images featured this sign.

Image 5 is a "bicycles crossing" sign. The image conditions are very dark and it is difficult to see the contents of the sign.

Image 6 is a "traffic signals" sign. To me it looked like the common exclamation point sign at first glance. The image is taken in bright sunlight and the sign content is somewhat washed out.

Image 7 is a "ahead only" sign. The sign is pretty clear, but only 1% of the training set contained this sign.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Road narrows on right	| Right-of-way at the next intersection			|
| Turn right ahead 		| Turn right ahead				 				|
| Road work      		| Road work  								 	|
| Turn left ahead		| Turn left ahead      							|
| Bicycles crossing		| Bicycles crossing    							|
| Traffic signals		| Traffic signals								|
| Ahead only     		| Ahead only         							|


The model was able to correctly guess 6 of the 7 traffic signs, which gives an accuracy of 86%. This compares favorably to the accuracy on the test set of 93% given the small size of the data set.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

For all images, the model is certain that the prediction is correct. I have checked several times at a Python prompt to compute the softmax of the output of my LeNet function, and it seems correct. However, I was expecting a little more uncertainty than [1, 0, 0, 0, 0] across the board.
