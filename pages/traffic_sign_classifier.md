# **Traffic Sign Recognition** 

The goals/steps of this project are the following:
* The GitHub repo for this project can found [here](https://github.com/DanWang1230/Traffic_Sign_Classifier)
* Load the data set
* Explore, summarize and visualize the data set
* Design, train, and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

The project is from the self-driving car nanodegree in Udacity. So are datasets.

[//]: # (Image References)

[image1]: ./images_traffic_sign_classifier/original_image.jpg
[image2]: ./images_traffic_sign_classifier/gray.jpg
[image3]: ./images_traffic_sign_classifier/class_distribution.png
[image4]: ./images_traffic_sign_classifier/template_0.jpg
[image5]: ./images_traffic_sign_classifier/template_1.jpg
[image6]: ./images_traffic_sign_classifier/template_2.jpg
[image7]: ./images_traffic_sign_classifier/template_3.jpg
[image8]: ./images_traffic_sign_classifier/template_4.jpg


---
### Data Set Summary & Exploration

#### 1. Basic summary of the data set

* The size of the training set is 34799
* The size of the validation set is 4410
* The size of the test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43
* No abnormalities is identified

#### 2. Visualization of the dataset.

This bar chart shows how the data is distributed: blue for the training data set and yellow for the validation.

![alt text][image3]

### Design and Test a Model Architecture

#### 1. Image data preprocessing

At the first step, I decided to convert the images to grayscale because color is not an important feature in the project.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image1] ![alt text][image2]

At the last step, I normalized the image data, so the data has zero mean and equal variance and the position of the image doesn't matter.

To increase the accuracy of the validation set, I decided to change the architecture of the LeNet network from the lecture.


#### 2. Model architecture

A LeNet-5 architecture is chosen for this task. Using the original LeNet-5, I achieved high accuracy on the training set but low accuracy on the validation set (around 0.89). To solve the overfitting problem, I added dropout layers after the fully connected layers. 

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x6 	|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 5x5x16	    |
| Flatten				| outputs 400									|
| Fully connected		| outputs 120 									|
| RELU					|												|
| Dropout				|												|
| Fully connected		| outputs 84 									|
| RELU					|												|
| Dropout				|												|
| Fully connected		| outputs 43									|
| Dropout				|												|
| Softmax				|      									        | 


#### 3. Model training

To train the model, I used the Adam optimizer, the batch size of 128, 50 epochs, the learning rate of 0.001, and the keep probability of 0.5 for dropout.

#### 4. Performance

* training set accuracy of 0.998
* validation set accuracy of 0.967 
* test set accuracy of 0.947

### Test the Model on New Images

#### 1. Choose five German traffic signs found on the web

Here are five German traffic signs that I found on the web after resizing and grayscaling:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

They all are under well lighting conditions and pretty clear and should not be difficult to classify.

#### 2. Model's predictions on new traffic signs
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Traffic signal      	| Traffic signal 								| 
| Wild animal crossing  | Wild animal crossing 							|
| No entry				| No entry          							|
| Pedestrians	   		| Pedestrians					 				|
| Stop          		| Stop               							|


The model correctly guessed all the five traffic signs.

#### 3. Determine how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.

The code for making predictions on my final model is located in the 19th cell of the notebook.

For the first image, the model is 100% sure that this is a traffic signal(probability of 1), and the image indeed is a traffic signal. And the other four softmax probabilities are close to 0. It is also the case for the other four new images I obtained from the web.

### Discussion

CNN in machine learning is very good at image classification problems. This is more and more proven to be true in different applications. Compared with conventional computer vision methods, CNN shows its better adaptability and performance. Besides the LeNet-5 used in this project, other CNN architectures are worth trying for different tasks.
