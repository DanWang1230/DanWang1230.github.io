# **Traffic Sign Recognition** 

The goals/steps of this project are the following:
* Load the data set of German traffic signs
* Explore, summarize and visualize the data set
* Design, train, and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images

The GitHub repo for this project can found [here](https://github.com/DanWang1230/Traffic_Sign_Classifier).

The dataset is from the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

The project is from the self-driving car nanodegree in Udacity. 

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
### Dataset Summary & Exploration

#### 1. Dataset

The train, validataion, and test datasets are pickled data that is a dictionary with 4 key/value pairs:

- `'features'` is a 4D array containing raw pixel data of the traffic sign images, (num examples, width, height, channels).
- `'labels'` is a 1D array containing the label/class id of the traffic sign. The file `signnames.csv` contains id -> name mappings for each id.
- `'sizes'` is a list containing tuples, (width, height) representing the original width and height the image.
- `'coords'` is a list containing tuples, (x1, y1, x2, y2) representing coordinates of a bounding box around the sign in the image. These coordinates assume the original image. The pickled data contains resized verisons (32 by 32) of these images


#### 2. Basic summary of the data set

Usiing `python` and `numpy` methods to calculate the data summary:

* The size of the training set is 34799
* The size of the validation set is 4410
* The size of the test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43
* No abnormalities is identified

#### 3. Visualization of the dataset.

It can be interesting to look at the distribution of classes in the training, validation and test set. We need to check if the distribution is the same and if there are more examples of some classes than others. The `matplotlib` library is a great resource for doing visualizations in Python.

This bar chart shows how the data is distributed: blue for the training data set and yellow for the validation.

![alt text][image3]

---

### Design and Test a Model Architecture

I designed and implemented a deep learning model that learns to recognize traffic signs on the German Traffic Sign Dataset. After looking into some CNN tutorials, the LeNet-5 implementation is a solid starting point. From there, I have changed the architecture of LeNet-5 to get better performance.

#### 1. Image data preprocessing

At the first step, I decided to convert the images to grayscale because color is not an important feature in the project. Here is an example of a traffic sign image before and after grayscaling.

![alt text][image1] ![alt text][image2]

The image data should be normalized so that the data has mean zero and equal variance. For image data, I use a quick way (pixel - 128)/ 128 to approximately normalize the data.

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

A validation set is used to assess how well the model is performing. A low accuracy on the training and validation sets imply underfitting. A high accuracy on the training set but low accuracy on the validation set implies overfitting.

#### 4. Performance

* training set accuracy of 0.998
* validation set accuracy of 0.967 
* test set accuracy of 0.947

---

### Test the Model on New Images

To gain more insight into how the model is working, I used five pictures of German traffic signs from the web and used my model to predict the traffic sign type. The `signnames.csv` file is useful as it contains mappings from the class id (integer) to the actual sign name.

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

The code for making predictions on my final model is located in the last two cells of the notebook. For each of the new images, I printed out the model's softmax probabilities to show the certainty of the model's predictions (limit the output to the top 5 probabilities for each image). `tf.nn.top_k` is helpful here. `tf.nn.top_k` returns the values and indices (class ids) of the top k predictions. So if k=3, for each sign, it'll return the 3 largest probabilities (out of 43) and the correspoding class ids.

For the first image, the model is 100% sure that this is a traffic signal(probability of 1), and the image indeed is a traffic signal. And the other four softmax probabilities are close to 0. It is also the case for the other four new images I obtained from the web.

---

### Discussion

* Pay attention to model underfitting and overfitting. Validataion set is used to check underfitting and overfitting. In this project, I used dropout method to reduce overfitting. There are many other methods as well, such as the L2 regularization. 

* CNN in machine learning is very good at image classification problems. This is more and more proven to be true in different applications. Compared with conventional computer vision methods, CNN shows its better adaptability and performance. Besides the LeNet-5 used in this project, other CNN architectures are worth trying for different tasks.
