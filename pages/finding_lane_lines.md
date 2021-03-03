# **Finding Lane Lines on the Road** 

## The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on the work in this written report
* The GitHub repo for this project can be found [here](https://github.com/DanWang1230/Finding_lane_lines).

[//]: # (Image References)

[image1]: ./images_finding_lane_lines/Hough_transform.jpg "Hough transform result"
[image2]: ./images_finding_lane_lines/average.jpg "Average result"
[image3]: ./images_finding_lane_lines/extrapolation.jpg "Extrapolation result"
[image4]: ./images_finding_lane_lines/solidWhiteCurve.jpg "Original image"
[image5]: ./images_finding_lane_lines/gray.jpg "Gray image"
[image6]: ./images_finding_lane_lines/canny_edges.jpg "Canny Edge detection"
[image7]: ./images_finding_lane_lines/masked_edges.jpg "Masked Edges"

---

## Reflection

### 1. Describe the pipeline.

My pipeline consists of the following steps. Here, I took solidWhiteCurve.jpg as an example to illustrate the pipeline.
![alt text][image4]
First, I converted the images to grayscale.
![alt text][image5]
Then I used Guassian smoothing and Canny edge detection to find the edges.
![alt text][image6]
After that, I created a mask to mask out the regions that are not of interest.
![alt text][image7]
Then I applied the Hough transform to find the line information and drew the lines on the original image.
![alt text][image1]
Based on the lines obtained from the Hough transform, I assigned the lines to two sets: left lane and right lane, taking into consideration the slopes of the lines. In order to draw a single line on the left and right lanes, I respectively found the average of the two sets. Below is the result after this averaging step.
![alt text][image2]
Then I extrapolated and extended the two lanes to the bottom and top limits.
![alt text][image3]
Note that I didn't modify the draw_lines() function but instead modified the "main" file directly.
### 2. Identify potential shortcomings with your current pipeline

* The parameters need to be tuned carefully, which is time-consuming.
* The method only identifies lines and will not work with curves.
* The method also will not work when two lanes are merging into one wider lane since we will lost the information of one lane with the fixed mask.


### 3. Suggest possible improvements to your pipeline

* One possible solution is to use not only line detection but also curve detection.
* Using masks with adaptive sizes instead of a fixed-size mask