## Advanced Lane Finding Project

The goals/steps of this project are the following:

* The GitHub repo can be found [here](https://github.com/DanWang1230/Advanced_Lane_Finding).
* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to the center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./images_advanced_finding_lane_lines/original_img_chessboard.jpg
[image2]: ./images_advanced_finding_lane_lines/undist_img_chessboard.jpg 
[image3]: ./images_advanced_finding_lane_lines/original_img_road.jpg 
[image4]: ./images_advanced_finding_lane_lines/undist_img_road.jpg 
[image5]: ./images_advanced_finding_lane_lines/threshold_binary.jpg 
[image6]: ./images_advanced_finding_lane_lines/threshold_color.jpg 
[image7]: ./images_advanced_finding_lane_lines/road_pre_perspective_trans.jpg 
[image8]: ./images_advanced_finding_lane_lines/road_post_perspective_trans.jpg 
[image9]: ./images_advanced_finding_lane_lines/sliding_window.jpg 
[image10]: ./images_advanced_finding_lane_lines/plot_back_result.jpg
[video1]: ./images_advanced_finding_lane_lines/project_video_dw.mp4 

### Camera Calibration

The code for this step is contained in the first code cell of the IPython notebook located in "./advanced_lane_finding.ipynb."  

I start by preparing "object points," which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result:

With distortion:
![alt text][image1]

Without distortion:
![alt text][image2]

### Pipeline (single images)

#### 1. An example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

Original image:
![alt text][image3]

Undistorted image:
![alt text][image4]

#### 2. Color transforms, gradients or other methods to create a thresholded binary image.

I used a combination of color and gradient thresholds to generate a binary image. The function is called `threshold()` in the 2nd code cell of the IPython notebook. Here's an example of my output for this step.

Combined binary:
![alt text][image5]

Color binary:
![alt text][image6]

#### 3. Perspective transform and an example of a transformed image.

The code for my perspective transform includes a function called `warp_plus()`, which appears in the 2nd code cell of the IPython notebook.  The `warp_plus()` function takes as inputs an image (`img`). The source (`src`) and destination (`dst`) points are defined inside the function. I chose the source and destination points in the following manner:

```python
src = np.float32(
    [[(img_size[0] / 2) - 60, img_size[1] / 2 + 100],
    [((img_size[0] / 6) - 20), img_size[1]],
    [(img_size[0] * 5 / 6) + 60, img_size[1]],
    [(img_size[0] / 2 + 65), img_size[1] / 2 + 100]])
dst = np.float32(
    [[(img_size[0] / 4), 0],
    [(img_size[0] / 4), img_size[1]],
    [(img_size[0] * 3 / 4), img_size[1]],
    [(img_size[0] * 3 / 4), 0]])
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580, 460      | 320, 0        | 
| 193, 720      | 320, 720      |
| 1127, 720     | 960, 720      |
| 705, 460      | 960, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image named 'straight_lines1.jpg' and its warped counterpart to verify that the lines appear parallel in the warped image.

Before perspective transform:
![alt text][image7]

After persepective transform:
![alt text][image8]

#### 4. Identify lane-line pixels and fit their positions with a polynomial.

First I applied the perspective transform to the combined binary result of Step 2. Then I used a mask function named `region_of_interest()` in the 2nd code cell of the IPython notebook to crop out the uninterested regions. The vertices for the mask are:

```python
vertices = np.array([[(200, img_size[0]),(1200, img_size[0]),(1200, 0),(200, 0)]], dtype=np.int32)
```
Then I used the sliding window method to find the lane lines and fit my lane lines with a 2nd order polynomial. The related functions are called `find_lane_pixels()` and `fit_polynomial()` in the 2nd code cell of the IPython notebook. The result is shown below:

![alt text][image9]

#### 5. Calculate the radius of curvature of the lane and the position of the vehicle with respect to the center.

I did this in function `measure_curvature_real()` in the 2nd code cell of the IPython notebook. I defined conversions in x and y from pixels space to meters and implemented the calculation of radius of curvature and offset of car center.

#### 6. Result

I implemented this step in the 12th code cell of the IPython notebook.  Here is an example of my result on a test image:

![alt text][image10]

---

### Pipeline (video)

#### 1. Final video output.

The video is located at test_videos_output/project_video_dw.mp4. Here's a [link to the file.](https://youtu.be/BisoW6Km2jo)

---

### Discussion

Here I'll briefly talk about the tricks that I used:
* Applying a mask to the warped binary image
* Using a Line class to track the polynomial coefficients for the last N iterations (16th code cell)
* Relating the sliding windows of left lanes with those of right lanes (lines 136 to 141 in 2nd code cell)
* Adding centers of the sliding windows to the good data point lists in function `find_lane_pixels()`.

The pipeline might fail with lighting condition changes (like shadows) and obvious line marks within the region of interest. To go further, I might try using the deep learning method to find a more robust solution. 
