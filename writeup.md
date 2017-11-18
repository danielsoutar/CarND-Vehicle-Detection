
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/training_car.png "Car-classified image"
[image2]: ./examples/training_notcar.png "Non-car-classified image"
[image4]: ./examples/detected.png "Detected cars"
[image5]: ./examples/heatmap.png "Heatmap"
[image6]: ./examples/final_result.png "Detected cars in pipeline"
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features (and colour features) from the training images.
#### 2. Explain how you settled on your final choice of HOG parameters.


The code for this step is contained in the fourth code cell of the notebook, although I adapted this code slightly for greater efficiency in the video pipeline further on.  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image2]

HOG features are simply gradient features at a chosen level of granularity. The idea is that they can give some notion of a 'signature shape' for a car as opposed to noise. When I read in my dataset (which I should add was augmented by flipping them across the y-axis), I converted the image to LUV colour space and then fed in the individual channels to skimage's `hog()` function. Colour features were easy enough to extract with this scheme, and I simply fed in the LUV variant of the image to the colour feature extraction functions `bin_spatial()` and `colour_hist()` respectively.

I had explored various colour spaces in the classes and the LUV and HLS colour spaces had seemed to work particularly well, and I chose the former as implied above. I also played around with the different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`), and decided to include a few colour features as well in the end as they incurred little overhead and improved reliability. Thankfully I had worked on this before in lectures (with a notebook 'Plots.ipynb' attached to verify some playing around was done), and it appeared as though the following parameters gave a decent accuracy/performance trade-off (*relative to other combinations in this context*, a point I will return to later):

| Parameters    |
|:-------------:|
| Orientations:       	  8      	|
| Pixels per cell:    	  9 	 	|
| Cells per block:	  	  2	     	|
| Number of HOG channels: 3 (all)   |
| Spatial compression:	  16x16     |
| Number of spatial bins: 32 	    |

Impressively, I was able to reduce the colour features substantially, because as Arpan hinted at in the lectures, you could reduce the size of the image down to as little as a 16th of its original size - bizarrely giving an *increase* in performance! Seems the reduction in sensitivity is probably to thank, although any further seemed to worsen performance so this was as low as I got. I cannot believe that any kind of fault could possibly be given to the colour features. As will be discussed, however, I think my HOG features were not the best for this project, although given time constraints and the initial promise they gave I think this was only something I could have learned by doing.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using `sklearn.svc`. Without even any kind of modification, I was able to get a very impressive 99.34% test accuracy on the test set (which being augmented was very large remember). The actual code for this is in the 7th code cell of the notebook. The main problem - as we will see later - is that this step was UNBELIEVABLY slow and I actually had issues with swap space due to Python hungrily devouring nearer 10 GB of data in the process. I had thought this was the SVM (and due to time constraints couldn't investigate), but I am now more confident it was the feature extraction that was the issue, once again I defer this to the end discussion.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search the sides and the distance in the centre for new cars, and the idea would be I would search in a targeted fashion for the remaining cars that I had picked up. This is a clever idea and realistically seemed close to what we humans actually do ourselves. It was overwhelmingly trial-and-error with a lot more error than trial to be honest. One particularly annoying point was that some divisions ended up saying I couldn't search the images at all with certain scales and regions. My belief is that this is due to the annoying feature of my model using 9x9 cells as opposed to 8x8. The overlap was pretty high in those regions because I figured that if I was only search a relatively small portion of the image that I could afford such overlap anyway and get a strong signal for refined search to pick up on later on. As it turned out, I didn't even need the tracking searches for the project video - the right one on its own was completely sufficient to pick them up! 

I should use this as a confession that I only search in the right half of the video - chiefly because the video took unbearably long to load otherwise. One scale at which I searched is removed because it takes *half an hour* to record just the first 40 seconds of the video(!!!!), although I do miss the white Toyota for a couple of frames. I should note however that in the video you will see that the pipeline works fairly well, with hardly any modification bar one, which all else equal would have suggested it should have performed badly!

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using LUV 3-channel HOG features plus spatial and histogram colour features (from LUV as well) in the feature vector, which provided a nice result.  Here's an example image:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video by returning a list of boxes where a match was made.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a frame of video and the bounding boxes then overlaid on the frame:

![alt text][image5]

### Here the resulting bounding boxes from the final pipeline are drawn onto the frame:

![alt text][image6]

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

So this project wasn't too difficult (in my opinion) to do a reasonable job in. But to make a pipeline that was efficient and capable of scaling to video was very hard.

One point I'd like to make was that my feature extraction was what made my code suffer so much, which due to time constraints meant I had little time for testing and iterative prototyping. The model and feature extraction had worked very well in the lectures, but I now appreciate that my HOG features were very expensive, meaning that pipeline is quite slow and so even a few searches are punishingly expensive. One thing to commend my pipeline is that it only searched in every second frame of video, which thanks to the quality of the model doesn't seem to stop it from correctly pinpointing the two cars (although the size of the frame meant I missed the Toyota as mentioned earlier).

Another thing that was definitely likely to be problematic was my pixel-per-cell parameter, as this was different to many other people's. In my defence my model's accuracy was amazingly high, although I confess that a slightly lower accuracy with greater ability to search would have made my pipeline much faster.

The nice thing about my pipeline is that it handled the merging and diverging of the two cars very well, although the separation could have been a bit smoother. The boxes could be smoothed over several frames to give a smoother result, and the tracking of cars would be very nice. A particularly nice extension would be that if one discovers a car in the side regions, then it only does the targeted search until that car moves out of the region, where it then starts searching the side again (on top of the tracking search).








