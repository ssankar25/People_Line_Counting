# Time_Of_Flight_Sensor_People_Tracking
This repository contains a line-counting algorithm to track the amount of people entering a room with video captured by a Time of Flight (ToF) sensor.

A convolutional neural network (CNN) that has had its kernel size and layer count optimized for the segmented head images is also included.
The network classifies whether an image contains a head or not.

CNN.py - Contains the convolutional neural network

CustomDataset.py - Links the segmented image dataset with the corresponding annotations given in the CSV file

labels.py - Used to generate CSV file for segmented image dataset

train.py - Contains training loop used to set kernel weights for the CNN

mainTvHeads.py - Image segmentation algorithm that denoises the image, locates head centers using connected component labeling, and segments the image into 100x100 blocks

line_counting.py - Algorithm that counts number of people entering a room

tracking.py - Used for applying further denoising to the test video, such as contour area filtering

Dataset link - https://drive.google.com/drive/folders/1_IZBPh19f8zDel46fh1AYXEknjyLmxcc?usp=sharing 

Dataset contains:

segments - Segmented images for CNN training, along with annotations in CSV file

test_video.avi - Original test video used in line counting after being run through tracking.py

output_video.avi - Output from tracking.py that is used in line counting

## Convolutional Neural Network
### Methodology
The CNN.py file contains the network architecture implemented with Pytorch. Each convolutional layer contains the convolution operation done with a specific kernel size, along with a max pooling layer to reduce the feature map size for the next convolutional layer and decrease computational cost. The initial convolutional layers are designed to learn the more elementary features of the head images, such as edges or textures, while the deeper layers learn larger patterns, such as the overall head shapes and bodies that are in complex positions within the images. 

The kernel size, which is the matrix size used in the convolution operation, is defined as 3x3 for all 5 convolutional layers, as I observed better validation accuracy with this configuration due to finer details being captured more efficiently with a smaller kernel size in the 100x100 images. A deeper network with 5 layers allows for heads in more complex positions to be detected, such as those surrounded by bodies, and more nonlinear activation functions to be introduced that better define the relationship between pixel intensity and head classification.


### Usage: Training the Model on a Local Machine
An IDE, such as Visual Studio Code or Pycharm, is required, along with setting up a virtual environment with Pytorch and OpenCV.

First, download the provided image dataset "segments", which contains the segmented images created by the mainTvHeads.py, which segments the images captured by the ToF sensor into 100x100 blocks. A labels.csv file is also included, has the annotations I defined for the over 700 segmented images. 

Next, run the train.py file, which loops over the dataset and performs classifications on the head images using the network defined in CNN.py and the dataset object defined in the CustomDataset class. The loss is calculated with the CrossEntropyLoss cost function, and this loss is attempted to be minimized in each consecutive epoch using the Adam backpropagation function. A sample training output is seen below.


Sample Train
:-----------:
![](https://i.gyazo.com/5e771dbc4668b6b05d9d93339213a3e0.png)

## People Counting Algorithm
### Methodology
Using a video captured by the ToF sensor, the algorithm in line_counting.py attempts to count the number of people entering a room on the provided video. The room is seen on the top left of the video. As a result, the algorithm is defined such that when a head center crosses the bottom line and then the top, the head count increments since a person enters the room. If a head crosses the top and then the bottom line, then the head count decrements. 

The frames of the video, in a while loop, are first processed by applying multiple denoising techniques, including Gaussian Blur, Max Pooling, and Morphological operations like erosion and dilation. Further denoising techniques, such as filtering out smaller contour areas, is done in tracking.py. This is done to increase the accuracy of the detected centroids. Then, the head count is updated in each frame using the tracking methodology previously described. The lines, centroids, and head count are then drawn to the video for better visualization.

### Usage
Download the test_video.avi file, which was captured by the ToF sensor, and run the tracking.py code with this video file. Once the video file completely processes, the output_video file is fed into the line_counting.py file. Run the line_counting.py file in a virtual enviornment with OpenCV installed. This should result in an output that contains the original video with lines drawn on the image, along with the detected head centers and head counts. A sample of this output is seen below.



Sample People Counting
:-----------:
![](https://s5.ezgif.com/tmp/ezgif-5-f14d44d9dd.gif)

### Current Difficulties
The current line counting algorithm, though having correct logic for updating the head count and finding the centroids, has difficulties in keeping the centroids stable and only detecting those centroids that are heads. To rectify this, I introduced further denoising techniques, such as erosion and dilation, and experimented with different kernel sizes and iterations to best optimize the center tracking. Though this did reduce noise, there are still some anomalous centroids that tamper with the line counting techniques, resulting in an incorrect head count, particularly when the heads are exiting.

This incorrect head count compounds as the video progresses since the head count in each frame relies on an accurate head count in the previous frame to make the correct prediction.
