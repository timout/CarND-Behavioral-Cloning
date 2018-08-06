# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: img/center_reg.jpg "Recorded image sample"
[image2]: img/center_rec_l_1.jpg "Recovery Left Image"
[image3]: img/center_rec_l_2.jpg "Recovery Left Image"
[image4]: img/center_rec_l_3.jpg "Recovery Left Image"
[image5]: img/center_fl.jpg "Flipped Image"
[image6]: img/center_cr.jpg "Cropped Image"

[angle_dist_image1]: img/angle_dist.png "Angle Dist Image"
[angle_dist_image2]: img/angle_dist_aug.png "Angle Dist Image"
[mse_image1]: img/mse_training.png "MSE Image"


[res_image1]: img/res1.gif "First test"
[res_image2]: img/res2.gif "Second test"

[res_2_image1]: img/res_t_2_1_r.gif "2 tracks first test"
[res_2_image2]: img/res_t_2_2_r.gif "2 tracks second test"

---
### Files Submitted

* `model.py`: Python script to import data, train model and save model.
* `model.h5`: Saved model.
* `video.mp4`: Saved video from model testing.
* `drive.py`: Python script that tells the car in the simulator how to drive
* `test.ipynb`: Notebook I used for model developing. Contains multiple models to test.
* `writeup.md` report summarizing the results

### Gathering data

Data was generated using Udacity's Car-Driving Simulator.  
To capture good driving behavior, I recorded:
* Approximately two laps on track one using center lane driving.  
* Approximately two laps on track one using center lane driving but in opposite direction.  
* Approximately two laps with recovering from the left and right sides of the road back to center. 
* In order to increase the dataset, images were flipped on the y-axis and multiplied their steering angle by -1. 

Here is an example image of center lane driving:

![image example][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would 
learn to drive back to the center of the road if it swerved to the side. These images show what a recovery looks like from left 
side to center:

![image example][image2]
![image example][image3]
![image example][image4]

After the collection process I had 20373 data points with the following distribution of steering angles:  
![angle distribution][angle_dist_image1]  

### Augmentation

To increase the dataset I flipped images and angles. For example, here is an image that has then been flipped:  

![image example][image1]
![image example][image5]

And steering angle distribution:  

![angle distribution][angle_dist_image2]

To make model run faster I added cropping layer to the model: Cropping2D(cropping=((70, 25), (0, 0))).  
This is an example how cropped image can look like:

![image example][image6]  

Finally the data set was randomly shuffled and 20% of the data were put into a validation set.   

### Training

Using simulator it was very easy to make a mistake save 'poor quality' driving data so I decided to store training data by saving it 
using multiple smaller tryouts. So to prepare data every tryout was was loaded from it is own directory ('data/tryNN'), 
csv files were merged into one DataFrame with paths fixed and that DataFrame was shuffled and split (80%/20/5) to training and validation sets.  
To reduce memory consumption I used 'generator' with 'load by chunks' functionality:  
``` chunk_iter = pd.read_csv(file_name, chunksize=batch_size) ```  

For training, I used an adam optimizer so that manually training the learning rate wasn't necessary. As the loss function, 
I used Mean Squared Error in order to penalize large deviances from the target. The model trained efficiently with a 
decreasing validation loss in every epoch.  

##### Model Architecture

This model architecture was a product of multiple trials. I tried several different models from very simple with hundreds of parameters 
to large one with millions of parameters. I end up with adaptation of NVidia example model with relatively big number of 
trainable parameters: 535,291.  

The final model architecture: 
```python
...
    Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3)),
    Cropping2D(cropping=((70, 25), (0, 0))),
    Convolution2D(24, (5, 5), activation='relu'),
    MaxPooling2D((2,2)),
    Convolution2D(36, (5, 5), activation='relu'),
    MaxPooling2D((2,2)),
    Convolution2D(48, (5, 5), activation='relu'),
    MaxPooling2D((2,2)),
    Convolution2D(64, (3, 3), activation='relu'),
    Flatten(),
    Dropout(.2),
    Dense(100),
    Dropout(.2),
    Dense(50),
    Dropout(.2),
    Dense(10),
    Dense(1)
...
    model.compile(optimizer="adam", loss="mse", metrics=['accuracy'])   
```
Details:  
<table>
	<th>Layer</th><th>Details</th>
    <tr><td>Preprocessing Layer</td>
		<td><ul>
				<li>Normalize</li>
                <li>Cropping: 70 from top, 25 from bottom</li>
			</ul>
		</td>
	</tr>
	<tr><td>Convolution Layer 1</td>
		<td><ul>
				<li>Filters: 24</li>
				<li>Kernel: 5 x 5</li>
				<li>Activation: RELU</li>
				<li>Pooling: 2 x 2</li>
			</ul>
		</td>
	</tr>
	<tr><td>Convolution Layer 2</td>
		<td><ul>
				<li>Filters: 36</li>
				<li>Kernel: 5 x 5</li>
				<li>Activation: RELU</li>
				<li>Pooling: 2 x 2</li>
			</ul>
		</td>
	</tr>
	<tr><td>Convolution Layer 3</td>
		<td><ul>
				<li>Filters: 48</li>
				<li>Kernel: 5 x 5</li>
				<li>Activation: RELU</li>
				<li>Pooling: 2 x 2</li>
			</ul>
		</td>
	</tr>   
    <tr><td>Convolution Layer 4</td>
		<td><ul>
				<li>Filters: 64</li>
				<li>Kernel: 3 x 3</li>
				<li>Activation: RELU</li>
			</ul>
		</td>
	</tr>  
	<tr><td>Flatten Layer</td><td><ul></ul></td></tr>
	<tr><td>Dropout Layer 1</td><td><ul><li>Rate: 0.2</li></ul></td></tr>
	<tr><td>Fully Connected Layer 1</td>
		<td><ul>
				<li>Neurons: 100</li>
                <li>Activation: Linear</li>
			</ul>
		</td>
	</tr>
	<tr><td>Dropout Layer 2</td><td><ul><li>Rate: 0.2</li></ul></td></tr>
   	<tr><td>Fully Connected Layer 2</td>
		<td><ul>
				<li>Neurons: 50</li>
                <li>Activation: Linear</li>
			</ul>
		</td>
	</tr>
	<tr><td>Dropout Layer 1</td><td><ul><li>Rate: 0.2</li></ul></td></tr>
	<tr><td>Fully Connected Layer 3</td>
		<td><ul>
				<li>Neurons: 10</li>
                <li>Activation: Linear</li>
			</ul>
		</td>
	</tr>
    <tr><td>Fully Connected Layer 4</td>
		<td><ul>
				<li>Neurons: 1</li>
                <li>Activation: Linear</li>
			</ul>
		</td>
	</tr>
</table>

#### Results
I started with number of epochs = 3:  
![test 1][res_image1]  
Result was good enough but I decided to see if results can be improved and increased number of epochs to 8.  
Initial model was without 'dropout' and validation loss started increasing after epoch = 3 and result was not that good. 
The reason of that, my guess, over-fitting.
To prevent the model from over-fitting I added 'dropout' layers and it allowed me to increase number epochs to 8:   
![mse image][mse_image1]  
And final result was:  
![test 2][res_image2]  


#### Addendum
 
After part 1 was done I decided to try with track 2.
As training data I used: 
* Track2: Driving on both lanes in both directions.
* Track2: Correction driving for both lanes.
* Previously generated data from Track 1  
The results was not very succesful.  
The car was able to be on the road but not in the same lane. The most problematic places were 
places with bigger gap in the middle line. I created good amount of 'correction' driving data for those
places and it helped but not as much as I expected.  
Track 2 files: 
* `model2.py`: Python script to import data, train model and save model - changed a little bit to simplify testing.  
* `model2.h5`: Saved model for the track 2.  

Result for the track 1:  
![test 2 1][res_2_image1]  
Result for the track 2: 
![test 2 2][res_2_image2]  

I guess next step will be line detection improvement.