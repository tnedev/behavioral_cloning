# Behavioral Cloning

This project is part of Udacity's self-driving cars nanodegree.
The task is to design a deep neural network to clone the human driving behavior.
To make testing and obtaining data easier, Udacity made a simulator, however, 
the data and techniques used are no different for real world application

The objective is to design a steering model for the car. Driving image comes in, steering angle comes out.
If successful the simulated car will be able to drive autonomously on the provided track.
We are also able to control the throttle of the car, but I decided that for this case, controlling the throttle is pointless. 

The simulator could provide front, left and right images with 320x160 images. 
I decided to only use the central(front) images, because later I want to test the model with a real life data, and obtaining single single camera data will bi a lot easier. 


I've spent days into experimenting and training with different models. 

First, decided to start with the Comma.ai steering model.
Found here (https://github.com/commaai/research/blob/master/train_steering_model.py)

I've started with a plain model weights instead of obtaining the weights of their training. 
After many iterations of obtaining data (driving the simulator), preprocessing and training, I could not make
this model behave well. I managed to go through half the track autonomously, but is seems like the model is too complex for
the data and without and initial training of transfer learning I won't be able to make it work. 

Second, I made my own model with 2 conv layers and 3 fully connected layers. Again, after many tries for collecting data and training
I could not make the model generalize well. 

My main confusion was with simultaneously collecting the data and testing my model. I did not know if the 
data I collected was enough, was it balanced well or was the problem in the model. This is a very vicious cycle of 
data collection and preprocessing and model changes. With this complexity of options is hard to decide on a model and data.

Third, I completely switched gears and decided to go minimalistic. I preprocessed all images to grayscale and 32x16, as I read
some people manages it run with it. My model stayed the same as previous one. I started completely fresh data minimalistic style. 
Did only two laps and then two more laps of data where the car goes off track and I turn it back to the center. Immediately saw improvement inn driving behavior, 
and the simulator car was able to drive well on 3/4 of track. The last two corners were hard. I started collecting more and more data just on those corners, 
but did not see improvement and even the opposite happened. After doubling the data, the car started behaving worse. 
I really liked this minimalistic style but it seemed like the car was not able to generalize. It drives reasonably well, but
it looked like it remembers rather than understands. I saw that this model could be made to fully work, but I was already
tired from experimenting without good results. Decided to move forward and try another approach.

Fourth and final approach was to test with transfer learning. 
I decided to go with the VGG model and Imagenet weights. Removed the last block of the VGG and added my layers on top. 
I knew I needed more data to make this work. Therefore, I also decided to flip each image and angle which will double every data point.
Even after my first training the car behaved very differently it seemed that it was able to generalize and knew that it should
not go over the marked lines and offtrack. Because the model is complex and images are quite big it takes a while to train. 
I used my Azure GPU VM with 12 CPU cores and 2xTesla K80 GPUs. With it training with 80k data points with Python generators it takes about
2 hours. Most of this time comes from using generators instead of in memory loading all the data.
After a few iterations and improvements and training cycles of the model I was able to successfully make it through the first track.

The final model all together uses 16 layers including the VGG ones.
On top of the VGG model we have 4 1D convolution layers with RELU activation.
They all use filter size of 3 pixels, keep the size of the layer the same and
the numbers of filters doubles with each layer starting from 32.
The model is then flattened fallowed by 3 fully connected layer with RELU activation (size of 1024, 512, 128). A dropout of 50% is added
between the first fully connected layer and the rest. This reduces the risk of overfitting.
The final layer is a single fully-connected layer wit tanh activation for smooth -1, +1 angles.

The final model looks like this:
[logo]: ./img/model.png

The reason behind using the VGG model underneath is that the VGG is very capable of recognizing images. Therefore, with
our driving the model quickly gets that is should stay between the lines. If we did not provide the VGG, during the training
our model should learn the concepts of figures and recognize them. By using the VGG as base, we skip this step.

The architecture on top of the VGG model is based on experimentation and inspired by NVIDIA's steering model. The convolution layers were chosen because of their 
capability in image recognition. 

The model was trained in 2x3 epoch with Adam optimizer and Mean Squared Error, however the second round of training did not lower the MSE error and the driving.
A 10% validation set was used to limit a potential overfitting. Changing the learning rate of 0.0001 did not seem to have affect on the results.
Using a test set data did not seem to provide useful results.

Normally, when you play such a driving game, you remember how to play, but you also learn the track which makes it a lot easier.
Our model has no idea of which comes after which. Even further, during each epoch of training, image order is randomized.
However, this keeps me asking if a recurrent model which has the concept of memory will perform better in the game.
This idea might be unsuitable for real life steering angle model, but for going around a track makes sense.
Looking forward to do the experiment. 

Funny enough, this model drives very well, but it is unable to run on my laptop in real time due to its complexity.
To be able to test I run the simulator on my laptop while the model runs on the Azure VM, connecting them though a SSH tunneling.
However, even that my network connection is good there is a significant lag maybe because of the nature of the ssh connection
and that it has to send 10 images a second. Therefore, I was only able to run in autonomous mode in 0.15 throttle.
Every more throttle and the lag makes the car goes offroad.

Another problem I found is with the simulator. In the simulator the car could easily get stuck on the sided of the road.
It's a pity, because otherwise we would be able to see the car completely off and the get back to the track. It did not get stuck with also
made sense for the model to control the throttle and being able to go back. However, due to those limitations, we only need to drive on track all the time.

Overall, I did days of training and experimenting to come with a good model, but in the process I learned a lot. 
The task is hard. I could spend further weeks of experimenting in order to come with a better model and optimum training data.
I am happy with the results and I will stop here with the simulator, however, I will continue with real life data.
I will first use the data provided by Comma.ai and then try collecting my own data by producing a simple device to do so.

Fun times ahead!