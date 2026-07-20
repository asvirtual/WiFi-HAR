# Report of the Project

In this file we will describe all the steps we have followed to obtain the solution starting from the baseline.

### Dataset

Starting from the baseline architecture, we opted to train and validate the model using the data from just the S1 folder, which contains the data obtained from the movements of person 1 always in the same room divided in 3 days, we used the first 2 to train and the last one to validate / test.

- in this way we just train using always the same room / person so the generalization to other rooms / person can be done by exploiting other of them as the test set

The data for a given folder are divided other than for the day (which is represented by the letter a, b or c) also by:

- The type of movement registered, which can be either jump, walk, stay still, sit, run, ...
- the channel that has been used for that track, indeed we have 4 channels for each registration, in this way we are less dependent by the position of the router in the room

The data registered for each folder, day, movement and channel are composed by a couple of minutes of registration of the same activity and what we are interested in are windows of a couple of seconds, indeed in input we want matrices of 340 x 100. \
So what we did was take each matrix composed by a couple of minutes of data and split them into smaller matrices of 340x100 so by dividing on the first dimension we obtained the required format
But just by partition each file we would have got only a few dozens of data.\
A better idea consists in using sliding windows that every 5 columns take another screenshot of the matrix, in this way we obtain a lot of data that are correlated with each other but have the important information in different position, making the data less prone to overfit since it has to detect the element indendently from where it is in the matrix and at the same time give us more data to train and validate the algorithm.
We opted for a stride equal to 5 and not smaller since we found it a good tradeoff between having enough data to train and validate the model on and having some variety in the data, causer otherwise with a pooling layer the invariance would make following screenshots basically the same so it would make the network overfit on data it has already saw which is a no-sense.

### Baseline Model

after implementing the baseline model, we have obtained the following results:
![Baseline Curves](./src/plot_data/training_curves_baseline.png)
We now want to make the architecture more robust by adding some batch normalization between the layers (regularazing and reducing covariate shift maybe)
Using some type of pooling before the flattening of the maps, the idea is to change the convolution layer after the concatenation such that it has more filters so the flattening is more "clean" while reducing height / width.

- The Pooling layer we opted to use is the adaptive max pool since it is better than the average in the case of the spectogramm since it is more able to identify the peeks while the average risks to diluite the changements making them noisier, moreover the maxpooling offer a great invariance to translations making it suitable for our case where we slide the window between the different instances.

Moreover we change the loss function to make it less sure about its own decision such that it is more capable of generalizing and less prone to overfit
We also put some kind of L2 regularization by increasing the weight decay cause u never knoe maybe its a good idea
The last optimization in the trainig process was to make the learning rate adaptive in a temporal way such that it is faster at the beginning and then slower when it need to find the optimum. To do so we opted to use the Cosine Annealing.
We obtained the following results:
![Updated Baseline Curves](./src/plot_data/training_curves_baseline2.png)
Which we can see tends to overfit in a reduced way but there is still a lot of noise in it, which we would like to remove

To do so we try to do two different things, change the pooling to not be adaptive but such that it just have a kernal size equal to 2 and stride = 2 so it reduce by 4 the dimension. Moreover we opted to use the InstanceNormalization instead of the BatchNormalization which basically normalize based on the value of the cells of the matrix instead of normalizing based on the value of the cells in different instances in the batch, which theoretically should make the model focus more on the content since each value in the matrix is normalized by the values in that matrix making matrices with different type of noises behave in the same way even if the user is positioned in different ways from the transmitter/receiver of signals.
We also opted in implementing gradient clipping since we saw that in case of strange batches of data we obtain way worst performances and make the model way worst with the risk of not coming back (collapse of the model).
With this new updates indeed we obtained a way better model that is less affected by noise and way more stable:
![Updated Baseline Curves 2nd Version](./src/plot_data/training_curves_baseline3.png)

The last thing we tried to implement on the standard convolutional architecture was to increase the number of training data by exploiting data augmentation, indeed in the case of the spectrograms we have specific techniques like the masking block

- Time Masking -> set to zero values in a given row, the model has to understand the activity even in case of temporal holes in the signal
- Frequency Masking -> set to zero values in a given column, the model has to understand the activity even in case of a specific frequency that is affected by a total interference

We have also implemented the confusion matrix to have more information about what is happening in our training and which are the classes that we are better at identify and which are the ones we have more difficulties:
![Confusion Matrices 2nd Version](./src/plot_data/confusion_matrix_baseline.png)
What we can notice is that some classes seems more difficult to predict than others:

- the network confuse L for E, meanwhile is very sure when it comes to predict E , seems like prediliges E between the two when it is undecided, probably given by the fact that the samples with E activities are more numerous than the ones with E, but still E has a decent amount of samples. If E represents the absence of movements, then L should be an activity with a very low energy and the network looking at the low energy in the doppler spectrogramm opts the safest way where there are more samples
- Jump Run and Walk are quite similiar to each other given the fact that they are big movements, especially jump have a low accuracy since it is often confused for walking
- H and C are quite interchangable since they are both the only other class that the model confuse the other with, so they could have a very similiar spectral firm
- S is confused for almost every class that requires a quite low amount of energy (so all other than jump, walk and run)

Our hypothesis is that the problem resides in the fact that the convolutional network is not able to classify correctly activities of similiar level of energy since they miss part of the temporal information and by the fact that movements at low energy levels could be confused by some background noise

So what we want to do now is to try to implement some kind of recurrency in the network such that we can extrapolate more information from each matrix.\
The idea is to put some kind of recurrent network after the inception layer, in this way we make the convolution filters extrapolate the useful patterns and then we pass those patterns in input to the recurrent layer that learn the evolution of the action in time such that we have more information to base our prediction on.\
Initially we implemented a bidirectional LSTM layer where we basically take in input the output of the convolutional network and indicate the temporal dimension as the one we want to use to extract the recurrence and unify the filter and frequencies dimensions as the other dimension that we want to extrapolate informations on. Indeed istead of flattening all dimensions together, we preserve the time dimension as a sequence.\
The bidirectional processing let the model leverage all the temporal context in both direction which is useful since we have the hole spectogramm to compute at the beginning and give us more informations. The output of the LSTM is averaged across the time dimension to obtain a single global representation to pass to the head projection used to merge the informations of the 2 directions together:
![Updated Baseline Curves 2nd Version](./src/plot_data/training_curves_recurrent.png)
![Confusion Matrices 2nd Version](./src/plot_data/confusion_matrix_recurrent.png)
