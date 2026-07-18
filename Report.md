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

### Baseline Model

after implementing the baseline model, we have obtained the following results:
![alt text](image.png)
We now want to make the architecture more robust by adding some batch normalization between the layers (regularazing and reducing covariate shift maybe)
Using some type of pooling before the flattening of the maps, the idea is to change the cor after the concatenation such that it has more filters
