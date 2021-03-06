# MNIST - Introduction to CNN's

This project served as an introduction to CNN's and their applications in Kaggle. 

The goal of the project was to implement a convolutional neural network to predict the value of 28x28 greyscale handwritten digits. This is a classic introductory problem in the field of deep learning.

There are a few improvements that could be made to V1:
* The model training was very short (10 epochs) and so dedicating more time to training the model would likely improve its accuracy (V2: 40 epochs)
* I only used the data provided by Kaggle. There are more MNIST-style datasets (28x28 greyscale handwritten digits) which I could add as training data
* I did not implement any form of data augmentation (V2: data augmentation added)

These improvements might be added at a later date to enhance the performance of the model.

### Results
V1 (22/10/21) achieved an accuracy of 0.98632, placing me at 624 on the leaderboard of the Kaggle Digit Recognizer starter competition (note this is a 2 month rolling leaderboard).

V2 (13/11/21) achieved accuracy of 0.99432, placing me at 200 on the leaderboard. This was achieved by adding data augmentation and increasing the number of epochs.
