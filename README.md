# Computer Vision Challenge: Soccer Match Monitoring” at the IEEE DPVSA 2021 - Team GPT4

<<<<<<< HEAD
![WhatsApp-Video-2021-10-23-at-21 31 20](https://user-images.githubusercontent.com/49798588/138575391-898fc3a0-1fce-45d7-8148-43a44a5c38ae.gif)
=======
<p align="center">
  <img src="https://user-images.githubusercontent.com/49798588/138575391-898fc3a0-1fce-45d7-8148-43a44a5c38ae.gif" width="80%">
</p>
>>>>>>> 3490534af5a8f47e185caed57b8dfe12c9172e89

## About
Implementation of a soccer match monitoring system for the “Computer Vision Challenge: Soccer Match Monitoring” at the IEEE DPVSA 2021.
The team is formed by CS undergraduate students at Federal University Of Rio Grande Do Sul (UFRGS), Paulo Gamarra Lessa Pinto, Thiago Sotoriva Lermen and Gabriel Couto Domingues mentored by the Professor Claudio Rosito Jung, PhD. 

In this project we have four main goals: team recognition, ball tracking, referee tracking and player tracking. For this task, we use classic computer vision techniques and machine learning.

## Running the model

## Dataset
To train our machine learning models and validate the metrics we use an augmented DPVSA Dataset version. We used rotations between -11 and +11 degrees, horizontal flip, Hue between -37 and +37 degrees, brightness between -34% and +34%, exposure between -18% and +18%, blur up to 1px. 

The original dataset is divided into different classes: ball, player_team_1, player_team_2, goalkeeper_team_1, goalkeeper_team_2, referee and outsider.To feed our model we replaced few base classes into a small subset. Our preprocessed dataset is splitted into ball, player, referee and outsider.

<<<<<<< HEAD
The following image is a sample from the DPVSA dataset.

![Screenshot from 2021-10-23 21-51-22](https://user-images.githubusercontent.com/49798588/138575735-3c60d42e-6a05-4cf1-aea4-f0d17e25ee11.png)

## Approach
We divide the task into 2 different segments: object detection (including players, ball, referees and outsiders) and players clustering. First, we preprocess the dataset to obtain the data to train the model. After that, loading [YOLOv5s](https://github.com/ultralytics/yolov5) model and training it from scratch, using the dataset classes mentioned in the last section. We perform object detection using the model to get all the bounding boxes predictions and the confidence score for each class. 

As soon as the first segment is done, we have our object detection approach done. The next step is to divide the players by team. We use the clustering technique through K-means algorithm. We calculate the RGB average per channel for each predicted player bounding box and we classify this image in two different clusters: 'player_team_1' and 'player_team_2'.

![diagram1](https://user-images.githubusercontent.com/49798588/138575649-7e641f96-0f45-418d-be4c-5601d41e8d0e.jpg)


## Results
To evaluate our model we focused on two different metrics and three different losses during both training and validation steps. The following image shows the results during the training and valitation during 100 epochs.

![results](https://user-images.githubusercontent.com/49798588/138575042-e520e62b-3f9b-4d5a-87f7-b4c0ad8c32f0.png)

To compute the precision for each class, we computed the confusion matrix as follows:

![confusion_matrix](https://user-images.githubusercontent.com/49798588/138575086-a27c8833-c1d6-4502-a8c1-57f5a67c4fe2.png)

=======
The following image are two samples from the DPVSA dataset.

<p align="center">
  <img src="https://user-images.githubusercontent.com/49798588/138706232-0f190bba-89b6-4afb-b25b-ca6ddfbe987b.jpeg" width="80%">
</p>

## Approach

We divide the task into 2 different segments: object detection (including players, ball, referees and outsiders) and players clustering. First, we preprocess the dataset to obtain the data to train the model. After that, loading [YOLOv5s](https://github.com/ultralytics/yolov5) model and training it from scratch, using the dataset classes mentioned in the last section. We perform object detection using the model to get all the bounding boxes predictions and the confidence score for each class. 

As soon as the first segment is done, we have our object detection approach done. The next step is to divide the players by team. We use the clustering technique through K-means algorithm. We calculate the RGB average per channel for each predicted player bounding box and we classify this image in two different clusters: 'player_team_1' and 'player_team_2'.

<p align="center">
  <img src="https://user-images.githubusercontent.com/49798588/138604540-e12845d6-2033-4551-9626-0cc1233e5ca2.png" width="80%">
</p>


## Results
To evaluate our model we focused on two different metrics and three different losses during both training and validation steps. The following image shows the results during the training and valitation during 100 epochs.

<p align="center">
  <img src="https://user-images.githubusercontent.com/49798588/138575042-e520e62b-3f9b-4d5a-87f7-b4c0ad8c32f0.png" width="80%">
</p>

To compute the precision for each class, we computed the confusion matrix as follows:

<p align="center">
  <img src="https://user-images.githubusercontent.com/49798588/138575086-a27c8833-c1d6-4502-a8c1-57f5a67c4fe2.png" width="80%">
</p>
>>>>>>> 3490534af5a8f47e185caed57b8dfe12c9172e89

