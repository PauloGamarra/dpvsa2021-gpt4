# Computer Vision Challenge: Soccer Match Monitoring” at the IEEE DPVSA 2021 - Team GPT4

<p align="center">
  <img src="https://user-images.githubusercontent.com/49798588/138575391-898fc3a0-1fce-45d7-8148-43a44a5c38ae.gif" width="80%">
</p>

## About
Implementation of a soccer match monitoring system for the “Computer Vision Challenge: Soccer Match Monitoring” at the IEEE DPVSA 2021.
The team is formed by CS undergraduate students at Federal University Of Rio Grande Do Sul (UFRGS), Paulo Gamarra Lessa Pinto, Thiago Sotoriva Lermen and Gabriel Couto Domingues mentored by the Professor Claudio Rosito Jung, PhD. 

In this project we have four main goals: team recognition, ball tracking, referee tracking and player tracking. For this task, we use classic computer vision techniques and machine learning.

## Running the model
The develpment process was using the Google Colab's environment. Therefore, our recommendations are to run the code using a Python Notebook inside the Colab's environment through the following link: 

<p align="center">
  <a href="https://colab.research.google.com/drive/18fpSzTw_Rp3z_EzOJbENmjwo9UXhrI7v?usp=sharing">
    <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
  </a>
</p>

Using the link above you can read and execute the code, but if you want to run some tests or edit the code, you should copy the colab notebook to you own environment, for example Google Drive. Inside the Colab's envirment you can find other tips and specifications.

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

