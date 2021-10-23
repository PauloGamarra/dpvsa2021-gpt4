# dpvsa2021-gpt4
## About
Implementation of a soccer match monitoring system for the “Computer Vision Challenge: Soccer Match Monitoring” at the IEEE DPVSA 2021.
The team is formed by CS undergraduate students at Federal University Of Rio Grande Do Sul (UFRGS), Paulo Gamarra Lessa Pinto, Thiago Sotoriva Lermen and Gabriel Couto Domingues menored by the Professor Claudio Rosito Jung, PhD. 

In this project we have four main goals: team recognition, ball tracking, referee tracking and player tracking. To solve this, we use classic computer vision techniques and machine learning.

## Running the model

## Dataset
To train our machine learning models and validate the metrics we use an augmented DPVSA Dataset version. We used rotations between -11 and +11 degrees, horizontal flip, Hue between -37 and +37 degrees, brightness between -34% and +34%, exposure between -18% and +18%, blur up to 1px.

## Approach
We divide the task into 2 different segments: object detection (including players, ball, referees and outsiders) and players clustering. Firstly we preprocess the dataset to obtain the data to train the model. After that, loading [YOLOv5s](https://github.com/ultralytics/yolov5) model and training it from scratch based on the dataset classes mentioned in the alst section. We perform objetc detection using the model to get all the bounding boxes predictions and the confidence score for each class. 

As soon as the first segment is done, we have our object detection approach done. The next step is divide the players by team. For that, we use the clustering technique through K-means algorithm. We calculate the RGB average per channel for each predicted player bounding box and we classify this image in two different classes: 'player_team_1' and 'player_team_2'.
## Results
