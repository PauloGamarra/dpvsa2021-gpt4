import cv2
from google.colab.patches import cv2_imshow 
from sklearn.cluster import KMeans
import numpy as np
import random
import torch
import argparse
from tqdm import tqdm
import os



class_to_color = {'ball': (255, 0, 255),
               'player_team_1': (255, 0, 0),
               'player_team_2': (0, 0, 255),
               'referee': (0, 255, 0),
               'outsider': (0, 204, 204)}

class_to_id = {'ball': 0,
               'player_team_1': 1,
               'player_team_2': 2,
               'goalkeeper_team_1': 3,
               'goalkeeper_team_2': 4,
               'referee': 5,
               'outsider': 6}

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, help='input video path', required=True)
    parser.add_argument('--output_video', default = 'output_video.mp4', type=str, help='output video path')
    parser.add_argument('--output_bboxes', default = './results/', type=str, help='output bboxes directory path')
    parser.add_argument('--yolo_repo', default='./yolov5', type=str, help='yolov5 repository path')
    parser.add_argument('--model_weights', default='./weights/dpvsa_detector_1080.pt', type=str, help='output video path')
    parser.add_argument('--imsz', default=640, type=int, help='model image input size')

    args = parser.parse_args()
    
    return args


def detect_video(args=None):

  # Initilizing video caputure and writing objects
  print("loading video input from {}".format(args.source))
  video = cv2.VideoCapture(args.source)

  length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
  width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
  height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
  fps    = video.get(cv2.CAP_PROP_FPS)

  print("writing video output to {}".format(args.output_video))
  video_writer = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

  # Loading pytorch object detector model
  print("loading model weights from {}".format(args.model_weights))
  model = torch.hub.load(args.yolo_repo, 'custom', path=args.model_weights, source='local') 

  # Initializing kmeans trained flag
  kmeans_trained = False


  print("using {} as model image input size".format(args.imsz))
  # Processing each frame with our pipeline
  print("Processing frames...")
  for i in tqdm(range(length)):

    ret, frame = video.read()

    if not ret:
      break

    # Getting object bounding boxes with yoloV5s 
    with torch.no_grad():
      pred = model(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), args.imsz)

    bboxes = pred.pandas().xyxy[0].copy()

    # Cropping player bounding boxes as separate small images 
    bboxes['cropped_image'] = None
    bboxes['rgb_average'] = None
    
    for idx, bbox in bboxes.iterrows():
      if bbox['name'] == 'person' or bbox['name'] == 'player':
          x1, x2, y1, y2 = int(bbox['xmin']), int(bbox['xmax']), int(bbox['ymin']), int(bbox['ymax'])
          cropped_image = frame[y1:y2, x1:x2]
          bboxes.at[idx,'cropped_image'] = cropped_image
          bboxes.at[idx,'rgb_average'] = cropped_image.mean(axis=0).mean(axis=0)

    # On the first frame, use the player patches to train a k-means model
    if not kmeans_trained and ('player' in list(bboxes['name']) or 'person' in list(bboxes['name'])):
      # cluster feature vectors
      kmeans = KMeans(n_clusters=2, random_state=22)
      avg = np.vstack(list(bboxes.loc[(bboxes['name'] == 'person') | (bboxes['name'] == 'player')]['rgb_average']))
      kmeans.fit(avg)
      kmeans_trained = True

    # Using k-mean model to cluster detected players in 2 groups
    bboxes['kmeans_result'] = None
    for idx, bbox in bboxes.iterrows():
      if bbox['name'] == 'person' or bbox['name'] == 'player':
        bboxes.at[idx,'kmeans_result'] = kmeans.predict([bbox['rgb_average']])[0]
    
    # Drawing bounding boxes and predicted labesl on frame
    for idx, bbox in bboxes.iterrows():
      if bbox['name'] == 'person' or bbox['name'] == 'player':
        if bbox['kmeans_result'] == 0:
          bbox['name'] = 'player_team_1'
          bboxes.at[idx, 'name'] = 'player_team_1'
        if bbox['kmeans_result'] == 1:
          bbox['name'] = 'player_team_2'
          bboxes.at[idx, 'name'] = 'player_team_2'

      x1, x2, y1, y2 = int(bbox['xmin']), int(bbox['xmax']), int(bbox['ymin']), int(bbox['ymax'])
      cv2.rectangle(frame, (x1,y1), (x2,y2), class_to_color[bbox['name']], thickness=2, lineType=cv2.LINE_AA)
      cv2.putText(frame, bbox['name'], (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, class_to_color[bbox['name']], 1)


    # Writing frame
    video_writer.write(frame)

    # Writing frame bboxes results
    if not os.path.isdir(args.output_bboxes):
      os.mkdir(args.output_bboxes)
    frame_results = np.array([[bbox['xmin'], bbox['ymin'], bbox['xmax'], bbox['ymax'], bbox['confidence'], class_to_id[bbox['name']]] for idx, bbox in bboxes.iterrows()])
    np.save(os.path.join(args.output_bboxes, 'result_frame_{}'.format(i+1)), frame_results)

  # After processing all frames, releasing video objects
  print("Done! Output video saved to {}".format(args.output_video))
  
  video.release()
  video_writer.release()
  cv2.destroyAllWindows()  

def main():
  args = parse_opt()
  detect_video(args)

if __name__ == '__main__':
  main()
