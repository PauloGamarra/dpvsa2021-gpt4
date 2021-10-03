# Imports
import os
import json
import shutil


def convert_labels_to_yolo_format(file_path):

    """
    Convert the annotation file to the YOLO's format
    'points' will be replaced to [[xc, yc], [width, height]] where
        - xc: bounding box x point
        - yc: bouding box y point
        - width: bounding box width
        - height: bounding box height
    """

    with open(file_path) as json_file:
        data = json.load(json_file)
        image_height = data['imageHeight']
        image_width = data['imageWidth']
        print(image_width, image_height)
        
        for shape in data['shapes']:
            shape_id = shape['id']
            x1 = shape['points'][0][0]
            x2 = shape['points'][0][1]
            y1 = shape['points'][1][0]
            y2 = shape['points'][1][1]
            shape_width = x2 - x1
            shape_height = y2 - y1
            xc = (x2 - x1)/2
            yc = (y2 - y1)/2
            norm_w = shape_width / image_width
            norm_h = shape_height / image_height
            shape['points'][0][0] = xc
            shape['points'][0][1] = yc
            shape['points'][1][0] = norm_w
            shape['points'][1][1] = norm_h
            print(f'Id: {shape_id}')
            print(f'Points: ({x1}, {y1}), ({x2}, {y2})')
            print(f'Dimensions: {shape_width} x {shape_height}')
            print(f'Normalized dimensions: {norm_w} x {norm_h}')
            print(f'Center: ({xc}, {yc})')
            print(f'New shape: {shape['points']}')
            break
            

def create_yolo_dataset(SOURCE_DATASET_DIR, DESTINATION_DATASET_DIR):
    for i, game in enumerate(os.listdir(SOURCE_DATASET_DIR)):
    
        print(f'Converting game {i} ...')
    
        game_dir = os.path.join(SOURCE_DATASET_DIR, game)
    
        for file in os.listdir(game_dir):
            source = os.path.join(game_dir, file)
        
            if not os.path.isdir(f'{DESTINATION_DATASET_DIR}/train/game{str(i)}'):
                os.makedirs(f'{DESTINATION_DATASET_DIR}/train/game{str(i)}')
            
            destination = os.path.join(f'{DESTINATION_DATASET_DIR}/train/game{str(i)}', file)
        
            if source.endswith('.json'):
                convert_labels_to_yolo_format(source)
            else:
                pass
                #shutil.copy(source, destination)
            break
        break
    
    print('All files converted successfully')
