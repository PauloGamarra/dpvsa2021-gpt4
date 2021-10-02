# Import utils and dataloader

import cv2
import json
import numpy as np
import torch
import pathlib as pl

class LabelmeDataloader(torch.utils.data.Dataset):
    """Dataloader for Labelme json format"""

    def __init__(self, json_folder, class_to_id):
        """
        :param images_folder: String with the path of the folder containing the images
        :param xml_filename: String with the filename of the output of the CVAT
        :param class_to_id: Dictionary. {'class_name':id, ...}
        :param augmentation: Bool defining if there is data augmentation or not
        :param max_side: Size of the biggest side of the image
        """
         
        self.json_file_list = []
            
        json_folder = pl.Path(json_folder)
        
        json_files = json_folder.glob('*.json')
        
        for json_file in json_files:
            
            img_file = json_file.parent / (json_file.stem + '.png')
            
            if not img_file.exists():
                print('Could not match json %s with image %s' % (json_file, img_file))
                continue
                
            self.json_file_list.append(json_file)

        self.class_to_id = class_to_id
        
        tuple_items = ()
        for key, _ in self.class_to_id.items():
            if key != 'dontcare':
                tuple_items += (key,)

        self.tuple_classes = tuple_items

    def __len__(self):
        
        return len(self.json_file_list)

    def __getitem__(self, idx):
        
        json_file = self.json_file_list[idx]
        img_file = json_file.parent / (json_file.stem + '.png')

        with open(json_file, 'r') as f:
            json_gt = json.load(f)
            
        img = cv2.imread(str(img_file), cv2.IMREAD_COLOR)[..., ::-1]

        shape = img.shape
        
        # Creates label image and sets everything to background
        label_np = np.zeros((shape[0], shape[1]), dtype=np.float32)
        label_np[...] = -1

        img_th = torch.from_numpy(img.copy())

        return {'image': img_th, 'bboxes': json_gt['shapes'], 'img_np': img.copy()}



