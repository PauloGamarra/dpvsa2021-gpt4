# Import libraries
from datasets import *
import random

def test_dataset(train_path, val_path, class_to_id, class_to_color):
    train_dataset = LabelmeDataloader(train_path, class_to_id)
    val_dataset = LabelmeDataloader(val_path, class_to_id)
        
    #Example of taking a random sample and reading the annotation
    idx = random.randrange(len(train_dataset))
    sample = train_dataset[idx]
    image = sample['img_np']
    count = 0
    for bbox in sample['bboxes']:
        cv2.rectangle(image, tuple(map(int, bbox['points'][0])),\
              tuple(map(int, bbox['points'][1])),\
              class_to_color[bbox['label']], thickness=2, \
              lineType=cv2.LINE_AA)
        
    cv2.imshow('image', image)


    #waits for user to press any key 
    #(this is necessary to avoid Python kernel form crashing)
    cv2.waitKey(0) 
      
    #closing all open windows 
    cv2.destroyAllWindows() 
	
if __name__ == '__main__':
	class_to_id = {'ball': 0,
               'player_team_1': 1,
               'player_team_2': 2,
               'goalkeeper_team_1': 3,
               'goalkeeper_team_2': 4,
               'referee': 5,
               'outsider': 6}

	class_to_color = {'ball': (random.randint(0,255), random.randint(0,255), random.randint(0,255)),
		       'player_team_1': (random.randint(0,255), random.randint(0,255), random.randint(0,255)),
		       'player_team_2': (random.randint(0,255), random.randint(0,255), random.randint(0,255)),
		       'goalkeeper_team_1': (random.randint(0,255), random.randint(0,255), random.randint(0,255)),
		       'goalkeeper_team_2': (random.randint(0,255), random.randint(0,255), random.randint(0,255)),
		       'referee': (random.randint(0,255), random.randint(0,255), random.randint(0,255)),
		       'outsider': (random.randint(0,255), random.randint(0,255), random.randint(0,255))}

	#You will need to create these folders and upload the images inside colab's environment
	train_path = '../../DPVSA_dataset/train'
	val_path = '../../DPVSA_dataset/val'

	test_dataset(train_path, val_path, class_to_id, class_to_color)
