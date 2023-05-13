import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

data_path = './assets/data/all_images.csv'

all_data = pd.read_csv(data_path)
all_data['image'] = all_data['image'] + '.jpg'
all_data = all_data[['image', 'label']]

#Dropping rows with 'Skip' or 'Not Sure' to avoid confusion for the model
all_data = all_data[all_data['label'] != 'Skip']
all_data = all_data[all_data['label'] != 'Not sure']

classes=list(all_data['label'].unique())
class_weights=class_weight.compute_class_weight(class_weight = "balanced", classes= classes, y=all_data['label'])

num_classes = len(classes)
class_ids = {}

#Replacing class names with integer ids
for i in range(num_classes):
    class_ids[classes[i]]=i
all_data = all_data.replace({'label' : class_ids}) 

#Splitting data into 90% train+val 10% test
train_val, test = train_test_split(all_data, test_size=0.10)
#Splitting the train data into 90% train 10% valid
train, val = train_test_split(train_val, test_size=0.10)

#Saving each dataframe into a separate csv
train.to_csv('./assets/data/train.csv', index = False, header=True)
test.to_csv('./assets/data/test.csv', index = False, header=True)
val.to_csv('./assets/data/val.csv', index = False, header=True)

#Saving class mappings and their crossponding class weight
with open('./assets/data/class_map.txt','w') as f:
	for cls in class_ids:
		cls_weight = class_weights[class_ids[cls]]
		f.write(f'{class_ids[cls]} {cls} {cls_weight}\n')
