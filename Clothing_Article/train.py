import os
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader

class ClothingArticlesDataset(Dataset):
    def __init__(self, images_dir, label_file, transforms=None):
        self.images_dir = images_dir
        self.labels = pd.read_csv(label_file)
        self.transforms = transforms

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image_name = self.labels.iloc[index, 0]
        image = Image.open(os.path.join(self.images_dir, image_name)).convert("RGB")
        label = torch.tensor(float(self.labels.iloc[index, 1]))

        if self.transforms is not None:
            image = self.transforms(image)

        return (image, label)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Paths
images_path = './assets/data/images'
train_csv_path = './assets/data/train.csv'
val_csv_path = './assets/data/val.csv'

#Training Params
batch_size = 32
num_epochs = 50
learning_rate = 0.005

#Reading the class_map file which contains class mapping for the original class names and class weights for balanced training
class_map = {}
class_weights = []
with open('./assets/data/class_map.txt','r') as f:
	for line in f:
		cls_id,cls,cls_weight = line.split()
		class_map[int(cls_id)] = cls
		class_weights.append(float(cls_weight))

#Image Transforms just for resizing and converting to tensor	
transforms = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]) 

train_set = ClothingArticlesDataset(
    images_dir=images_path,
    label_file=train_csv_path,
    transforms = transforms
    )  
    
val_set = ClothingArticlesDataset(
    images_dir=images_path,
    label_file=val_csv_path,
    transforms = transforms
    )   
    
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=2)

#Creating model using resnet18 
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs,len(class_map))
model = model.to(device)

#Balanced class CrossEntropyLoss
criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights,dtype=torch.float).to(device),reduction='mean')
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

best_val_acc = 0
for epoch in range(num_epochs):
	print('Epoch {}/{}'.format(epoch + 1, num_epochs))
	print('-' * 10)

	model.train()
	running_loss = 0.0

	for batch, data in enumerate(train_loader, 0):
		inputs, labels = data
		labels = labels.type(torch.LongTensor)
		inputs = inputs.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)
		
		optimizer.zero_grad()
		outputs = model(inputs)

		loss = criterion(outputs, labels)
		loss.backward()
		
		with torch.no_grad():
			optimizer.step()
			optimizer.zero_grad()
			
		batch_loss = loss.item()
		running_loss += batch_loss
		if batch % 20 == 19:
			print('Batch:{} Batch_Loss: {:.4f}'.format(batch + 1, batch_loss))

	epoch_loss = running_loss / batch + 1
	print('\nAverage_Epoch_Loss: {:.4f}'.format(epoch_loss))
	
	model.eval()
	correct = 0
	for inputs,labels in val_loader:
		labels = labels.type(torch.LongTensor)
		inputs = inputs.to(device, non_blocking=True)
		labels = labels.to(device, non_blocking=True)
		outputs = model(inputs)
		_, preds = torch.max(outputs, 1)
		correct += torch.sum(preds == labels.data)
		
	accuracy = (correct / val_loader.__len__() )*100
	print('Validation_Accuracy: {:.2f}\n'.format(accuracy))
	
	#Saving the best model on validation set
	if accuracy > best_val_acc:
		torch.save(model, './assets/models/best_val_weights.pth')
		best_val_acc = accuracy

torch.save(model, './assets/models/final_weights.pth')
        
