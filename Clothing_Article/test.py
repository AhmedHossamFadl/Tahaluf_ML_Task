import os
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
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

#Params
images_path = './assets/data/images/'
test_csv_path = './assets/data/test.csv'
model_path = './assets/models/final_weights.pth'

#Image Transforms just for resizing and converting to tensor	
transforms = transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]) 

test_set = ClothingArticlesDataset(
    images_dir=images_path,
    label_file=test_csv_path,
    transforms = transforms
    )  
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

model= torch.load(model_path).to(device)
model.eval()
correct = 0
for inputs,labels in test_loader:
	labels = labels.type(torch.LongTensor)
	inputs = inputs.to(device, non_blocking=True)
	labels = labels.to(device, non_blocking=True)
	outputs = model(inputs)
	_, preds = torch.max(outputs, 1)
	correct += torch.sum(preds == labels.data)
	
accuracy = (correct / test_loader.__len__() )*100
print('\nAccuracy: {:.2f}\n'.format(accuracy))
	
