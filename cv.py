#get_ipython().system('pip install torchsummary')
import os                       # for working with files
import numpy as np              # for numerical computationss
import pandas as pd             # for working with dataframes
import torch                    # Pytorch module 
import matplotlib.pyplot as plt # for plotting informations on graph and images using tensors
import torch.nn as nn           # for creating  neural networks
from torch.utils.data import DataLoader # for dataloaders 
from PIL import Image           # for checking images
import torch.nn.functional as F # for functions for calculating loss
import torchvision.transforms as transforms   # for transforming images into tensors 
from torchvision.utils import make_grid       # for data checking
from torchvision.datasets import ImageFolder  # for working with classes and images
from torchsummary import summary              # for getting the summary of our model
get_ipython().run_line_magic('matplotlib', 'inline')
def cropInfectDetect():
    data_dir = r"C:\Users\yudhveer singh\Downloads\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)"
    train_dir = data_dir + r"\train"
    valid_dir = data_dir + r"\valid"
    diseases = os.listdir(train_dir)
    plants = []
    NumberOfDiseases = 0
    for plant in diseases:
        if plant.split('___')[0] not in plants:
            plants.append(plant.split('___')[0])
        if plant.split('___')[1] != 'healthy':
            NumberOfDiseases += 1
    nums = {}
    for disease in diseases:
        nums[disease] = len(os.listdir(train_dir + '/' + disease))
    img_per_class = pd.DataFrame(nums.values(), index=nums.keys(), columns=["no. of images"])
    n_train = 0
    for value in nums.values():
        n_train += value
    train = ImageFolder(train_dir, transform=transforms.ToTensor())
    valid = ImageFolder(valid_dir, transform=transforms.ToTensor()) 
    img, label = train[0]
    random_seed = 7
    torch.manual_seed(random_seed)
    batch_size = 32
    train_dl = DataLoader(train, batch_size, shuffle=True, num_workers=2, pin_memory=True)
    valid_dl = DataLoader(valid, batch_size, num_workers=2, pin_memory=True)
    def get_default_device():
        return torch.device("cpu")
    def to_device(data, device):
        if isinstance(data, (list,tuple)):
            return [to_device(x, device) for x in data]
        return data.to("cpu", non_blocking=True)
    class DeviceDataLoader():
        def __init__(self, dl, device):
            self.dl = dl
            self.device = "cpu"
        def __iter__(self):
            for b in self.dl:
                yield to_device(b, self.device)        
        def __len__(self):
            return len(self.dl)
    device = get_default_device()
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)
    class SimpleResidualBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1)
            self.relu2 = nn.ReLU()
        def forward(self, x):
            out = self.conv1(x)
            out = self.relu1(out)
            out = self.conv2(out)
            return self.relu2(out) + x # ReLU can be applied before or after adding the input
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    class ImageClassificationBase(nn.Module):
        def training_step(self, batch):
            images, labels = batch
            out = self(images)                  # Generate predictions
            loss = F.cross_entropy(out, labels) # Calculate loss
            return loss
        def validation_step(self, batch):
            images, labels = batch
            out = self(images)                   # Generate prediction
            loss = F.cross_entropy(out, labels)  # Calculate loss
            acc = accuracy(out, labels)          # Calculate accuracy
            return {"val_loss": loss.detach(), "val_accuracy": acc}
        def validation_epoch_end(self, outputs):
            batch_losses = [x["val_loss"] for x in outputs]
            batch_accuracy = [x["val_accuracy"] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()       # Combine loss  
            epoch_accuracy = torch.stack(batch_accuracy).mean()
            return {"val_loss": epoch_loss, "val_accuracy": epoch_accuracy} # Combine accuracies
        def epoch_end(self, epoch, result):
            print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_accuracy']))
    def ConvBlock(in_channels, out_channels, pool=False):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                 nn.BatchNorm2d(out_channels),
                 nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(4))
        return nn.Sequential(*layers)
    class ResNet9(ImageClassificationBase):
        def __init__(self, in_channels, num_diseases):
            super().__init__()
            self.conv1 = ConvBlock(in_channels, 64)
            self.conv2 = ConvBlock(64, 128, pool=True) # out_dim : 128 x 64 x 64 
            self.res1 = nn.Sequential(ConvBlock(128, 128), ConvBlock(128, 128))
            self.conv3 = ConvBlock(128, 256, pool=True) # out_dim : 256 x 16 x 16
            self.conv4 = ConvBlock(256, 512, pool=True) # out_dim : 512 x 4 x 44
            self.res2 = nn.Sequential(ConvBlock(512, 512), ConvBlock(512, 512))
            self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                           nn.Flatten(),
                                           nn.Linear(512, num_diseases))
        def forward(self, xb): # xb is the loaded batch
            out = self.conv1(xb)
            out = self.conv2(out)
            out = self.res1(out) + out
            out = self.conv3(out)
            out = self.conv4(out)
            out = self.res2(out) + out
            out = self.classifier(out)
            return out        
    model = to_device(ResNet9(3, len(train.classes)), "cpu") 
    return model , to_device
    
# INPUT_SHAPE = (3, 256, 256)
# @torch.no_grad()
# def evaluate(model, val_loader):
#     model.eval()
#     outputs = [model.validation_step(batch) for batch in val_loader]
#     return model.validation_epoch_end(outputs)
# def get_lr(optimizer):
#     for param_group in optimizer.param_groups:
#         return param_group['lr']
# def fit_OneCycle(epochs, max_lr, model, train_loader, val_loader, weight_decay=0,
#                 grad_clip=None, opt_func=torch.optim.SGD):
#     torch.cpu.empty_cache()
#     history = []
#     optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
#     sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=len(train_loader))
#     for epoch in range(epochs):
#         model.train()
#         train_losses = []
#         lrs = []
#         for batch in train_loader:
#             loss = model.training_step(batch)
#             train_losses.append(loss)
#             loss.backward()
#             if grad_clip: 
#                 nn.utils.clip_grad_value_(model.parameters(), grad_clip)
#             optimizer.step()
#             optimizer.zero_grad()
#             lrs.append(get_lr(optimizer))
#             sched.step()
#         result = evaluate(model, val_loader)
#         result['train_loss'] = torch.stack(train_losses).mean().item()
#         result['lrs'] = lrs
#         model.epoch_end(epoch, result)
#         history.append(result)
#     return history
# epochs = 2
# max_lr = 0.01
# grad_clip = 0.1
# weight_decay = 1e-4
# opt_func = torch.optim.Adam
    
    
    #Testing the model
    
    
    #test_dir = r"C:\Users\yudhveer singh\Downloads\test"
    #test = ImageFolder(test_dir, transform=transforms.ToTensor())
    #test_images = sorted(os.listdir(test_dir + '/test')) # since images in test folder are in alphabetical order
def predict_image(img, model):
    def to_device(data, device):
        return data.to("cpu", non_blocking=True)
    img = Image.open(img)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Adjust the size if necessary for your model
        transforms.ToTensor(),          # Convert image to tensor
    ])
    img = transform(img)
    xb = to_device(img.unsqueeze(0), "cpu")
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return train.classes[preds[0].item()]
#img, label = test[0]
#plt.imshow(img.permute(1, 2, 0))
#print('Label:', test_images[0], ', Predicted:', predict_image(img, model))
#for i, (img, label) in enumerate(test):
    #print('Label:', test_images[i], ', Predicted:', predict_image(img, model))





#Code for saving the model
    
#PATH = './plant-disease-model.pth'  
#torch.save(model.state_dict(), PATH)
# PATH = './plant-disease-model-complete.pth'
# torch.save(model, PATH)
#get_ipython().system('pip install onnx onnxscipt')
#get_ipython().system('pip install --upgrade pip')
#get_ipython().system('pip install onnx')
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# class MyModel(nn.Module):
#     def __init__(self):
#         super(MyModel, self).__init__()
#         self.conv1 = nn.Conv2d(1, 6, 5)
#         self.conv2 = nn.Conv2d(6, 16, 5)
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#     def forward(self, x):
#         x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
#         x = F.max_pool2d(F.relu(self.conv2(x)), 2)
#         x = torch.flatten(x, 1)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
# model = MyModel()
# model.eval()
# dummy_input = torch.randn(1, 1, 32, 32)  # Adjust the dimensions if necessary
# torch.onnx.export(model, dummy_input, "model.onnx", verbose=True)
# print("Model has been exported to ONNX format.")