


import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from collections import OrderedDict
import numpy as np

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform = train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform = valid_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform = test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloaders = torch.utils.data.DataLoader(train_datasets, batch_size = 16, shuffle = True)
validloaders = torch.utils.data.DataLoader(valid_datasets, batch_size = 16)
testloaders = torch.utils.data.DataLoader(test_datasets, batch_size = 16)
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.densenet121(pretrained = True)

for param in model.parameters():
    param.requiers_grad = False
  

model.classifier = nn.Sequential(nn.Linear(1024, 500),
                                        nn.ReLU(),
                                        nn.Linear(500, 250),
                                        nn.ReLU(),
                                        nn.Linear(250, 102),
                                        nn.LogSoftmax(dim=1))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=.003)
model.to(device);

epochs = 2
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloaders:
        steps += 1
        inputs, labels= inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloaders:
                    inputs, labels= inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()

                    ps  = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            print(f"Epoch {epoch+1}/{epochs}.. "
                f"Train loss: {running_loss/print_every:.3f}.. "
                 f"Test loss: {test_loss/len(testloaders):.3f}.. "
                 f"Test accuracy: {accuracy/len(testloaders):.3f}")
            running_loss = 0
            model.train()            

            
# TODO: Do validation on the test set
epochs = 2
epoch = 1
test_loss = 0
accuracy = 0
model.to(device)
model.eval()
with torch.no_grad():
    for inputs, labels in validloaders:
        inputs, labels= inputs.to(device), labels.to(device)
        logps = model.forward(inputs)
        batch_loss = criterion(logps, labels)
        test_loss += batch_loss.item()
                    
        ps  = torch.exp(logps)
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    print(f"Epoch {epoch+1}/{epochs}.. "
        f"Valid loss: {test_loss/len(testloaders):.3f}.. "
        f"Test accuracy: {accuracy/len(testloaders):.3f}")
    running_loss = 0
    model.train()            
    
    
model.class_to_idx = train_datasets.class_to_idx

checkpoint = {'architecture': 'densenet121',
              'input_size': 1024,
              'output_size': 102,
              'hidden_layer': 500,
              'state_dict': model.state_dict(),
              'optimizer':optimizer.state_dict(),
              'idx_to_class': model.class_to_idx,
              'epochs': epochs,
              'criterion': criterion,
             }
torch.save({'structure' :1024, 'hidden_layer1':500, 'state_dict':model.state_dict(), 'idx_to_class':model.class_to_idx}, 'model_checkpoint.pth')

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(1024, 500),
                    nn.ReLU(),
                    nn.Linear(500, 250),
                    nn.ReLU(),
                    nn.Linear(250, 102),
                    nn.LogSoftmax(dim=1))
                         
    model.load_state_dict(checkpoint['state_dict'])
    model.idx_to_class = checkpoint['idx_to_class']
    
    return model
    
model = load_checkpoint('model_checkpoint.pth')

model    
def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    model = models.densenet121(pretrained=True)
    model.classifier = nn.Sequential(nn.Linear(1024, 500),
                    nn.ReLU(),
                    nn.Linear(500, 250),
                    nn.ReLU(),
                    nn.Linear(250, 102),
                    nn.LogSoftmax(dim=1))
                         
    model.load_state_dict(checkpoint['state_dict'])
    model.idx_to_class = checkpoint['idx_to_class']
    
    return model
    
model = load_checkpoint('model_checkpoint.pth')

print(model)    


