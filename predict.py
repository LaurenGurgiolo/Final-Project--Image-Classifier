


import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from collections import OrderedDict
import numpy as np

from PIL import Image
import train

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_image(image):
    image = Image.open(image)
    width, height = image.size
    # resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    image = image.resize((256, int(256*(height/width))) if width < height else (int(256*(width/height)), 256))
    width, height = image.size
    # create 224x224 image
    #crop sizes of left,top,right,bottom
      
    left = (width - 224)/2
    top = (height - 224)/2
    right = (width + 224)/2
    bottom = (height + 224)/2
    
    image = image.crop((left, top, right, bottom))
    
    np_image = np.array(image)/255 # imshow() rewuires binary(0,1) so divided by 255
    
    # Seting the color to the first channel
    np_image = np_image.transpose(2, 0, 1)
    # Normalize 
      
    normalise_means = [0.485, 0.456, 0.406]
    normalise_std = [0.229, 0.224, 0.225]
    np_image = (np_image-normalise_means)/normalise_std
    
    
      
   #Turn into a torch tensor
    img = torch.from_numpy(np_image)
    processed_image = image.float()
     
    return processed_image


 

def imshow(image, ax=None, title=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    
    return ax
                                                           
def predict(image_path, model, topk=5):
    model = train.load_checkpoint('model_checkpoint.pth')

    
    model.load_state_dict
    model.to(device)
    
    image = process_image(image_path).type(torch.FloatTensor)
    image = image.unsqueeze(0)
  
    image = image.to('cuda')
    
    model.eval()
    with torch.no_grad():
        output = model(image.to(device).float())
    probability = F.softmax(output.data,dim=1)
    top_p, classes = probability.topk(topk)
    top_p, classes  = top_p.to(device), classes.to(device)
    print(classes)
    
    
    
        #convert from these indices to the actual class labels
    
    top_p = top_p.squeeze().tolist()
    idx_to_class={key:val for val,key in model.idx_to_class.items()}
   
    classes = classes[0].tolist()
    print(classes)
    class1 = [str(idx_to_class[each]) for each in classes] 

    return top_p,class1 # MUST RE probability.topk(topk)
    

    # TODO: Implement the code to predict the class    
                                                           
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    


import seaborn as sns
def plot_solution(image_path, model):
    # Set up plot
   
    # Set up title
    flower_num = image_path.split('/')[2]
   
    title_ = cat_to_name[flower_num]
    print(title_)
    # Plot flower
    img = process_image(image_path).type(torch.FloatTensor)
    imshow(img, ax, title = title_);
    # Make prediction
    top_p, classes = predict(image_path, model)
    name = []
    for x in classes:
        name.append(cat_to_name[x])
    print(name)
    
   
    
                                                
image = input('Please enter the flower image file directory')
top_p, classes = predict(image, model)
print(top_p)
print(classes)
plot_solution(image, model)                                                          
                                                          
