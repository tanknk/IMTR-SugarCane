import pandas as pd 
import numpy as np 
from Net import Net

# for reading and displaying images
from skimage.io import imread
import matplotlib.pyplot as plt

# for creating validation set
from sklearn.model_selection import train_test_split

# for evaluating the model
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# PyTorch libraries and modules
import torch
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss , Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
 
""" Model Loader """
 # Load the Trained Model from model04.pt
model = torch.load('C:/model04.pt')

""" Data Loader """
# Load Data File
test = pd.read_csv('C:/test/test.csv')
outXls = pd.read_csv('C:/out.csv')
# loading test images
test_img = []
for img_name in tqdm(test['id']):
    # defining the image path
    image_path = '/test/' + str(img_name) + '.png'
    # reading the image
    img = imread(image_path, as_gray=True)
    # normalizing the pixel values
    img /= 255.0
    # converting the type of pixel to float 32
    img = img.astype('float32')
    # appending the image into the list
    test_img.append(img)
# converting the list to numpy array
test_x = np.array(test_img)
# converting training images into torch format
test_x = test_x.reshape
(10000, 1, 28, 28)
test_x = torch.from_numpy
(test_x).to(torch.float32)

""" Executing Model with Testing Image """
# generating predictions for test set
with torch.no_grad():
    #output = model(test_x.cuda())
    output = model(test_x)

softmax = torch.exp(output).cpu()
prob = list(softmax.numpy())
predictions = np.argmax(prob, axis=1)

# replacing the label with prediction
outXls['label'] = predictions 
outXls.head()

# saving the file
outXls.to_csv('C:/out.csv', index=False)
 