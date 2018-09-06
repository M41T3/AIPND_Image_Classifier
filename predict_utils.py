import argparse
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict

def get_cmd_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('input', type=str)
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--category_names', type=str, default=None)
    parser.add_argument('--topk', type=int, default=5)
    parser.add_argument('--gpu',action='store_true')

    return parser.parse_args()

def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # look if cuda is available

    image = process_image(Image.open(image_path))
    image_tensor = torch.from_numpy(image)
    image_tensor.type(torch.DoubleTensor)
    image_tensor = image_tensor.to(device)
    model.to(device)

    model.eval()
    with torch.no_grad():
        output = model.forward(image_tensor.unsqueeze(0))

    topk_prob, topk_idxs = torch.topk(output, topk) # load top propabilities and top labels (output numbers of last layer)
    topk_prob = topk_prob.exp() # exponetial to increase failure -> also positive
    topk_prob_array = topk_prob.data.cpu().numpy()[0] # save data as array

    idx_to_class = {idx: cls for cls, idx in model.class_to_idx.items()} #invert class_to_idx -> idx to class
                                                                                       #idx: output last layer, class: classes
    topk_idxs_data = topk_idxs.data.cpu().numpy() # top indexes as array
    top_idxs_list = topk_idxs_data[0].tolist()  # convert to list

    topk_classes = [idx_to_class[x] for x in top_idxs_list] # get top classes from top labels (indexes)

    return topk_prob_array, topk_classes # return top 5 probabilities and top 5 classes

def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # TODO: Process a PIL image for use in a PyTorch model
    x,y = image.size

    if x < y:    #get x,y for thumbnail
        x_thumb = 256
        y_thumb = int(x_thumb * y/x)
    elif x > y:
        y_thumb = 256
        x_thumb = int(y_thumb * x/y)
    else:
        x_thumb = 256
        y_thumb = 256

    image.thumbnail((x_thumb,y_thumb), Image.ANTIALIAS)

    x1 = x_thumb / 2 - 224 / 2 # center crop image 224x224
    x2 = x1 + 224
    y1 = y_thumb / 2 - 224 / 2
    y2 = y1 + 224

    image = image.crop((x1, y1, x2, y2))
    np_image = np.array(image,dtype=np.float32) # convert to np array

    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    np_image = ((np_image / 255) - means) / std # normalize image
    #print(np_image.dtype)

    return np_image.transpose(2,0,1).astype(np.float32) # transpose ndarray

def load_model(filepath, device):
    #checkpoint = torch.load(filepath, map_location=device)
    checkpoint = torch.load(filepath, map_location=device)
    return create_model(checkpoint["model_name"],checkpoint["class_to_idx"],classifier=checkpoint["classifier"], optimizer=checkpoint["optimizer"],
                 criterion=checkpoint["criterion"], state_dict=checkpoint["state_dict"] )

def create_model(model_name, class_to_idx, classifier=None, criterion=None, optimizer=None,
                input_size=None, hidden_size=None, output_size=None, dropout=None,
                state_dict=None, lr=0.05):

    if model_name == "vgg16":
        model = models.vgg16(pretrained=True)
    elif model_name == "densenet":
        model = models.densenet161(pretrained=True)
    elif model_name == "alexnet":
        model = models.alexnet(pretrained=True)

    # build classifier at the end of the pretrained
    for param in model.parameters():
        param.requieres_grad = False # Disable optimizer on pretrained network

    #if classifier has to be created by layer sizes (when you don't load checkpoint)
    if (classifier == None) and ((input_size != None) and (hidden_size != None) and (output_size != None)):
        model.classifier = create_classifier(input_size, hidden_size, output_size)  # replace old classifier sequence with our new one
    #if classifier is available (ex. from checkpoint)
    elif not classifier == None:
        model.classifier = classifier
    else:
        print("Error!")

    if not state_dict == None:                               #load state_dict, if available
         model.load_state_dict(state_dict)
    if criterion == None:
        criterion = nn.CrossEntropyLoss()
    if optimizer == None:
        optimizer = optim.SGD(model.classifier.parameters(), lr) # just optimize OUR classifier with learnrate = 0.001

    model.class_to_idx = class_to_idx

    return model, criterion, optimizer

def create_classifier(input_size, hidden_size, output_size, dropout = 0.5):
    """Creates classifer for the model."""

    layer_size = zip(hidden_size[:-1], hidden_size[1:])

    sequence = []

    sequence.append(["fc_in",nn.Linear(input_size,hidden_size[0])])
    sequence.append(["relu_in",nn.ReLU(True)])
    sequence.append(["drop_in",nn.Dropout(dropout)])

    for i, (l1, l2) in enumerate(layer_size):
        sequence.append(["fc"+str(i+1),nn.Linear(l1,l2)])
        sequence.append(["relu"+str(i+1),nn.ReLU(True)])
        sequence.append(["drop"+str(i+1),nn.Dropout(dropout)])

    sequence.append(["fc_out",nn.Linear(hidden_size[-1],output_size)])
    sequence.append(["soft_out",nn.LogSoftmax(dim=1)])

    return nn.Sequential(OrderedDict(sequence))
