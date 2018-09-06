#major functions for predict and train
import argparse
from PIL import Image
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from datetime import datetime

def get_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str, default='flowers', help='Directory of a dataset. Must include /dir/train/ ,             /dir/valid/ and /dir/test/.')
    parser.add_argument('--save_dir', type=str, default='saved_checkpoints/checkpoint.pth', help='Save directory.')
    parser.add_argument('--arch', type=str, default='vgg16', help='Choose between VGG16 and ?.')
    parser.add_argument('--hidden_units', type=str, default='512,124', help='Layer sizes. Seperate with comma (,).')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=0.005)
    parser.add_argument('--gpu',action='store_true')
    return parser.parse_args()

def transform_data(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    #  Define your transforms for the training, validation, and testing sets
    data_transforms = {"train": transforms.Compose([transforms.Resize(255), transforms.RandomVerticalFlip(),
                                                transforms.RandomHorizontalFlip(),
                                                transforms.CenterCrop(224), transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                       "test": transforms.Compose([transforms.Resize(255), transforms.CenterCrop(224), transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                       "valid": transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])}

    # Load the datasets with ImageFolder
    image_datasets = {"train": datasets.ImageFolder(train_dir, transform=data_transforms["train"]),
                      "test": datasets.ImageFolder(test_dir, transform=data_transforms["test"]),
                      "valid": datasets.ImageFolder(valid_dir, transform=data_transforms["valid"])}

    # Using the image datasets and the trainforms, define the dataloaders
    dataloaders = {"train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=4, shuffle=True),
                   "test": torch.utils.data.DataLoader(image_datasets["test"], batch_size=4, shuffle=True),
                   "valid": torch.utils.data.DataLoader(image_datasets["valid"], batch_size=4, shuffle=True)}

    class_to_idx = image_datasets["train"].class_to_idx # mapping classes -> network output

    return dataloaders, class_to_idx

def create_model(model_name, class_to_idx, classifier=None, criterion=None, optimizer=None,
                input_size=None, hidden_size=None, output_size=None, dropout=None,
                state_dict=None, lr=0.05):

    model = model_name
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

def train_model(model, train_dataloader, test_dataloader, optimizer, criterion, epochs = 3, print_sequence = 50, gpu = False):
    """TODO DocStr: Train model"""
    print("Training begins..")

    steps = 0

    if gpu == True:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # look if cuda is available
        if device == "cpu": print("GPU (CUDA) not available! Uses CPU instead.")
    else:
        device = "cpu"
    model.to(device)

    time_start = datetime.now() # set start-timestamp

    for e in range(epochs):
        model.train() # set model in training-mode
        running_loss = 0
        for images, labels in iter(train_dataloader): # loop through training-images
            steps += 1

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad() # reset gradient

            # Forward and backward passes
            with torch.set_grad_enabled(True): # make sure gradient decent is enabled
                outputs = model.forward(images) # feed forward batch of images
                loss = criterion(outputs, labels) # get loss
                loss.backward() # calculate gradient decent
                optimizer.step() # optimize weights

            running_loss += loss.item()

            if steps % print_sequence == 0:
                validation = validate(model, test_dataloader, criterion, device)
                duration = datetime.now() - time_start # calculate time since start of training
                print("{} - Iterations: {} - Epoch: {}/{} - Loss: {} - Accuracy: {}".format(duration, steps, e+1, epochs, running_loss /                       print_sequence, validation))
                running_loss = 0
    print("Done.")

def validate(model, val_dataloader, criterion, device):
    """Validates trained network"""

    model.to(device) # calculate with CUDA, if available
    model.eval() # set model to evaluation mode -> no gradient decent

    test_loss = 0
    accuracy = 0
    steps = 0
    with torch.no_grad(): # make sure that gradient decent is deactivated
        for images, labels in iter(val_dataloader): # loop through validation-pictures
            steps += 1

            images, labels = images.to(device), labels.to(device)

            output = model.forward(images) # feed batch of images through network
            test_loss += criterion(output, labels).item() # calculate total loss of test-routine

            ps = torch.exp(output) #propabilities
            equal = (ps.max(dim=1)[1] == labels.data) # check if pictures is classified correctly
            accuracy += equal.type(torch.FloatTensor).mean() # calculate total accuracy
        #test_loss = test_loss / steps
        accuracy = accuracy / steps # calculate mean accuracy
    return accuracy.item()

def save_model(model_name, model, optimizer, criterion, path=""):
    checkpoint = {"model_name": model_name,
                  "classifier": model.classifier,
                  "state_dict": model.state_dict(),
                  "class_to_idx": model.class_to_idx,
                  "optimizer": optimizer,
                  "criterion": criterion}

    torch.save(checkpoint, path)
    #print(checkpoint["state_dict"].keys())
    print("Saved.")

if __name__ == "__main__":
    print("Collection of functions to create, train and test models.")
