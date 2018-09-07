# AI Programming with Python Project (pytorch)

## Image Classifier Notebook
1. Import data using 'torchvision.ImageFolder'.
2. Create model using a pretrained CNN and replace its classifer by a new one, which fits the input data.
3. Train the model and validate its accuracy during training with validate-dataset.
4. Test model with test-dataset.
5. Save model.
6. Load model.
7. Preprocess input image to fit the model
8. Predict class.

## train.py
Create a model from a pretrained CNN's (vgg16, alexnet or densenet161) for a dataset and save it.

```
usage: train.py [-h] [--save_dir SAVE_DIR] [--arch ARCH]
                [--hidden_units HIDDEN_UNITS] [--epochs EPOCHS]
                [--learning_rate LEARNING_RATE] [--gpu]
                data_dir

positional arguments:
  data_dir              Directory of a dataset. Must include /dir/train/,
                        /dir/valid/ and /dir/test/.

optional arguments:
  -h, --help            show this help message and exit
  --save_dir SAVE_DIR   Save directory.
  --arch ARCH           Choose between "vgg16" and "alexnet" and "densenet".
  --hidden_units HIDDEN_UNITS
                        Hidden layer sizes. Seperate with comma (,).
  --epochs EPOCHS       Number of epochs.
  --learning_rate LEARNING_RATE
  --gpu                 Append if CUDA is available.
```

Example:

```
python train.py flowers --gpu --save_dir flowers_checkpoints/check_alexnet_1024_512_e3.pth --epochs 3 --hidden_units 1024,512 --arch alexnet
```

## predict.py
Load a checkpoint and build its model. Predict class of the image.

```
usage: predict.py [-h] [--category_names CATEGORY_NAMES] [--topk TOPK] [--gpu]
                  input checkpoint

positional arguments:
  input                 Input image.
  checkpoint            Saved checkpoint.

optional arguments:
  -h, --help            show this help message and exit
  --category_names CATEGORY_NAMES
                        Include the mapping of classes to the output of the
                        network.
  --topk TOPK           Number of returned probabilities.
  --gpu                 Append if CUDA is available
```

Example:

```
python predict.py image_path checkpoint.pth --gpu --category_names category_to_name.json 
```
