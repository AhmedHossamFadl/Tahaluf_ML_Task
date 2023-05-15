# Clothing Articles Classifer

[Introduction](#introduction)  
[Data preperation](#data-preperation)  
[Training method](#training-method)  
[Results](#results)  
[Answering Questinons](#answering-questinons)  

## Introduction

This repo aims to build a classifer for clothing articles, it will take an image and try to predict which clothing article (Shirt, Pants, Blouse, etc.) is in the image.

## Data preperation

The data used can be downloaded [here](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full).  

Data contains `5404` images of various clothing articles in true resolution and compressed resolution, compressed resolution is used here for faster training because the true resolution images go beyond 4k which is out of the scope of this repo.  

There is also a csv file containing the annotations of the images.

After downloading the data the following steps are taken to prepare the data for training:  
1. Remove the 'Skip' and 'Not Sure' classes from the dataset.  
   - 'Skip' are faulty or corrputed images.
   - 'Not Sure' are image of clothing articles that are undefined.
   - I remove both of those classes to avoid causing unnecessary confusion to the model during training.
2. Replace the class names with integer ids.
3. Calculate a class weight for each class to help combat the class imbalance that is present in the data 
   - Some classes had 600+ images while other class had less than 100.
4. Split data into Train+Validation/Test with a 9:1 ratio.
   
This results in `4182` for Train `466` for Validation and `518` for Test. 

## Training method

After preparing the data I train a classifer using the ResNet18 architecture with pre-trained weights, I use balanced cross entropy loss with the weights calculated during the data preperation step using sk-learn's `compute_class_weight` function, for the optimizer I use a standard SGD optimizer with `lr = 0.005` and `momentum = 0.9` the training goes for `50` epoch with a batch size of `32`.

I tried using ResNet18 from scratch as well as freezing the pre-trained weights and only training the final layer but training full network starting with the pre-trained weights ultimately proved to be best for this problem.

While training after each epoch I inference the model using the validation set and weights with the best accuracy for the validation set is saved as well as the final weights after all epochs.

## Results

The results with on these [Validation](./assets/data/val.csv) and [Test](./assets/data/test.csv) splits using this [Train](./assets/data/train.csv) 

|Subset|[Best Val Weights](./assets/models/best_val_weights.pth)|[Final Weights](./assets/models/final_weights.pth)|
|:-:|:-:|:-:|
|Validation|85.38%|84.73%|
|Test|86.85%|87.81%|

## Answering Questinons

1. Approach used VS other possible approachs.
   - The approach used here is single-label classifer for images using ResNet-18, other approachs for this problem could include.
     -  Multi-label classifer, that is able to identify more than 1 piece of clothing within the image.
     -  Object detection, to both identify and localize all clothing articles within a given image and would more accurately predict each class due to limiting the image to a specific area for each clothing article.
   - Both of these approachs would require specific data annotations which isn't easily accessible as well as more complex models. 
  
2. Receptive field
   - The overall receptive field of ResNet-18 is `435` this can either be increased by using a more depth version of ResNet (34 or 50) or decreased using a simple model like VGG-16 which has a receptive field of `212` .
  

3. Model FLOPS

   - The overall FLOPS of ResNet-18 is known to be ~1.8(G) FLOPS 
   - This is an estimation of each layer's number of FLOPS 
        - ResNet-18 is 18 layer architecture that consists of 17 convolutional layers and 1 fully connected layer at the end
        - FLOPS is calculated by the formula `K*K*C_in*C_out*H_out*W_out` where `K` is kernal size `C_in` is input channels `C_out` is output channels `H_out` is height of output channels and `W_out` is width of output channels
        - By tracing the ResNet-18 model and applying the above formula the following results are obtained (for Conv and fully connected layers, rest are ignored)
            ```python
                conv1 118013952.0
                conv1 115605504.0
                conv2 115605504.0
                conv1 115605504.0
                conv2 115605504.0
                conv1 57802752.0
                conv2 115605504.0
                conv1 115605504.0
                conv2 115605504.0
                conv1 115605504.0
                conv2 115605504.0
                conv1 115605504.0
                conv2 115605504.0
                conv1 57802752.0
                conv2 115605504.0
                conv1 115605504.0
                conv2 115605504.0
                fc 9216.0
            ```
        - From these calculations it is apparent that almost all layers have the same number of FLOPS with the first conv layer have the most
        - The first conv layer has kernal size of `7`, channels in of `3` channels out of `64` and output shape of `112*112` (Input shape is 224*224) all of which contribute to having the highest number of FLOPS.
        - This can be lowered by multiple modifications, lowering any of the above factors will decrease the total number of FLOPS but will affect both the receptive field and the accuracy of the model.
        - This can be done for all layers but will also affect the overall performance of the model.
