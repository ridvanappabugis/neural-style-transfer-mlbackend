# Neural Style Transfer

Implementation based on the paper "A Neural Algorithm of Artistic Style" by Leon A. Gatys, Alexander S. Ecker, Matthias Bethge. https://arxiv.org/pdf/1508.06576.pdf

This repository is an implementation of the above mentioned paper, using Python's Keras/Tensorflow API.

## Intro

The purpose of this project is to serve a model that is able to take two images as an input (style, content), combine them, producing a third image. 
The resulting image will have the contents of the original content image, represented in the style of the the original style image.

The process done is by using the VGG convolutional network, as to extract the tensor matrices for style and content. 
Resulting image's tensor matrices are a middle point between these two.

This projects imlements a modular Model that is able to be served in a rest api. The behaviour of the Model is to produce a stylized image in one Model call, so training was implemented in the actual call. As tensorflow does not have any way to do this graph-wise, subclassed Models and object oriented apporach was used to produce a Model that trains itself while calling it.

A graph of the model logic can be seen below:
![graph drawio](https://user-images.githubusercontent.com/37189321/145400226-e918e23e-6d67-419a-bc5a-26474abe9f57.png)


## How to run

This project consists of two parts - <b>API Server</b> that is able to take requests, and a <b>Jupyter Notebook</b> that is able to run the model as well.

## Requirements
- Python ^3.6
- package requirements listed under requirements.txt
- CUDA Support for GPU based processing, otherwise it will run on CPU

## API Server

Included is a simple Flask rest server that will bootstrap the model for running. 
Exposes a <b>POST</b> endpoint: ```/stylize```.

Request format:
```
{
    "content_img": "byte64_img",
    "style_img": "byte64_img",
    "epochs": 5,
    "steps_per_epoch": 25
}
```

Response format:
```
{
    "result": "byte64_img"
}
```

## Jupyter Notebook

Anothery way is to run the model via the Jupyter Notebook. Provided are 2 Notebooks - ModelRunDefinition.ipynb & QuickRunModel.ipynb.

Both are available under ```/notebooks```

## References

- "A Neural Algorithm of Artistic Style" by Leon A. Gatys, Alexander S. Ecker
- https://www.tensorflow.org/tutorials/generative/style_transfer
