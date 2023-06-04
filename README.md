<div align="left">
    <h1>Captionize it app</h1>
    <img src="https://github.com/quocanh34/captionize-it-app/assets/79373948/effa9c2c-8810-4976-99c1-a6c959862edf" width="500" height="250">
</div>

## Table Of Contents
-  [Description](#description)
-  [Features](#features)
-  [Details blog](#for-more-details-of-model-training-architectures-and-webapp)
-  [Requirements](#requirements)
-  [Installation](#installation)
-  [Citation](#citation)

## Description   
This project implements an image captioning system using PyTorch and Flask. It generates descriptive captions for input images using a deep learning encoder-decoder model and provides a web interface for captioning images.

## Features
- Trained model for image captioning and utilization of vast.ai GPU.
- Web interface for captioning images using Flask.
- Adding cross attention for models (not completed)
- Docker package (not completed)
- Cloud server deployment (not completed)

## For more details of model training architectures and webapp
- Reading my blog at ...

## Requirements
- Flask, nltk, numpy, tqdm, python-dotenv, torch, torchvision.

## Installation
First, clone and set up virtual environment

```bash
# clone project   
git clone https://github.com/quocanh34/captionize-it-app.git
cd captionize-it-app

# set up virtual env   
python -m venv captionize

# activate the env
source captionize/bin/activate  # for Unix/Linux
captionize\Scripts\activate  # for Windows
```   
Second, install dependencies.   

```bash
pip install -r requirements.txt
```  
Next, download the trained model

- Go to the link: https://drive.google.com/file/d/142-ZaLaKSUNYLg82mUYwtxJ6X_AGQtAI/view?usp=drive_link

- Download and move it to **src/checkpoint/**

Now run the app
```bash
# run flask 
python app.py
```   

### Citation   
```
@{article{Anh Pham},
  title={Captionize it app},
  author={Anh Pham},
  year={2023}
}
```   

