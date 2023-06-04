<div align="left">    

# Captionize it app     

</div>

## Description   
This project implements an image captioning system using PyTorch and Flask. It generates descriptive captions for input images using a deep learning encoder-decoder model and provides a web interface for captioning images.

## Features
- Trained model for image captioning and utilization of vast.ai GPU.
- Web interface for captioning images using Flask.

## Installation
First, clone and set up virtual environment

```bash
# clone project   
git clone https://github.com/quocanh34/captionize-it-app.git
cd captionize-it-app

# set up virtual env   
python -m venv captionize
source captionize/bin/activate  # for Unix/Linux
captionize\Scripts\activate  # for Windows
```   
Next, install dependencies.   
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

