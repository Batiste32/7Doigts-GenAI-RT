# 7Doigts-GenAI-RT
## Introduction
This project is designed for the 7 Doigts / 7 Fingers company.
It contains the code for generating live pictures using Stable Diffusion based on a camera input.
## Installation
### Download
Clone this repository using `git clone https://github.com/Batiste32/7Doigts-GenAI-RT/`.
### (Recommended) Create a venv
Creating a virtual environment is recommended in order to preserve your other installations.
This will allow you to have a separate Python installation with the correct packages.
This project was built using **Python 3.11**.
```
cd 7Doigts-GenAI-RT
python -m venv .venv
cd .venv/Scripts
activate
cd ../..
```
### Installing the libraries
This project was built using **Python 3.11** and **CUDA 1.12**.
```
python -m pip install -r requirements.txt
```
Then install the correct torch installation. If you use a different CUDA (check it in `Programs/NVIDIA GPU Computing Toolkit`), please refer to [Torch official site](https://pytorch.org/get-started/locally/) to get the correct version.
```
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
## Using the project
### External setup
To make this project work, you will need a camera connected to your computer and **available**
(ie. not being used by another software).
```
python main.py
```
An input field should appear, asking for a model path. This is where the models will be (down)loaded. It should only ask this once as this path will be set as a system environment variable. Current path will also be written in the config file.
### Main Interface
A first Tkinter interface should open up with a selection of parameters.
Some presets have been defined and can be accessed by using the spinbox on the left. 
![Location of the preset selection button](/assets/screen-preset.PNG)