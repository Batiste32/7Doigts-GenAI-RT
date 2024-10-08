# 7Doigts-GenAI-RT <img src="/assets/Lab7_BlancRouge.png" alt="Logo for the 7 Fingers" height="50" style="vertical-align: middle;"/>

## Introduction

This project is designed for the [7 Doigts](https://7doigts.com/) / [7 Fingers](https://7fingers.com/) company.
It contains the code for generating live pictures using Stable Diffusion based on a camera input.

## Installation

### Download

Clone this repository using `git clone https://github.com/Batiste32/7Doigts-GenAI-RT/`.

### (Recommended) Creating a VENV

Creating a virtual environment (venv) is recommended in order to preserve your other installations.
This will allow you to have a separate Python installation with the correct packages.
This project was built using **Python 3.11**.

```bash
cd 7Doigts-GenAI-RT
python -m venv .venv
cd .venv/Scripts
activate
cd ../..
```

### Installing the libraries

This project was built using **Python 3.11** and **CUDA 1.12**.

```bash
python -m pip install -r requirements.txt
```

Then install the correct torch installation. If you use a different CUDA (check it in `Programs/NVIDIA GPU Computing Toolkit`), please refer to [Torch official site](https://pytorch.org/get-started/locally/) to get the correct version.

```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Using the project

### External setup

To make this project work, you will need a camera connected to your computer and **available**
(ie. not being used by another software).

```bash
python main.py
```

An input field should appear, asking for a model path. This is where the models will be (down)loaded. It should only ask this once as this path will be set as a system environment variable. Current path will also be written in the config file.

### Main Interface

A first Tkinter interface should open up with a selection of parameters.
Some presets have been defined and can be accessed by using the spinbox on the left.

<p align="center">
    <img src="/assets/screen-preset.PNG" alt="Location of the preset selection button" height="150"/>
</p>

### Generation Interface

A second interface should be created displaying the results.
If the debug mode is left unchecked, only the output will be shown, in fullscreen.

<table align="center">
<tr>
<td><img src="./assets/screen-debug.PNG" alt="Location of the debug checkbox" height="150"/></td>
<td><img src="./assets/screen-preview.PNG" alt="Preview of the output" height="150"/></td>
</tr>
</table>

If the debug mode is activated, the input will be shown with the output.

<p align="center">
    <img src="/assets/screen-preview-debug.PNG" alt="Preview in debug mode" height="150"/>
</p>
