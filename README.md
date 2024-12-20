﻿# 7Doigts-GenAI-RT <img src="/assets/Lab7_BlancRouge.png" alt="Logo for the 7 Fingers" height="50" />

## Table of Contents
- [7Doigts-GenAI-RT ](#7doigts-genai-rt-)
  - [Table of Contents](#table-of-contents)
  - [I - Introduction](#i---introduction)
  - [II - Installation](#ii---installation)
    - [1. Download](#1-download)
    - [2. (Recommended) Creating a VENV](#2-recommended-creating-a-venv)
    - [3. Installing the libraries](#3-installing-the-libraries)
  - [III - Using the project](#iii---using-the-project)
    - [External setup](#external-setup)
    - [Main Interface](#main-interface)
    - [Generation Interface](#generation-interface)
    - [The different effects](#the-different-effects)
  - [IV - Advanced uses](#iv---advanced-uses)
    - [Visualising the effects by WiFi (Flask server)](#visualising-the-effects-by-wifi-flask-server)
    - [Recording the output](#recording-the-output)
    - [Using a video as an input (WIP)](#using-a-video-as-an-input-wip)


## I - Introduction

This project is designed for the [7 Doigts](https://7doigts.com/) / [7 Fingers](https://7fingers.com/) company.
It contains the code for generating live pictures using Stable Diffusion based on a camera input.

## II - Installation

### 1. Download

Clone this repository using `git clone https://github.com/Batiste32/7Doigts-GenAI-RT/`.

### 2. (Recommended) Creating a VENV

Express way (check your Python and CUDA before) !
Run the file :

```bash
setup.bat
```

Creating a virtual environment (venv) is recommended in order to preserve your other installations.
This will allow you to have a separate Python installation with the correct packages.
This project was built for **Python 3.11**.

```bash
cd 7Doigts-GenAI-RT
python -m venv .venv
call .venv/Scripts/activate
```

### 3. Installing the libraries

This project was built for **Python 3.11** using **CUDA 1.12**.

```bash
pip install -r requirements.txt
```

Retrograde numpy to version 1 :

```bash
pip install numpy==1.26.4
```

Then install the correct torch installation. If you use a different CUDA (check it in `Programs/NVIDIA GPU Computing Toolkit`), please refer to [Torch official site](https://pytorch.org/get-started/locally/) to get the correct version.

```bash
pip install -U xformers --index-url https://download.pytorch.org/whl/cu121
```

## III - Using the project

### External setup

To make this project work, you will need a camera connected to your computer and **available**
(ie. not being used by another software).

You can launch the program with this file :

```bash
launcher.bat
```

Otherwise, activate your venv and run :

```bash
python.exe main.py
```

An input field should appear, asking for a model path. This is where the models will be (down)loaded. It should only ask this once as this path will be set as a system environment variable. Current path will also be written in the config file. To make sure it has taken effect, you should restart the program.

Afterward, a message will be displayed in the terminal asking wether or not the output should be broadcast to a server :

```
Should the output be broadcast to the server ? Y/N   ->
```

Answer `Y` or `N`.

**Warning :** Using the server might degrade performances or slow the program down.

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

Note : For this example, I use a Kinect camera with a body tracking algorithm (stored in the `.vl` file), which gives this black and white input, ideal for abstract transformation. If you use a standard webcam, it might be preferable to use more "realistic" prompts.

### The different effects

At the bottom of the main interface, there is a collection of buttons to start the different generation styles :

<p align="center">
    <img src="/assets/screen-effects.PNG" alt="Preview in debug mode" height="150"/>
</p>

Each of them requires different input fields. To explore each possibility, it is recommended to use the preset selection to see what kind of effect requires what inputs and what kind of output you can expect. If an input isn't defined, default values will be used.
**Warning :** The Perspective effect requires a white figure on a black background. Segmentate your input beforehand.


**List of Inputs per effect :**

| Effect Type       | Positive Prompt | Negative Prompt           | Adapter Image        | Background GIF        | Blending Value |
|-------------------|------------------|---------------------------|-----------------------|-----------------------|-----------------|
| **Standard effect**    | ✓                | ✓                         |                       |                       | ✓               |
| **Adapter effect**     | ✓                | ✓                         | ✓                     |                       | ✓               |
| **Background effect**  | ✓                | ✓                         | ✓                     | ✓                     | ✓               |
| **Perspective effect** | ✓                | ✓                         | _(optional)_          |                       |                 |

## IV - Advanced uses

### Visualising the effects by WiFi (Flask server)

You can use a Flask server to view your output in real time on any device by answering `Y` when booting the program.

You will then be able to connect to the server on the ip of the server, port `3142` and adding `/video_feed`
(ex : `123.456.789.0:3142/video_feed`)
Note : you will only be able to view the output, **not the input** enven in debug mode.

In the interface, 2 buttons will appear, related to the server :

<p align="center">
    <img src="/assets/screen-server.PNG" alt="Picture of the server buttons" height="150"/>
</p>

The first one will indicate the URL to enter to access the broadcast, clicking it will directly open the page in your browser.
The second one will kill the application and close the server correctly.
**This is the best way to close the app, otherwise you will have to kill the terminal / kernel.**

### Recording the output

On the left of the interface, you will find a checkbox and entry to record the output.

<p align="center">
    <img src="/assets/screen-record.PNG" alt="Picture of the recording parameters" height="150"/>
</p>

This will record every frame generated as a video file (.avi) with 3 FPS.
The file will be located in the project folder.

### Using a video as an input (WIP)

To input a video instead of a webcam, launch the video script instead of the main one :
```bash
python video.py
```

It will generate a similar interface with more options on the left, that will define the video properties :

<p align="center">
    <img src="/assets/screen-video.PNG" alt="Picture of the video parameters" height="150"/>
</p>

To upload the video, click on the large button at the top. The correct file path should appear below.
Selecting an effect like before should work, using the video instead.
You can adjust the video sampling to match the real speed.
(if the video appear too slow, increase the sampling rate).