import cv2
from PIL import Image
import numpy as np
import keyboard
from diffusers import AutoPipelineForImage2Image, EulerAncestralDiscreteScheduler, StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch
import os
import threading
from transformers import CLIPVisionModelWithProjection
import subprocess
import json

# GUI for .py only
import tkinter as tk
from tkinter import ttk, Tk, font, colorchooser
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, GifImagePlugin
import time

# To send the frames as a stream
from flask import Flask, Response
import webbrowser
import socket
import signal
import requests

# For type hint
from typing import Generator

open_file = open("config.txt", "r")
model_path = open_file.readline()
open_file.close()
if model_path == "" or model_path == "\n" : # Check if path defined, if not, ask the user
    model_path = input("Write path to where models should be downloaded : ")
    open_file = open("config.txt", "w")
    open_file.writelines(model_path)
    open_file.close()
print(model_path)

os.environ['HF_HOME'] = model_path
os.environ['TRANSFORMERS_CACHE'] = model_path    # path to downloaded models
subprocess.run('setx HF_HOME '+model_path,shell=True,check=False)

torch.backends.cuda.matmul.allow_tf32 = True       # use 32 precision floats

GifImagePlugin.LOADING_STRATEGY = GifImagePlugin.LoadingStrategy.RGB_ALWAYS # Ensures that GIF frames are always loaded in RGB format, even if the GIF has a palette or other color mode, to simplify image processing.

class WebcamCapture:
    """
    A class to handle capturing frames from a webcam using OpenCV.
    
    Attributes:
        cap (cv2.VideoCapture): The video capture object for accessing the webcam.
    
    Methods:
        __init__(cam_index): Initializes the webcam with the specified camera index.
        release(): Releases the webcam resource.
    """
    def __init__(self, cam_index:int=0):
        """
        Initializes the webcam using the specified camera index (default is 0, the first webcam).
        
        Args:
            cam_index (int): The index of the camera to be used. Default is 0.
        
        Raises:
            Exception: If the video device cannot be opened.
        """
        # Initialize the webcam once
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")
    
    def capture_image(self) -> Image.Image:
        """
        Captures the current frame from the camera and returns it as a PIL Image.
        
        Returns:
            Image.Image: The current camera frame as a PIL Image.
        """
        # Capture a single frame
        ret, frame = self.cap.read()
        if not ret:
            raise Exception("Failed to capture image")

        # Convert the captured frame (BGR) to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        cropped_frame = frame_rgb[0:480,80:560]
        # Convert NumPy array to PIL Image
        image = Image.fromarray(cropped_frame).resize((image_size.get(),image_size.get()))
        
        return image

    def release(self) -> None:
        """
        Releases the camera resource, closing any open connections.
        """
        # Release the webcam when done
        self.cap.release()

class SDPipeline:
    """
    A class to handle image transformation using the Stable Diffusion pipeline.

    Attributes:
        model_name (str): The name of the model to be used.
        seed (int): The random seed for generating images.
        pipe (AutoPipelineForImage2Image): The Stable Diffusion pipeline for image-to-image generation.
        generator (torch.Generator): The random number generator for deterministic results.
        adapter_image (Optional[Image]): The adapter image for IP-Adapter processing (default is None).
        image_encoder (Optional[CLIPVisionModelWithProjection]): The image encoder for IP-Adapter (default is None).
    
    Methods:
        __init__(model_name: str, seed: int) -> None: 
            Initializes the Stable Diffusion pipeline with the specified model name and seed.
        
        transform_image(prompt: str, input_image: str, negative_prompt: str = "", num_steps: int = 2, cfg: float = 1.0, strength: float = 0.5, return_type: str = "image") -> Image.Image: 
            Transforms an input image using the Stable Diffusion pipeline.
        
        accelerate_pipe() -> None:
            Optimizes the pipeline for faster execution on GPU.
        
        add_ip_adapter(image: Image.Image) -> None:
            Adds an IP-Adapter for improved image transformation.
    """

    def __init__(self, model_name: str = "stabilityai/sdxl-turbo", seed: int = 314159):
        """
        Initializes the Stable Diffusion pipeline with the specified model name and seed.

        Args:
            model_name (str): The name of the model to be used (default is "stabilityai/sdxl-turbo").
            seed (int): The random seed for generating images (default is 314159).
        """
        # Store pipe info
        self.model_name = model_name
        # Initialize the SD pipeline
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            variant="fp16", 
            use_safetensors=True
        ).to("cuda")
        # Set scheduler
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipe.scheduler.config, 
            use_safetensors=True, timestep_spacing='trailing'
        )
        # Initialize generator
        self.seed = seed
        self.generator = torch.Generator(device="cuda").manual_seed(seed)
        # Initialize adapter fields to None
        self.adapter_image = None
        self.image_encoder = None

    def transform_image(self, prompt: str, input_image: str, negative_prompt: str = "", num_steps: int = 2, cfg: float = 1.0, strength: float = 0.5, return_type: str = "image") -> Image.Image:
        """
        Transforms an input image using the Stable Diffusion pipeline.

        Args:
            prompt (str): The prompt to guide the image transformation.
            input_image (str): The path to the input image.
            negative_prompt (str): An optional negative prompt (default is "").
            num_steps (int): The number of inference steps (default is 2).
            cfg (float): The classifier-free guidance scale (default is 1.0).
            strength (float): The strength of the transformation (default is 0.5).
            return_type (str): The type of output to return (default is "image").

        Returns:
            Image.Image: The transformed image as a PIL Image.
        """
        input_image = load_image(input_image)
        if self.adapter_image is None:
            output = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                image=input_image,
                num_inference_steps=num_steps,
                strength=strength,
                guidance_scale=cfg,
                generator=self.generator
            )
        else:
            output = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                image=input_image,
                ip_adapter_image=self.adapter_image,
                num_inference_steps=num_steps,
                strength=strength,
                guidance_scale=cfg,
                generator=self.generator
            )
        # Access the generated image correctly
        self.output_image = output.images[0]
        return self.output_image

    def accelerate_pipe(self) -> None:
        """
        Optimizes the pipeline for faster execution on GPU.

        This method configures various settings to improve memory efficiency
        and performance during image generation.
        """
        # self.pipe.unet.set_attn_processor(AttnProcessor2_0())
        # self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        self.pipe.to("cuda")
        self.pipe.upcast_vae()
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()
        self.pipe.enable_xformers_memory_efficient_attention()
        # self.pipe.enable_model_cpu_offload()

    def add_ip_adapter(self, image: Image.Image) -> None:
        """
        Adds an IP-Adapter for improved image transformation.

        Args:
            image (Image.Image): The image to be used with the IP-Adapter.
        """
        self.adapter_image = image
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16, 
            variant="fp16", 
            image_encoder=self.image_encoder, 
            use_safetensors=True
        ).to("cuda")
        self.accelerate_pipe()
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter", 
            subfolder="sdxl_models", 
            weight_name="ip-adapter-plus_sdxl_vit-h.bin"
        )
        self.pipe.set_ip_adapter_scale(0.7)
        
class CNPipeline:
    """
    A class to handle image transformation using the ControlNet with Stable Diffusion pipeline.

    Attributes:
        model_name (str): The name of the Stable Diffusion model to be used.
        controlnet (ControlNetModel): The ControlNet model for conditioning image transformations.
        pipe (StableDiffusionXLControlNetPipeline): The pipeline for Stable Diffusion with ControlNet.
        generator (torch.Generator): The random number generator for deterministic results.
        adapter_image (Optional[Image]): The adapter image for IP-Adapter processing (default is None).
        image_encoder (Optional[CLIPVisionModelWithProjection]): The image encoder for IP-Adapter (default is None).
        control_image (Optional[Image]): The image used for ControlNet conditioning (default is None).
        controlnet_conditioning_scale (float): The scale for ControlNet conditioning (default is 1.0).
    
    Methods:
        __init__(model_name: str, control_net: str, seed: int) -> None:
            Initializes the ControlNet pipeline with the specified model name and seed.
        
        transform_image(prompt: str, negative_prompt: str, num_steps: int, cfg: float, strength: float, return_type: str) -> Image.Image:
            Transforms an input image using the ControlNet pipeline.

        accelerate_pipe() -> None:
            Optimizes the pipeline for faster execution on GPU.

        add_ip_adapter(image: Image.Image) -> None:
            Adds an IP-Adapter for improved image transformation.

        load_control_image(image: Image.Image, conditioning: float) -> None:
            Loads the control image for conditioning.
    """

    def __init__(self, model_name: str = "stabilityai/sdxl-turbo", control_net: str = "diffusers/controlnet-depth-sdxl-1.0", seed: int = 314159):
        """
        Initializes the ControlNet pipeline with the specified model name and seed.

        Args:
            model_name (str): The name of the Stable Diffusion model to be used (default is "stabilityai/sdxl-turbo").
            control_net (str): The name of the ControlNet model to be used (default is "diffusers/controlnet-depth-sdxl-1.0").
            seed (int): The random seed for generating images (default is 314159).
        """
        # Store pipe info
        self.model_name = model_name
        # Initialize the ControlNet
        self.controlnet = ControlNetModel.from_pretrained(control_net, torch_dtype=torch.float16, use_safetensors=True)
        # Initialize the SDXLPipe
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(self.model_name, controlnet=self.controlnet, torch_dtype=torch.float16, use_safetensors=True)
        # Set scheduler
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config,timestep_spacing='trailing')
        # Initialize generator
        self.seed = seed
        self.generator = torch.Generator(device="cuda").manual_seed(seed)
        # Initialize adapter fields to None
        self.adapter_image = None
        self.image_encoder = None
        self.control_image = None
        self.controlnet_conditioning_scale = 1.0

    def transform_image(self, prompt: str, negative_prompt: str = "", num_steps: int = 1, cfg: float = 1.0, strength: float = 1, return_type: str = "image") -> Image.Image:
        """
        Transforms an input image using the ControlNet pipeline.

        Args:
            prompt (str): The prompt to guide the image transformation.
            negative_prompt (str): An optional negative prompt (default is "").
            num_steps (int): The number of inference steps (default is 1).
            cfg (float): The classifier-free guidance scale (default is 1.0).
            strength (float): The strength of the transformation (default is 1).
            return_type (str): The type of output to return (default is "image").

        Returns:
            Image.Image: The transformed image as a PIL Image.
        """
        if self.adapter_image is None:
            output = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                image=self.control_image,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                num_inference_steps=num_steps,
                strength=strength,
                guidance_scale=cfg,
                generator=self.generator
            )
        else:
            output = self.pipe(
                prompt,
                negative_prompt=negative_prompt,
                image=self.control_image,
                controlnet_conditioning_scale=self.controlnet_conditioning_scale,
                ip_adapter_image=self.adapter_image,
                num_inference_steps=num_steps,
                strength=strength,
                guidance_scale=cfg,
                generator=self.generator
            )
        # Access the generated image correctly
        self.output_image = output.images[0]
        return self.output_image

    def accelerate_pipe(self) -> None:
        """
        Optimizes the pipeline for faster execution on GPU.

        This method configures various settings to improve memory efficiency
        and performance during image generation.
        """
        # self.pipe.unet.set_attn_processor(AttnProcessor2_0())
        # self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        self.pipe.to("cuda")
        self.pipe.upcast_vae()
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()
        self.pipe.enable_xformers_memory_efficient_attention()
        # self.pipe.enable_model_cpu_offload()

    def add_ip_adapter(self, image: Image.Image) -> None:
        """
        Adds an IP-Adapter for improved image transformation.

        Args:
            image (Image.Image): The image to be used with the IP-Adapter.
        """
        self.adapter_image = image
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "h94/IP-Adapter",
            subfolder="models/image_encoder",
            torch_dtype=torch.float16,
            use_safetensors=True
        )
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.model_name,
            controlnet=self.controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
            image_encoder=self.image_encoder,
            use_safetensors=True
        ).to("cuda")
        self.accelerate_pipe()
        self.pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="sdxl_models",
            weight_name="ip-adapter-plus_sdxl_vit-h.bin"
        )
        self.pipe.set_ip_adapter_scale(0.7)

    def load_control_image(self, image: Image.Image, conditioning: float = 1.0) -> None:
        """
        Loads the control image for conditioning.

        Args:
            image (Image.Image): The control image to be used.
            conditioning (float): The scale for ControlNet conditioning (default is 1.0).
        """
        self.control_image = image
        self.controlnet_conditioning_scale = conditioning
        
def invert_image(image: Image.Image) -> Image.Image:
    """
    Inverts the colors of a given image if a specified variable allows it.

    Args:
        image (Image.Image): The image to be inverted.

    Returns:
        Image.Image: The inverted image if allowed; otherwise, the original image.
    """
    # Skip function if variable set to False
    if invert_var.get():
        # Convert image to RGB mode if it's not in RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert the image to a NumPy array
        image_np = np.array(image)

        # Invert the colors by subtracting pixel values from 255
        inverted_np = 255 - image_np

        # Convert back to a PIL image
        inverted_image = Image.fromarray(inverted_np)
        
        return inverted_image
    else:
        return image

def blend_images(image1: Image.Image, image2: Image.Image, alpha: float = 0.45) -> Image.Image:
    """
    Blends two images together using a specified alpha ratio.

    Args:
        image1 (Image.Image): The first image to blend.
        image2 (Image.Image): The second image to blend.
        alpha (float): The blending ratio, with a default value of 0.45.

    Returns:
        Image.Image: The blended image.
    
    Raises:
        ValueError: If the images are not the same size.
    """
    # Ensure images are the same size
    if image1.size != image2.size:
        raise ValueError("Images must be the same size for blending.")

    # Ensure both images are in the same mode (e.g., both RGB or RGBA)
    if image1.mode != image2.mode:
        image2 = image2.convert(image1.mode)

    # Blend the images using the alpha ratio
    blended_image = Image.blend(image1, image2, alpha)
    
    return blended_image

def concatenate_images(image1: Image.Image, image2: Image.Image) -> Image.Image:
    """
    Concatenates two images horizontally after ensuring they have the same height.

    Args:
        image1 (Image.Image): The first image to concatenate.
        image2 (Image.Image): The second image to concatenate.

    Returns:
        Image.Image: The concatenated image.
    """
    # Ensure both images are in RGB mode
    if image1.mode != 'RGB':
        image1 = image1.convert('RGB')
    if image2.mode != 'RGB':
        image2 = image2.convert('RGB')

    # Resize the images if needed to ensure they have the same height
    if image1.height != image2.height:
        # Resize image2 to match image1 height, keeping aspect ratio
        image2 = image2.resize((int(image2.width * (image1.height / image2.height)), image1.height))
    
    # Concatenate images
    concatenated_image = Image.new('RGB', (image1.width + image2.width, image1.height))
    concatenated_image.paste(image1, (0, 0))
    concatenated_image.paste(image2, (image1.width, 0))
    
    return concatenated_image

class GifFrameCacher:
    """
    A class to cache and manipulate frames of a GIF image.

    Attributes:
        gif (Image.Image): The GIF image object.
        frame_cache (list): A list storing all frames of the GIF.

    Methods:
        _cache_all_frames(): Caches all frames of the GIF.
        get_frame(frame_index: int) -> Image.Image: Retrieves a frame by its index.
        get_total_frames() -> int: Returns the total number of cached frames.
        resize_frames(width: int, height: int) -> None: Resizes all cached frames to the specified dimensions.
    """

    def __init__(self, gif_path: str):
        """
        Initializes the GifFrameCacher with the specified GIF file path.

        Args:
            gif_path (str): The file path to the GIF image.
        """
        self.gif = Image.open(gif_path)
        self.frame_cache = []
        self._cache_all_frames()

    def _cache_all_frames(self) -> None:
        """Cache all frames of the GIF."""
        try:
            while True:
                # Cache the current frame
                self.frame_cache.append(self.gif.copy())
                # Move to the next frame
                self.gif.seek(self.gif.tell() + 1)
        except EOFError:
            # When no more frames are available
            pass

    def get_frame(self, frame_index: int) -> Image.Image:
        """Retrieve a frame by its index.

        Args:
            frame_index (int): The index of the frame to retrieve.

        Returns:
            Image.Image: The requested frame if it exists; otherwise, None.
        """
        if 0 <= frame_index < len(self.frame_cache):
            return self.frame_cache[frame_index]
        else:
            print(f"Frame index {frame_index} is out of range.")
            return None

    def get_total_frames(self) -> int:
        """Get the total number of cached frames.

        Returns:
            int: The total number of frames cached.
        """
        return len(self.frame_cache)
    
    def resize_frames(self, width: int, height: int) -> None:
        """Resize all cached frames to the specified width and height.

        Args:
            width (int): The new width for the frames.
            height (int): The new height for the frames.
        """
        resized_frames = []
        for frame in self.frame_cache:
            resized_frame = frame.resize((width, height), Image.Resampling.LANCZOS)
            resized_frames.append(resized_frame)
        self.frame_cache = resized_frames
        print(f"All frames resized to {width}x{height}.")

def triangular_scheduler(index: int, max_index: int) -> tuple[int, int]:
    """
    Generates an index that loops through values in a triangular pattern.

    Args:
        index (int): The current index (starts from 0).
        max_index (int): The maximum index for the triangular loop (exclusive).

    Returns:
        tuple(int, int): A tuple containing the new frame index and the updated internal index for the next call.
    """
    # Triangular logic: increasing then decreasing
    if index < max_index:
        result = index
    else:
        result = 2 * max_index - index

    # Update index for next call
    index = (index + 1) % (2 * max_index)

    return result, index

def compute_bounding_box_center_and_size(image: Image.Image) -> tuple[int, int, int, int] | None:
    """
    Compute the center and size of the bounding box for a white object in a black image.

    Args:
        image (Image.Image): Input image (PIL Image).

    Returns:
        tuple[int, int, int, int] | None: A tuple containing the (center_x, center_y, width, height) of the bounding box, 
        or None if no white object is found.
    """
    # Convert the image to grayscale and then to a numpy array directly
    img_array = np.array(image.convert("L"))
    
    # Threshold the image to binary (assuming the object is white and the background is black)
    binary_image = img_array > 128  # Adjust threshold if necessary
    
    # Check if there are any white pixels (early exit)
    if not np.any(binary_image):
        return None
    
    # Find the coordinates of the white pixels (bounding box corners)
    rows = np.any(binary_image, axis=1)
    cols = np.any(binary_image, axis=0)
    
    # Get the min and max row and column indices where white pixels are found
    min_y, max_y = np.where(rows)[0][[0, -1]]
    min_x, max_x = np.where(cols)[0][[0, -1]]
    
    # Compute center and size
    center_x = (min_x + max_x) // 2
    center_y = (min_y + max_y) // 2
    width = max_x - min_x
    height = max_y - min_y
    
    return (center_x, center_y, width, height)

def drawPerspective(image_height: int, image_width: int, box_height: int, box_width: int, center_x: int, center_y: int) -> Image.Image:
    """
    Draw a perspective effect based on an image size and bounding box.

    Args:
        image_height (int): The height of the image.
        image_width (int): The width of the image.
        box_height (int): The height of the box.
        box_width (int): The width of the box.
        center_x (int): The x-coordinate of the box center.
        center_y (int): The y-coordinate of the box center.

    Returns:
        Image.Image: A PIL Image with the perspective effect.
    """
    # Initialize a results array with zeros
    results = np.zeros((1, image_height, image_width, 3), dtype=np.float32)

    left = center_x - box_width // 2
    right = center_x + box_width // 2
    top = center_y - box_height // 2
    bottom = center_y + box_height // 2
    
    lim_x = max(left, image_width - right)
    lim_y = max(top, image_height - bottom)
    expand_limit = max(lim_x, lim_y) - 1

    # Precompute gradient values
    values = np.arange(expand_limit + 1) / expand_limit

    for ex in range(expand_limit):
        left -= 1
        right += 1
        top -= 1
        bottom += 1

        top_clamped = max(top, 0)
        bottom_clamped = min(bottom, image_height - 1)
        left_clamped = max(left, 0)
        right_clamped = min(right, image_width - 1)

        # Only perform assignments if indices are within bounds
        if left >= 0:
            results[0, top_clamped:bottom_clamped, left, :] = values[ex]
        if right < image_width:
            results[0, top_clamped:bottom_clamped, right, :] = values[ex]
        if top >= 0:
            results[0, top, left_clamped:right_clamped, :] = values[ex]
        if bottom < image_height:
            results[0, bottom, left_clamped:right_clamped, :] = values[ex]

    # Convert results to uint8 and remove the extra dimension
    results_image = (results[0] * 255).astype(np.uint8)

    # Create a PIL image from the NumPy array
    pil_image = Image.fromarray(results_image)

    return pil_image

def screen_blend(source_image: Image.Image, target_image: Image.Image) -> Image.Image:
    """
    Blend the white values from a black and white source image onto a target RGB image using screen blending.

    Args:
        source_image (Image.Image): A PIL Image object (black and white).
        target_image (Image.Image): A PIL Image object (target RGB for blending).

    Returns:
        Image.Image: A new PIL Image with screen blended values from the source image onto the target image.
    """
    # Ensure the source image is in grayscale
    source_image = source_image.convert("L")  # Convert to grayscale if not already

    # Ensure the target image is in RGB mode
    target_image = target_image.convert("RGB")  # Ensure target is in RGB

    # Create a new image for the result
    blended_image = Image.new("RGB", target_image.size)

    # Get the pixel data for both images
    source_pixels = source_image.load()
    target_pixels = target_image.load()
    blended_pixels = blended_image.load()

    # Perform screen blending
    for y in range(target_image.height):
        for x in range(target_image.width):
            # Get the grayscale value from the source image
            source_value = source_pixels[x, y]

            # Get the RGB values from the target image
            target_r, target_g, target_b = target_pixels[x, y]

            # Calculate the screen blended values
            blended_r = 255 - ((255 - target_r) * (255 - source_value) // 255)
            blended_g = 255 - ((255 - target_g) * (255 - source_value) // 255)
            blended_b = 255 - ((255 - target_b) * (255 - source_value) // 255)

            # Set the blended pixel in the new image
            blended_pixels[x, y] = (blended_r, blended_g, blended_b)

    return blended_image

# Global variable to control the main loop
global looping
looping = True

def keyboard_listener() -> None:
    """
    Listens for the 'esc' key press to set a stop signal.

    This function blocks until the 'esc' key is pressed, then it sets a global variable
    to signal that the application should stop running.
    """
    global looping
    keyboard.wait('esc')  # Blocks until the 'esc' key is pressed
    looping = False      # Set the stop signal

def fullsize_image(img_width: int, img_height: int) -> tuple[int, int]:
    """
    Calculate the full size of an image that fits within the screen dimensions while maintaining aspect ratio.

    Args:
        img_width (int): The width of the image.
        img_height (int): The height of the image.

    Returns:
        tuple[int, int]: A tuple containing the new height and width of the image that fits the screen.
    """
    # Initialize Tkinter to get screen dimensions
    root = Tk()
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    root.destroy()  # Close the Tkinter window
    
    # Calculate the aspect ratio of the image
    aspect_ratio = img_width / img_height
    
    # Calculate the new width and height while maintaining aspect ratio
    if img_width / screen_width > img_height / screen_height:
        new_width = screen_width
        new_height = int(screen_width / aspect_ratio)
    else:
        new_height = screen_height
        new_width = int(screen_height * aspect_ratio)
    
    return new_height, new_width

def image_updater(full_width, full_height, input_slot, output_slot, out=None) -> None:
    """
    Update the input and output images in the GUI.

    This function updates the visuals in the GUI for both the input and output image slots.
    The input image is only updated in debug mode, while the output image is resized and updated
    in all cases.
    """
    global photo_input
    global input_image
    global photo_output
    global output_image
    # EDIT
    global fullsize_image
    
    # EDIT
    fullsize_image = output_image.resize((full_width, full_height), Image.Resampling.LANCZOS)
    
    # Update GUI visuals
    if debug_var.get():  # Input only in debug mode
        photo_input = ImageTk.PhotoImage(input_image)
        input_slot.config(image=photo_input)
    else:
        # EDIT
        output_image = fullsize_image
        # output_image = output_image.resize((full_width, full_height), Image.Resampling.LANCZOS)
        
    
    photo_output = ImageTk.PhotoImage(output_image)
    output_slot.config(image=photo_output)
    if out : # Saving frame to video :
        if debug_var.get():
            frame = concatenate_images(input_image,output_image).resize((full_width, full_height//2), Image.Resampling.LANCZOS)
            out.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_BGR2RGB))    
        out.write(cv2.cvtColor(np.asarray(output_image), cv2.COLOR_BGR2RGB))
    
def start_record(full_width,full_height):
    # Define video codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec
    if debug_var.get():
        out = cv2.VideoWriter(video_save_var.get()+'.avi', fourcc, 3, (full_width,full_height//2))  # Output video file for output and input
    else :
        out = cv2.VideoWriter(video_save_var.get()+'.avi', fourcc, 3, (full_width,full_height))  # Output video file for output only (square ratio)
    print("Started record on file : "+video_save_var.get()+".avi")
    return(out)

def classic_loop(webcam : WebcamCapture, pipeline : SDPipeline, process_window, full_width, full_height, input_slot, output_slot, out=None) -> None:
    """
    Capture and process images in a continuous loop.

    This function captures an image from the webcam, blends it with the last output image,
    and transforms the blended image using a defined pipeline. The function updates the images
    displayed in the GUI and continues to loop until the global `looping` variable is set to False.
    """
    global input_image
    global output_image
    global looping
    
    # Capture and process the input image
    input_image = invert_image(webcam.capture_image())
    
    # Blend last output with new input
    blended_image = blend_images(input_image, output_image.resize((image_size.get(), image_size.get())), float(blend_var.get()))
    
    # Transform the image with the pipeline
    output_image = pipeline.transform_image(positive_prompt_var.get(), input_image=blended_image)
    
    if silhouette_var.get() :
        # Color the white silhouette
        colored_input = color_white_pixels(input_image)
        # Paste silhouette on the processed image
        output_image = paste_color_pixels(colored_input, output_image)
    
    # Update the GUI
    image_updater(full_width, full_height, input_slot, output_slot,out)
    
    # Loop or destroy the process window
    if looping:
        process_window.after(1, lambda: classic_loop(webcam, pipeline, process_window, full_width, full_height, input_slot, output_slot, out))
    else:   
        process_window.destroy()

def classic_handler(webcam : WebcamCapture) -> None:
    """
    Initialize and start the classic image processing loop.

    This function sets up the GUI, creates the image processing pipeline, and starts
    capturing images in a loop. It also sets default prompts if none are provided.

    The GUI is displayed in a separate window, and image updates are handled by
    `classic_loop`.
    """
    full_width, full_height = fullsize_image(image_size.get(), image_size.get())
    
    # Check if default prompt
    if positive_prompt_var.get() == "":
        positive_prompt_var.set("abstract, sparks, shiny, electricity, gold")
    
    # Begin looping
    global looping
    looping = True

    # Create Pipeline
    pipeline = SDPipeline(model_name=model_name_var.get())
    pipeline.accelerate_pipe()

    # Start the keyboard listener in a separate thread
    listener_thread = threading.Thread(target=keyboard_listener)
    listener_thread.start()

    # If asked, start the record
    out = None
    if record_var.get():
        out = start_record(full_width,full_height)

    # Generate the interface first
    process_window = tk.Toplevel(main_window)
    process_window.title("Generation")
    
    global input_image
    global output_image
    input_image = Image.new("RGB", (image_size.get(), image_size.get()))
    output_image = Image.new("RGB", (image_size.get(), image_size.get()))

    # Convert to PhotoImage and store references
    global input_photo
    global output_photo
    input_photo = ImageTk.PhotoImage(input_image)
    output_photo = ImageTk.PhotoImage(output_image)

    # Create the interface elements
    if debug_var.get():
        input_slot = tk.Label(process_window, image=input_photo)
        input_slot.image = input_photo
        input_slot.grid(row=0, column=0)
        output_slot = tk.Label(process_window, image=output_photo)
        output_slot.grid(row=0, column=1)
    else:
        input_slot = None
        output_slot = tk.Label(process_window, image=output_photo)
        output_slot.grid(row=0, column=0)
        process_window.attributes('-fullscreen', True)
        process_window.rowconfigure(0, weight=1, minsize=50)
        process_window.columnconfigure(0, weight=1, minsize=75)
        
    # Run classic_loop frequently to update images
    process_window.after(1, lambda: classic_loop(webcam, pipeline, process_window, full_width, full_height, input_slot, output_slot,out))
    process_window.mainloop()

    # Wait for the keyboard listener thread to finish
    try :
        listener_thread.join()
    except :
        pass
    if out :
        out.release()
    print("Exited loop and cleared objects")
    
def adapter_loop(webcam : WebcamCapture, pipeline : SDPipeline, process_window, full_width, full_height, input_slot, output_slot, out) -> None:
    """
    Capture and process images in a continuous loop for the adapter mode.

    This function captures an image from the webcam, blends it with the last output image,
    and transforms the blended image using a defined pipeline with an image adapter. 
    The function updates the images displayed in the GUI and continues to loop until the
    global `looping` variable is set to False.
    """
    global input_image
    global output_image
    global looping
    
    # Capture and process the input image
    input_image = invert_image(webcam.capture_image())
    
    # Blend last output with new input
    blended_image = blend_images(input_image, output_image.resize((image_size.get(), image_size.get())), float(blend_var.get()))
    
    # Transform the image with the pipeline
    output_image = pipeline.transform_image(positive_prompt_var.get(), input_image=blended_image)
    
    if silhouette_var.get() :
        # Color the white silhouette
        colored_input = color_white_pixels(input_image)
        # Paste silhouette on the processed image
        output_image = paste_color_pixels(colored_input, output_image)
    
    # Update the GUI
    image_updater(full_width, full_height, input_slot, output_slot,out)
    
    # Loop or destroy the process window
    if looping:
        process_window.after(1, lambda: adapter_loop(webcam, pipeline, process_window, full_width, full_height, input_slot, output_slot, out))
    else:   
        process_window.destroy()

def adapter_handler(webcam : WebcamCapture) -> None:
    """
    Initialize and start the adapter image processing loop.

    This function sets up the GUI, creates the image processing pipeline with an image adapter,
    and starts capturing images in a loop. It sets default prompts and adapter images if none are provided.

    The GUI is displayed in a separate window, and image updates are handled by
    `adapter_loop`.
    """
    full_width, full_height = fullsize_image(image_size.get(), image_size.get())
    
    # Check if default prompt
    if positive_prompt_var.get() == "":
        positive_prompt_var.set("underwater landscape")
    if adapter_image_var.get() == "":
        adapter_image_var.set("Images/underwater.png")
        adapter_image_slot = Image.open("Images/underwater.png")
        adapter_image_slot = resize_longer_side(adapter_image_slot, 64)
        adapter_imagetk_slot = ImageTk.PhotoImage(adapter_image_slot)
        adapter_preview["image"] = adapter_imagetk_slot

    # Begin looping
    global looping
    looping = True

    # Create Pipeline
    pipeline = SDPipeline(model_name=model_name_var.get())
    pipeline.accelerate_pipe()
    
    adapter_image = load_image(adapter_image_var.get())
    pipeline.add_ip_adapter(adapter_image)

    # If asked, start the record
    out = None
    if record_var.get():
        out = start_record(full_width,full_height)

    # Start the keyboard listener in a separate thread
    listener_thread = threading.Thread(target=keyboard_listener)
    listener_thread.start()

    # Generate the interface first
    process_window = tk.Toplevel(main_window)
    process_window.title("Generation")
    
    global input_image
    global output_image
    input_image = Image.new("RGB", (image_size.get(), image_size.get()))
    output_image = Image.new("RGB", (image_size.get(), image_size.get()))

    # Convert to PhotoImage and store references
    global input_photo
    global output_photo
    input_photo = ImageTk.PhotoImage(input_image)
    output_photo = ImageTk.PhotoImage(output_image)

    # Create the interface elements
    if debug_var.get():
        input_slot = tk.Label(process_window, image=input_photo)
        input_slot.image = input_photo
        input_slot.grid(row=0, column=0)
        output_slot = tk.Label(process_window, image=output_photo)
        output_slot.grid(row=0, column=1)
    else:
        input_slot = None
        output_slot = tk.Label(process_window, image=output_photo)
        output_slot.grid(row=0, column=0)
        process_window.attributes('-fullscreen', True)
        process_window.rowconfigure(0, weight=1, minsize=50)
        process_window.columnconfigure(0, weight=1, minsize=75)
        
    # Run adapter_loop frequently to update images
    process_window.after(1, lambda: adapter_loop(webcam,pipeline,process_window, full_width, full_height, input_slot, output_slot, out))
    process_window.mainloop()

    try : # Wait for the keyboard listener thread to finish
        listener_thread.join()
    except :
        pass
    if out :
        out.release()
    print("Exited loop and cleared objects")
    
def background_loop(webcam : WebcamCapture, pipeline : SDPipeline, process_window, full_width, full_height, input_slot, output_slot, max_index, index, gif_cacher, out) -> None:
    """
    Capture and process images in a continuous loop for background image processing.

    This function captures an image from the webcam, blends it with the last output image,
    retrieves a frame from a cached GIF, blends it with the input image, and transforms the
    blended image using a defined pipeline. The function updates the images displayed in the GUI
    and continues to loop until the global `looping` variable is set to False.

    The blending of images and frames allows for dynamic background integration in the output image.
    """
    global input_image
    global output_image
    global looping
    
    # Capture and process the input image
    input_image = invert_image(webcam.capture_image())
    
    # Blend last output with new input
    blended_image = blend_images(input_image, output_image.resize((image_size.get(), image_size.get())), float(blend_var.get()))
    
    # Get the correct index
    frame_index, index = triangular_scheduler(index, max_index)
    
    # Retrieve gif frame
    background_frame = gif_cacher.get_frame(frame_index)
    
    # Blend frame with the input
    background_input = blend_images(blended_image, background_frame, 0.20)

    # Transform the image with the pipeline
    output_image = pipeline.transform_image(positive_prompt_var.get(), input_image=background_input)
    
    if silhouette_var.get() :
        # Color the white silhouette
        colored_input = color_white_pixels(input_image)
        # Paste silhouette on the processed image
        output_image = paste_color_pixels(colored_input, output_image)
    
    # Update the GUI
    image_updater(full_width, full_height, input_slot, output_slot, out)
    
    # Loop or destroy the process window
    if looping:
        process_window.after(1, lambda : background_loop(webcam, pipeline, process_window, full_width, full_height, input_slot, output_slot, max_index, index, gif_cacher, out))
    else:
        process_window.destroy()

def background_handler(webcam : WebcamCapture) -> None:
    """
    Initialize and start the background image processing loop.

    This function sets up the GUI, creates the image processing pipeline with an image adapter,
    and starts capturing images in a loop. It sets default prompts and GIF paths if none are provided.

    The GUI is displayed in a separate window, and image updates are handled by
    `background_loop`. This allows for dynamic backgrounds to be blended with the main image.

    Returns:
        None
    """
    full_width, full_height = fullsize_image(image_size.get(), image_size.get())
    
    # Check if default prompt
    if positive_prompt_var.get() == "":
        positive_prompt_var.set("underwater landscape")
    if adapter_image_var.get() == "":
        adapter_image_var.set("Images/underwater.png")
    if background_gif_var.get() == "":
        background_gif_var.set("Images/underwater.gif")
    
    # Begin looping
    global looping
    looping = True

    # Create Pipeline
    pipeline = SDPipeline(model_name=model_name_var.get())
    pipeline.accelerate_pipe()
    
    adapter_image = load_image(adapter_image_var.get())
    pipeline.add_ip_adapter(adapter_image)

    # If asked, start the record
    out = None
    if record_var.get():
        out = start_record(full_width,full_height)

    # Start the keyboard listener in a separate thread
    listener_thread = threading.Thread(target=keyboard_listener)
    listener_thread.start()

    # Create Gif Cacher and initialize triangular selection
    gif_cacher = GifFrameCacher(background_gif_var.get())  # Cache every frame of the gif for easier access later on
    gif_cacher.resize_frames(image_size.get(), image_size.get())  # Resize every frame to the correct size
    max_index = gif_cacher.get_total_frames() - 1
    index = 0

    # Generate the interface first
    process_window = tk.Toplevel(main_window)
    process_window.title("Generation")
    
    global input_image
    global output_image
    input_image = Image.new("RGB", (image_size.get(), image_size.get()))
    output_image = Image.new("RGB", (image_size.get(), image_size.get()))

    # Convert to PhotoImage and store references
    global input_photo
    global output_photo
    input_photo = ImageTk.PhotoImage(input_image)
    output_photo = ImageTk.PhotoImage(output_image)

    # Create the interface elements
    if debug_var.get():
        input_slot = tk.Label(process_window, image=input_photo)
        input_slot.image = input_photo
        input_slot.grid(row=0, column=0)
        output_slot = tk.Label(process_window, image=output_photo)
        output_slot.grid(row=0, column=1)
    else:
        input_slot= None
        output_slot = tk.Label(process_window, image=output_photo)
        output_slot.grid(row=0, column=0)
        process_window.attributes('-fullscreen', True)
        process_window.rowconfigure(0, weight=1, minsize=50)
        process_window.columnconfigure(0, weight=1, minsize=75)

    # Run background_loop frequently to update images
    process_window.after(1, lambda : background_loop(webcam, pipeline, process_window, full_width, full_height, input_slot, output_slot, max_index, index, gif_cacher, out))
    process_window.mainloop()

    try : # Wait for the keyboard listener thread to finish
        listener_thread.join()
    except :
        pass
    if out :
        out.release()
    print("Exited loop and cleared objects")
    
def perspective_loop(webcam : WebcamCapture, pipeline : SDPipeline, process_window, full_width, full_height, input_slot, output_slot, center_x,center_y,box_width,box_height,out) -> None:
    """
    Continuously captures images from the webcam and processes them for perspective transformation.

    This function captures an image, computes the bounding box for the main object in the image,
    draws a perspective depth map based on the bounding box, loads the control image into the pipeline,
    and transforms the image using the pipeline. The output image is blended with the input image
    and updated in the GUI. The function loops until the global `looping` variable is set to False.

    Returns:
        None
    """
    global input_image
    global output_image
    global looping
    
    # Capture and process the input image
    input_image = invert_image(webcam.capture_image())
    
    # Compute bounding box
    bbox_center_and_size = compute_bounding_box_center_and_size(input_image)
    if bbox_center_and_size:
        center_x, center_y, box_width, box_height = bbox_center_and_size
    
    # Draw ControlNet depth map
    perspective = drawPerspective(image_size.get(), image_size.get(), box_height, box_width, center_x, center_y)
    
    # Load the control image in the pipeline
    pipeline.load_control_image(perspective)

    # Transform the image with the pipeline
    output_image = pipeline.transform_image(positive_prompt_var.get())
    
    if silhouette_var.get() :
        # Color the white silhouette
        colored_input = color_white_pixels(input_image)
        # Paste silhouette on the processed image
        output_image = paste_color_pixels(colored_input, output_image)
    
    # Update the GUI
    image_updater(full_width, full_height, input_slot, output_slot, out)
    
    # Loop or destroy the process window
    if looping:
        process_window.after(1, lambda : perspective_loop(webcam, pipeline, process_window, full_width, full_height, input_slot, output_slot,center_x,center_y,box_width,box_height,out))
    else:   
        process_window.destroy()

def perspective_handler(webcam : WebcamCapture) -> None:
    """
    Initializes and starts the perspective transformation image processing loop.

    This function sets up the GUI, creates the image processing pipeline,
    and starts capturing images in a loop. It sets a default prompt if none is provided
    and optionally adds an image adapter to the pipeline.

    Returns:
        None
    """
    full_width, full_height = fullsize_image(image_size.get(), image_size.get())
    
    # Check if default prompt
    if positive_prompt_var.get() == "":
        positive_prompt_var.set("perspective, brick wall, highly detailed")
    
    # Begin looping
    global looping
    looping = True

    # Create Pipeline
    pipeline = CNPipeline(model_name=model_name_var.get())
    pipeline.accelerate_pipe()
    
    # Initialize box values
    center_x = int(int(image_size.get())/2)
    center_y = center_x
    box_width = 1
    box_height = box_width
    
    if adapter_image_var.get() != "":
        adapter_image = load_image(adapter_image_var.get())
        pipeline.add_ip_adapter(adapter_image)

    # If asked, start the record
    out = None
    if record_var.get():
        out = start_record(full_width,full_height)

    # Start the keyboard listener in a separate thread
    listener_thread = threading.Thread(target=keyboard_listener)
    listener_thread.start()

    # Generate the interface first
    process_window = tk.Toplevel(main_window)
    process_window.title("Generation")
    
    global input_image
    global output_image
    input_image = Image.new("RGB", (image_size.get(), image_size.get()))
    output_image = Image.new("RGB", (image_size.get(), image_size.get()))

    # Convert to PhotoImage and store references
    global input_photo
    global output_photo
    input_photo = ImageTk.PhotoImage(input_image)
    output_photo = ImageTk.PhotoImage(output_image)

    # Create the interface elements
    if debug_var.get():
        input_slot = tk.Label(process_window, image=input_photo)
        input_slot.image = input_photo
        input_slot.grid(row=0, column=0)
        output_slot = tk.Label(process_window, image=output_photo)
        output_slot.grid(row=0, column=1)
    else:
        input_slot = None
        output_slot = tk.Label(process_window, image=output_photo)
        output_slot.grid(row=0, column=0)
        process_window.attributes('-fullscreen', True)
        process_window.rowconfigure(0, weight=1, minsize=50)
        process_window.columnconfigure(0, weight=1, minsize=75)
        
    # Run perspective_loop frequently to update images
    process_window.after(1, lambda : perspective_loop(webcam, pipeline, process_window, full_width, full_height, input_slot, output_slot,center_x,center_y,box_width,box_height,out))
    process_window.mainloop()

    try : # Wait for the keyboard listener thread to finish
        listener_thread.join()
    except :
        pass
    if out :
        out.release()
    print("Exited loop and cleared objects")

def resize_longer_side(image: Image.Image, size: int) -> Image.Image:
    """
    Resize the given image to have the specified size on its longer side.

    The image is resized while maintaining its aspect ratio. The longer side of the image
    is scaled to the specified size, and the shorter side is adjusted accordingly.

    Args:
        image (Image.Image): The image to resize.
        size (int): The target size for the longer side of the image.

    Returns:
        Image.Image: The resized image.
    """
    scale_factor = size / max(image.size[0], image.size[1])
    height = int(scale_factor * image.size[0])
    width = int(scale_factor * image.size[1])
    return image.resize((height, width))

def crop_and_resize(image: Image.Image, size: int) -> Image.Image:
    """
    Crop the given image to a centered square and resize it to the specified size.

    The image is cropped to a square shape by taking the center of the image,
    and then resized to the specified dimensions.

    Args:
        image (Image.Image): The image to crop and resize.
        size (int): The target size for the cropped image.

    Returns:
        Image.Image: The cropped and resized square image.
    """
    # Get the dimensions of the input image
    width, height = image.size
    
    # Determine the smaller dimension to create a square crop
    min_dim = min(width, height)
    
    # Calculate cropping coordinates to get a centered square
    left = (width - min_dim) // 2
    top = (height - min_dim) // 2
    right = (width + min_dim) // 2
    bottom = (height + min_dim) // 2
    
    # Crop the image to the square
    cropped_image = image.crop((left, top, right, bottom))
    
    # Resize the cropped image to the target size
    resized_image = cropped_image.resize((size, size), Image.Resampling.LANCZOS)
    
    return resized_image

def choose_color():
    """
    Opens a color chooser dialog for the user to select a color. 
    The selected color is then converted from RGB (0-255) to a tuple of floats (0-1)
    and stored in the global variable `color_code`. 
    Additionally, it updates the color label in the UI to display the selected color.

    Returns:
        None
    """
    global color_code
    selected_color = colorchooser.askcolor(title="Choose a color")
    if selected_color and selected_color[0]:  # Check if a color was selected
        color_code = tuple(c / 255 for c in selected_color[0])  # Convert RGB to tuple of floats (0 to 1)
        color_label.config(text=f"Selected color: {selected_color[1]}", bg=selected_color[1])

def color_white_pixels(image: Image.Image) -> Image.Image:
    """
    Replaces white pixels in an image with a specified color.

    Args:
        image (Image.Image): The input image in which white pixels will be replaced.

    Returns:
        Image.Image: The modified image with white pixels replaced by the specified color.
    """
    global color_code  # Ensure color_code is recognized as a global variable

    # Convert the image to RGBA (if not already in that mode)
    img = image.convert("RGBA")

    # Get the data of the image
    data = img.getdata()

    # Create a new list to hold the modified pixel data
    new_data = []

    # Loop through each pixel in the image
    for item in data:
        # Change all white (also shades of white)
        if item[0] > 240 and item[1] > 240 and item[2] > 240:  # Adjust threshold as needed
            # Replace white pixel with the specified color
            new_data.append((int(color_code[0] * 255), int(color_code[1] * 255), int(color_code[2] * 255), item[3]))  # Keep the original alpha
        else:
            new_data.append(item)

    # Update the image data
    img.putdata(new_data)

    return img  # Return the modified image

def paste_color_pixels(img1: Image.Image, img2: Image.Image) -> Image.Image:
    """
    Paste pixels from img1 to img2 where img1's pixels match the color_code.

    Parameters:
        img1 (Image.Image): The source image to take pixels from.
        img2 (Image.Image): The destination image to paste pixels onto.

    Returns:
        Image.Image: The modified img2 with pixels from img1 pasted on it.
    """
    global color_code

    # Convert images to RGBA if they are not already in that mode
    img1 = img1.convert("RGBA")
    img2 = img2.convert("RGBA")

    # Get the data of both images
    data1 = img1.getdata()
    data2 = img2.getdata()

    # Create a new list to hold the modified pixel data for img2
    new_data = []

    # Loop through each pixel in img1
    for i, item in enumerate(data1):
        # Check if the pixel matches the color_code
        if (item[0] / 255, item[1] / 255, item[2] / 255) == color_code:  # Normalize to 0-1 range
            new_data.append(item)  # Keep the pixel from img1
        else:
            new_data.append(data2[i])  # Keep the pixel from img2

    # Create a new image for the result
    result_img = Image.new("RGBA", img2.size)
    result_img.putdata(new_data)

    return result_img  # Return the modified img2

# Load presets from the JSON file
with open('presets.json', 'r') as file:
    presets = json.load(file)

def load_preset() -> None:
    """
    Load the selected preset from the presets dictionary and update the application state.

    This function retrieves the selected preset, sets the corresponding prompts, and updates
    the adapter and background images in the application. It also updates the invert and 
    effect type settings based on the selected preset.

    Returns:
        None
    """
    global adapter_image_slot
    global adapter_imagetk_slot
    global background_slot
    global backgroundtk_slot
    preset_selection = preset_var.get()
    chosen_preset = presets[preset_selection]

    # Update prompts and image paths
    positive_prompt_var.set(chosen_preset[0])
    negative_prompt_var.set(chosen_preset[1])
    adapter_image_var.set(chosen_preset[2])

    # Update adapter image preview
    if chosen_preset[2] == "":
        adapter_image_slot = Image.new("RGB", (128, 128))  # Set empty image in the preview
        adapter_imagetk_slot = ImageTk.PhotoImage(adapter_image_slot)
        adapter_preview["image"] = adapter_imagetk_slot
    else:
        adapter_image_slot = Image.open(chosen_preset[2])  # Set the image in the preview
        adapter_image_slot = crop_and_resize(adapter_image_slot, 128)
        adapter_imagetk_slot = ImageTk.PhotoImage(adapter_image_slot)
        adapter_preview["image"] = adapter_imagetk_slot

    # Update background gif
    background_gif_var.set(chosen_preset[3])
    if chosen_preset[3] == "":
        background_slot = Image.new("RGB", (128, 128))  # Set empty gif in the preview
        backgroundtk_slot = ImageTk.PhotoImage(background_slot)
        background_preview["image"] = backgroundtk_slot
    else:
        background_slot = Image.open(chosen_preset[3])  # Set the gif in the preview
        background_slot = crop_and_resize(background_slot, 128)
        backgroundtk_slot = ImageTk.PhotoImage(background_slot)
        background_preview["image"] = backgroundtk_slot

    # Update invert variable, blend value and effect type
    invert_var.set(chosen_preset[4] == "True")
    blend_var.set(chosen_preset[5])
    effect_type_var.set(chosen_preset[-1])  # Last element is always the effect type.

def upload_adapter_image() -> None:
    """
    Open a file dialog to upload an adapter image and update the preview.

    This function allows the user to select an image file from their filesystem,
    opens the selected image, crops and resizes it, and then updates the adapter 
    image preview in the application.

    Returns:
        None
    """
    img_name = askopenfilename()  # Open explorer window
    global adapter_image_slot
    global adapter_imagetk_slot
    adapter_image_slot = Image.open(img_name)
    adapter_image_slot = crop_and_resize(adapter_image_slot, 128)
    adapter_imagetk_slot = ImageTk.PhotoImage(adapter_image_slot)
    adapter_preview["image"] = adapter_imagetk_slot
    adapter_image_var.set(img_name)
    
def cycle_gif() -> None:
    """
    Cycle through the frames of the loaded GIF and update the background image preview.

    This function retrieves the next frame from the GIF stored in `np_gif`, resizes it,
    and updates the preview displayed in the application. If the end of the GIF is reached,
    it wraps around to the first frame.

    Returns:
        None
    """
    global background_slot
    global backgroundtk_slot
    global np_gif
    global gif_index

    try:
        gif_index += 1
        background_slot = Image.fromarray(np_gif[gif_index, :, :, :], 'RGB')
        background_slot = resize_longer_side(background_slot, 128)
    except IndexError:  # Catch index errors specifically
        gif_index = 0
        background_slot = Image.fromarray(np_gif[gif_index, :, :, :], 'RGB')
        background_slot = resize_longer_side(background_slot, 128)

    backgroundtk_slot = ImageTk.PhotoImage(background_slot)
    background_preview["image"] = backgroundtk_slot

def cycle_gif_loop() -> None:
    """
    Continuously cycle through the GIF frames in a loop.

    This function repeatedly calls `cycle_gif` to update the background image
    at intervals determined by `gif_delay`.

    Returns:
        None
    """
    while True:
        cycle_gif()
        time.sleep(gif_delay / 1000)

def upload_gif() -> None:
    """
    Open a file dialog to upload a GIF and prepare it for cycling.

    This function allows the user to select a GIF file from their filesystem,
    reads its frames into a NumPy array, and updates the background image preview
    in the application. It also starts a thread to continuously cycle through
    the GIF frames.

    Returns:
        None
    """
    global background_slot
    global backgroundtk_slot
    global np_gif
    global gif_delay

    gif_path = askopenfilename()  # Open explorer window

    background_gif_var.set(gif_path)

    with Image.open(gif_path) as im:
        gif_delay = im.info["duration"]
        np_gif = np.array(im)
        np_gif = np.resize(np_gif, (1, im.size[1], im.size[0], 3))

        try:
            while True:
                im.seek(im.tell() + 1)
                expand_frame = np.expand_dims(np.array(im), axis=0)
                np_gif = np.concatenate((np_gif, expand_frame))
        except EOFError:
            pass

    background_slot = Image.fromarray(np_gif[0, :, :, :], 'RGB')
    background_slot = resize_longer_side(background_slot, 128)
    backgroundtk_slot = ImageTk.PhotoImage(background_slot)
    background_preview["image"] = backgroundtk_slot

    # Open a thread to update the GIF in the preview.
    try:
        gif_thread.join()  # Ensure we don't duplicate the gif thread and close it if it already exists.
    except:
        pass
    gif_thread = threading.Thread(target=cycle_gif_loop)
    #gif_thread.daemon = True
    gif_thread.start()

def send_image_server() -> Generator[bytes, None, None]:
    """
    A generator function that continuously sends the current output image
    to a server as JPEG frames.

    The function converts the current `output_image` from RGB to BGR format,
    encodes it as a JPEG image, and yields the image data in a format suitable
    for streaming over HTTP. This is typically used for serving images in a
    web application.

    Yields:
        bytes: The JPEG image data in a multipart response format.
    """
    global fullsize_image
    while True:
        if fullsize_image is not None:
            numpy_image = np.array(fullsize_image)  
            bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR) # Convert from RGB to BGR
            _, buffer = cv2.imencode('.jpg', bgr_image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def get_url():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # Try to connect to a public IP (does not actually send data)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "Unable to retrieve IP"
    finally:
        s.close()
    return ip+":3142/video_feed"

def open_web_feed():
    webbrowser.open("http://"+get_url())

# Open Webcam
webcam = WebcamCapture(cam_index=0)

using_server=input("Should the output be broadcast to the server ? Y/N   -> ")
if using_server=="Y" or using_server=="y":
    # Create server
    app = Flask(__name__)

    @app.route('/')
    def home():
        return "Welcome to the Video Stream! Connect using this url and /video_feed"
    @app.route('/video_feed')        
    def video_feed():
        return Response(send_image_server(), mimetype='multipart/x-mixed-replace; boundary=frame')
    @app.route('/shutdown', methods=['POST'])
    def shutdown():
        """Shutdown the Flask app programmatically"""
        os.kill(os.getpid(), signal.SIGINT)  # Sends a SIGINT signal to the current process to shut down
        return 'Shutting down...'
    def shutdown_trigger():
        requests.post('http://localhost:3142/shutdown')
    def run_flask():
        app.run(host='0.0.0.0', port=3142)  # Run Flask on port 3142
        
    # Start Flask server in a separate thread
    flask_thread = threading.Thread(target=run_flask)#, daemon=True)
    flask_thread.start()
    using_server=True
else :
    using_server=False

# Initialize images once before the loop
global input_image
global output_image
input_image = Image.new("RGB", (512, 512))
output_image = Image.new("RGB", (512, 512))

# Initialize gif variables
gif_index=0
np_gif=np.zeros((2,1024,1024,3))
gif_delay=20

# Create Main Window 
main_window = tk.Tk()
main_window.title("Settings")

# Font definitions
font_size=3
title_font=font.Font(name="Title", family = "Helvetica", size=font_size*6) #Defining fonts for the interface
large_font=font.Font(name="Large", family = "Helvetica", size=font_size*5) 
medium_font=font.Font(name="Medium", family = "Helvetica", size=font_size*3)

# Frames creation
header_frame = tk.Frame(main_window)
header_frame.grid(row=1,column=1,sticky="nsew")
global_settings_frame = tk.Frame(main_window)
global_settings_frame.grid(row=2,column=0,sticky="nsew")
parameters_frame = tk.Frame(main_window)
parameters_frame.grid(row=2, column=1, sticky="nsew")
standard_parameters_frame = tk.Frame(parameters_frame)
standard_parameters_frame.grid(row=0, column=0, sticky="nsew")
adapter_parameters_frame = tk.Frame(parameters_frame)
adapter_parameters_frame.grid(row=0, column=1, sticky="nsew")
adapter_row_frame = tk.Frame(adapter_parameters_frame)
adapter_row_frame.grid(row=1,column=0,sticky="nsew")
background_row_frame = tk.Frame(adapter_parameters_frame)
background_row_frame.grid(row=2,column=0,sticky="nsew")
perspective_frame = tk.Frame(parameters_frame)
perspective_frame.grid(row=0, column=2, sticky="nsew")
adapter_buttons_frame = tk.Frame(parameters_frame)
adapter_buttons_frame.grid(row=10,column=1,sticky="nsew")

global_settings_frame.lift()
parameters_frame.lift()
standard_parameters_frame.lift()
adapter_parameters_frame.lift()
perspective_frame.lift()

# Logo
tk.Label(master=main_window, text = "Picture Generation",fg="white",bg="red", font='Large').grid(row=0,column=0, sticky="ew")
logo_lab7 = ImageTk.PhotoImage(resize_longer_side(Image.open("assets/Lab7_BlancRouge.png"),256))
logo=tk.Label(master=main_window, image = logo_lab7)
logo.configure(background='black')
logo.grid(row=0,column=1, sticky="ew")

# Standard Parameters
tk.Label(header_frame,text="Parameters for Standard FXs",font="Large").grid(row=0,column=0,sticky="nsew")
preset_var = tk.StringVar()
tk.Label(standard_parameters_frame, text="Preset Selection",font="Medium").grid(row=1,column=0,sticky="nsew")
tk.Spinbox(standard_parameters_frame, from_=0, to=len(presets)-1, font="Medium", textvariable=preset_var, command=load_preset, state='readonly').grid(row=1,column=1,sticky="nsew")
tk.Label(standard_parameters_frame,text="Preset Type :",font="Medium").grid(row=2,column=0,sticky="nsew")
effect_type_var = tk.StringVar()
tk.Label(standard_parameters_frame,textvariable=effect_type_var,font="Medium").grid(row=2,column=1,sticky="nsew")
tk.Label(standard_parameters_frame,text="",font="Medium").grid(row=3,column=0,sticky="nsew")
tk.Label(standard_parameters_frame,text="Positive Prompt",font="Medium").grid(row=4,column=0,sticky="nsew")
positive_prompt_var = tk.StringVar()
positive_prompt_entry = tk.Entry(standard_parameters_frame,textvariable=positive_prompt_var,font="Medium")
positive_prompt_entry.grid(row=4,column=1,sticky="nsew")
tk.Label(standard_parameters_frame,text="Negative Prompt",font="Medium").grid(row=5,column=0,sticky="nsew")
negative_prompt_var = tk.StringVar()
negative_prompt_entry = tk.Entry(standard_parameters_frame,textvariable=negative_prompt_var,font="Medium")
negative_prompt_entry.grid(row=5,column=1,sticky="nsew")
tk.Label(standard_parameters_frame, text="Image Size",font="Medium").grid(row=6,column=0,sticky="nsew")
image_size = tk.IntVar()
tk.Spinbox(standard_parameters_frame, values=(512,768,1024), font="Medium", textvariable=image_size, state='readonly', wrap=True).grid(row=6,column=1,sticky="nsew")
invert_var = tk.BooleanVar()
tk.Checkbutton(standard_parameters_frame, variable=invert_var, text="Invert Camera Image ?",font="Medium").grid(row=7,column=0,sticky="nsew")
tk.Label(standard_parameters_frame, text="Blending Value",font="Medium").grid(row=8,column=0,sticky="nsew")
blend_var = tk.DoubleVar()
blend_var.set(0.55)
tk.Spinbox(standard_parameters_frame, textvariable=blend_var, from_=0.0, to=1.0, increment=0.05, wrap=True).grid(row=8,column=1,sticky="nsew")
tk.Label(standard_parameters_frame, text="Model Selection",font="Medium").grid(row=9,column=0,sticky="nsew")
model_name_var = tk.StringVar()
model_name_var.set("stabilityai/sdxl-turbo")
model_name_entry = tk.Entry(standard_parameters_frame,textvariable=model_name_var,font="Medium")
model_name_entry.grid(row=9,column=1,sticky="nsew")

# Adapter Parameters
tk.Label(header_frame,text="Parameters for Adapter",font="Large").grid(row=0,column=1,sticky="nsew")
tk.Label(header_frame,text="/ Background FXs",font="Large").grid(row=0,column=2,sticky="nsew")
adapter_image_var = tk.StringVar()
tk.Button(adapter_row_frame, text="Select Adapter Image", command=upload_adapter_image).grid(row=0,column=0,sticky="nsew")
adapter_image_entry = tk.Entry(adapter_row_frame, textvariable=adapter_image_var, font="Medium")
adapter_image_entry.grid(row=1,column=0,sticky="nsew")
adapter_image = Image.new("RGB",(128,128))
adapter_tk_image = ImageTk.PhotoImage(adapter_image)
adapter_preview = tk.Label(adapter_parameters_frame,image=adapter_tk_image)
adapter_preview.grid(row=1,column=1,sticky="nsew")
background_gif_var = tk.StringVar()
tk.Button(background_row_frame, text="Select Background GIF", command=upload_gif).grid(row=0,column=0,sticky="nsew")
background_entry = tk.Entry(background_row_frame, textvariable=background_gif_var, font="Medium")
background_entry.grid(row=1,column=0,sticky="nsew")
background_preview = tk.Label(adapter_parameters_frame,image=adapter_tk_image)
background_preview.grid(row=2,column=1,sticky="nsew")

# Perspective Parameters
tk.Label(header_frame,text="Parameters for Perspective FXs",font="Large").grid(row=0,column=3,sticky="nsew")
global color_code
color_code=(1,1,1)
tk.Button(perspective_frame, text="Choose Color", command=choose_color, font="Medium").grid(row=1,column=0,sticky="nsew")
color_label = tk.Label(perspective_frame, text="Default Color White", font="Medium")
color_label.grid(row=2,column=0,sticky="nsew")
silhouette_var = tk.BooleanVar()
tk.Checkbutton(perspective_frame, variable=silhouette_var, text="Paste silhouette on output ?",font="Medium").grid(row=3,column=0,sticky="nsew")

# Should record ?
record_var=tk.BooleanVar()
tk.Checkbutton(global_settings_frame, variable=record_var, text="Record Output ?",font="Medium").grid(row=0,column=0,sticky="nsew")
tk.Label(global_settings_frame,text="Output filename (no extension)",font="Medium").grid(row=1,column=0,sticky="nsew")
video_save_var=tk.StringVar()
tk.Entry(global_settings_frame, textvariable=video_save_var,font="Medium").grid(row=2,column=0,sticky="nsew")

# Debug Mode
debug_var = tk.BooleanVar()
tk.Checkbutton(global_settings_frame, variable=debug_var, text="Enable Debug ?",font="Medium").grid(row=3,column=0,sticky="nsew")

# Open Preview with URL
url_button=tk.Button(global_settings_frame, text="Server unactive",font="Medium",command=open_web_feed)
url_button.config(state=tk.DISABLED)
url_button.grid(row=5,column=0,sticky="nsew")
if using_server :
    url_button.config(text="Hosting on URL : "+get_url(),state=tk.NORMAL)
    kill_button=tk.Button(global_settings_frame, text="/!\ Shutdown Server and App",font="Large",command=shutdown_trigger,bg="red",fg="white")
    kill_button.grid(row=6,column=0,sticky="nsew")

# Start Buttons
tk.Button(parameters_frame, text="Standard FXs",command=lambda : classic_handler(webcam),font="Large").grid(row=10,column=0,sticky="nsew")
tk.Button(adapter_buttons_frame, text="Adapter FXs",command=lambda : adapter_handler(webcam),font="Large").grid(row=0,column=0,sticky="nsew")
tk.Button(adapter_buttons_frame, text="Background FXs",command=lambda : background_handler(webcam),font="Large").grid(row=0,column=1,sticky="nsew")
tk.Button(parameters_frame, text="Perspective FXs",command=lambda : perspective_handler(webcam),font="Large").grid(row=10,column=2,sticky="nsew")

# Configure rows and columns for each frame
for frame in [main_window, parameters_frame, standard_parameters_frame, adapter_parameters_frame,
              adapter_row_frame, background_row_frame, adapter_buttons_frame, global_settings_frame]:
    col_number, row_number = frame.grid_size()
    frame.grid_rowconfigure(list(range(col_number)), weight=1)  # This can be adjusted if you have more rows
    frame.grid_columnconfigure(list(range(row_number)), weight=1)  # This can be adjusted if you have more columns

main_window.mainloop()

# Close everything
try :
    listener_thread.join()
except :
    pass
try :
    flask_thread.join()
except :
    pass
try :
    webcam.release()
except:
    pass

# Clean up
try :
    client_socket.close()
    server_socket.close()
except :
    pass

try :
    shutdown_trigger()
except :
    pass

try :
    out.release()
except :
    pass

print(threading.enumerate())