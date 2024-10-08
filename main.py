import cv2
from PIL import Image
import numpy as np
import keyboard
from IPython.display import display, clear_output
import ipywidgets
from diffusers import AutoPipelineForImage2Image, EulerAncestralDiscreteScheduler, StableDiffusionXLControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
import torch
import os
from optimum.onnxruntime import ORTStableDiffusionXLPipeline
import threading
from diffusers.models.attention_processor import AttnProcessor2_0
from transformers import CLIPVisionModelWithProjection

# GUI for .py only
import tkinter as tk
from tkinter import ttk, Tk, font
from tkinter.filedialog import askopenfilename
from PIL import ImageTk, GifImagePlugin
import time

# To send the frames as a stream
from flask import Flask, Response

"""
if os.environ.get('TRANSFORMERS_CACHE') and os.environ.get('HF_HOME'):    # Environment variables already defined, (down)load models there
    print("Path already defined, skipping")
    print(os.environ.get('HF_HOME'))
    pass
"""

open_file = open("config.txt", "r")
model_path = open_file.readline()
open_file.close()
if model_path == "" :
    open_file = open("config.txt", "w")
    model_path = input("Write path to where models should be downloaded")
    open_file.writelines(model_path)
    open_file.close()
print(model_path)

os.environ["HF_HOME"] = model_path
os.environ["TRANSFORMERS_CACHE"] = model_path    # path to downloaded models


torch.backends.cuda.matmul.allow_tf32 = True       # use 32 precision floats

GifImagePlugin.LOADING_STRATEGY = GifImagePlugin.LoadingStrategy.RGB_ALWAYS

class WebcamCapture:
    def __init__(self, cam_index=0):
        # Initialize the webcam once
        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise Exception("Could not open video device")
    
    def capture_image(self):
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

    def release(self):
        # Release the webcam when done
        self.cap.release()

class SDPipeline:
    def __init__(self, model_name="stabilityai/sdxl-turbo", seed=314159):
        # Store pipe info
        self.model_name = model_name
        # Initialize the SD pipeline
        self.pipe = AutoPipelineForImage2Image.from_pretrained(model_name, torch_dtype=torch.float16, variant="fp16", use_safetensors=True).to("cuda")
        # Set scheduler
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config, use_safetensors=True)
        # Initialize generator
        self.seed = seed
        self.generator = torch.Generator(device="cuda").manual_seed(seed)
        #Initialize adapter fields to None
        self.adapter_image=None
        self.image_encoder=None
        
    def transform_image(self, prompt,  input_image, negative_prompt="",num_steps=2, cfg=1.0, strength=0.5, return_type="image"):
        input_image = load_image(input_image)
        if self.adapter_image==None :
            output = self.pipe(prompt, negative_prompt=negative_prompt, image=input_image, num_inference_steps=num_steps, strength=strength, guidance_scale=cfg, generator=self.generator)
        else :
            output = self.pipe(prompt, negative_prompt=negative_prompt, image=input_image, ip_adapter_image=self.adapter_image, num_inference_steps=num_steps, strength=strength, guidance_scale=cfg, generator=self.generator)
        # Access the generated image correctly
        self.output_image = output.images[0]
        return self.output_image
    
    def accelerate_pipe(self):
        #self.pipe.unet.set_attn_processor(AttnProcessor2_0())
        #self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        self.pipe.to("cuda")
        self.pipe.upcast_vae()
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()
        self.pipe.enable_xformers_memory_efficient_attention()
        #self.pipe.enable_model_cpu_offload()

    def add_ip_adapter(self,image):
        self.adapter_image=image
        self.image_encoder=CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter",subfolder="models/image_encoder",torch_dtype=torch.float16, use_safetensors=True)
        self.pipe = AutoPipelineForImage2Image.from_pretrained(self.model_name, torch_dtype=torch.float16, variant="fp16", image_encoder=self.image_encoder, use_safetensors=True).to("cuda")
        self.accelerate_pipe()
        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.bin")
        self.pipe.set_ip_adapter_scale(0.7)
        
class CNPipeline:
    def __init__(self, model_name="stabilityai/sdxl-turbo", control_net="diffusers/controlnet-depth-sdxl-1.0",seed=314159):
        # Store pipe info
        self.model_name = model_name
        # Initialize the ControlNet
        self.controlnet = ControlNetModel.from_pretrained(control_net, torch_dtype=torch.float16, use_safetensors=True)
        # Initialize the SDXLPipe
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(self.model_name, controlnet=self.controlnet, torch_dtype=torch.float16, use_safetensors=True)
        # Set scheduler
        self.pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(self.pipe.scheduler.config)
        # Initialize generator
        self.seed = seed
        self.generator = torch.Generator(device="cuda").manual_seed(seed)
        #Initialize adapter fields to None
        self.adapter_image=None
        self.image_encoder=None
        
    def transform_image(self, prompt, negative_prompt="", num_steps=1, cfg=1.0, strength=1, return_type="image"):
        if self.adapter_image==None :
            output = self.pipe(prompt, negative_prompt=negative_prompt, image=self.control_image, controlnet_conditioning_scale=self.controlnet_conditioning_scale, num_inference_steps=num_steps, strength=strength, guidance_scale=cfg, generator=self.generator)
        else :
            output = self.pipe(prompt, negative_prompt=negative_prompt, image=self.control_image, controlnet_conditioning_scale=self.controlnet_conditioning_scale, ip_adapter_image=self.adapter_image, num_inference_steps=num_steps, strength=strength, guidance_scale=cfg, generator=self.generator)
        # Access the generated image correctly
        self.output_image = output.images[0]
        return self.output_image
    
    def accelerate_pipe(self):
        #self.pipe.unet.set_attn_processor(AttnProcessor2_0())
        #self.pipe.unet = torch.compile(self.pipe.unet, mode="reduce-overhead", fullgraph=True)
        self.pipe.to("cuda")
        self.pipe.upcast_vae()
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()
        self.pipe.enable_xformers_memory_efficient_attention()
        #self.pipe.enable_model_cpu_offload()

    def add_ip_adapter(self,image):
        self.adapter_image=image
        self.image_encoder=CLIPVisionModelWithProjection.from_pretrained("h94/IP-Adapter",subfolder="models/image_encoder",torch_dtype=torch.float16, use_safetensors=True)
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(self.model_name, controlnet=self.controlnet, torch_dtype=torch.float16, variant="fp16", image_encoder=self.image_encoder, use_safetensors=True).to("cuda")
        self.accelerate_pipe()
        self.pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter-plus_sdxl_vit-h.bin")
        self.pipe.set_ip_adapter_scale(0.7)
        
    def load_control_image(self,image,conditioning=1.0):
        self.control_image = image
        self.controlnet_conditioning_scale = conditioning
        
def invert_image(image):
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
    else :
        return(image)

def blend_images(image1, image2, alpha=0.45):
    # Ensure images are the same size
    if image1.size != image2.size:
        raise ValueError("Images must be the same size for blending.")

    # Ensure both images are in the same mode (e.g., both RGB or RGBA)
    if image1.mode != image2.mode:
        image2 = image2.convert(image1.mode)

    # Blend the images using the alpha ratio
    blended_image = Image.blend(image1, image2, alpha)
    
    return blended_image

def concatenate_images(image1, image2):
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
    def __init__(self, gif_path):
        self.gif = Image.open(gif_path)
        self.frame_cache = []
        self._cache_all_frames()

    def _cache_all_frames(self):
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

    def get_frame(self, frame_index):
        """Retrieve a frame by its index."""
        if 0 <= frame_index < len(self.frame_cache):
            return self.frame_cache[frame_index]
        else:
            print(f"Frame index {frame_index} is out of range.")
            return None

    def get_total_frames(self):
        """Get the total number of frames."""
        return len(self.frame_cache)
    
    def resize_frames(self, width, height):
        """Resize all cached frames to the specified width and height."""
        resized_frames = []
        for frame in self.frame_cache:
            resized_frame = frame.resize((width, height), Image.Resampling.LANCZOS)
            resized_frames.append(resized_frame)
        self.frame_cache = resized_frames
        print(f"All frames resized to {width}x{height}.")

def triangular_scheduler(index, max_index):
    """
    Generates an index that loops through values in a triangular pattern.
    
    :param index: The current index (starts from 0).
    :param max_index: The maximum index for the triangular loop (exclusive).
    :return: The new frame index and the updated internal index for the next call.
    """
    # Triangular logic: increasing then decreasing
    if index < max_index:
        result = index
    else:
        result = 2 * max_index - index

    # Update index for next call
    index = (index + 1) % (2 * max_index)

    return result, index

def compute_bounding_box_center_and_size(image):
    """
    Compute the center and size of the bounding box for a white object in a black image.
    
    :param image: Input image (PIL Image).
    :return: A tuple containing the (center_x, center_y, width, height) of the bounding box.
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

def drawPerspective(image_height, image_width, box_height, box_width, center_x, center_y):
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

def screen_blend(source_image, target_image):
    """
    Blend the white values from a black and white source image onto a target RGB image using screen blending.

    :param source_image: A PIL Image object (black and white).
    :param target_image: A PIL Image object (target RGB for blending).
    :return: A new PIL Image with screen blended values from the source image onto the target image.
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

def keyboard_listener():
    global looping
    keyboard.wait('esc')  # Blocks until the 'esc' key is pressed
    looping = False      # Set the stop signal
   
def fullsize_image(img_width, img_height):
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
 
def image_updater():
    global input_slot
    global photo_input
    global input_image
    global output_slot
    global photo_output
    global output_image
    global ndi_stream
    
    # Update GUI visuals
    if debug_var.get() :            # Input only in debug mode
        photo_input = ImageTk.PhotoImage(input_image)
        input_slot.config(image=photo_input)
        input_slot["image"] = photo_input
    else :
        output_image=output_image.resize((full_width,full_height),Image.Resampling.LANCZOS)
    photo_output = ImageTk.PhotoImage(output_image)
    output_slot.config(image=photo_output)
    output_slot["image"] = photo_output
    
def classic_loop():
    global webcam
    global input_image
    global output_image
    global pipeline
    global looping
    global process_window
    # Capture and process the input image
    input_image = invert_image(webcam.capture_image())
    # Blend last output with new input
    blended_image = blend_images(input_image, output_image.resize((image_size.get(),image_size.get())), float(blend_var.get()))
    # Transform the image with the pipeline
    output_image = pipeline.transform_image(positive_prompt_var.get(), input_image=blended_image)
    image_updater()
    if looping :
        process_window.after(1,classic_loop)
    else :   
        process_window.destroy()
        
def classic_handler():
    
    # Get screen size
    global full_width, full_height
    full_width, full_height = fullsize_image(image_size.get(), image_size.get())
    
    # Check if default prompt
    if positive_prompt_var.get() == "":
        positive_prompt_var.set("abstract, sparks, shiny, electricity, gold")
    
    # Begin looping
    global looping
    looping = True

    # Create Pipeline
    global pipeline
    pipeline = SDPipeline()
    pipeline.accelerate_pipe()

    # Start the keyboard listener in a separate thread
    listener_thread = threading.Thread(target=keyboard_listener)
    listener_thread.start()

    # Generate the interface first
    global process_window
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
    global input_slot
    global output_slot
    if debug_var.get() :
        input_slot = tk.Label(process_window, image=input_photo)
        input_slot.image = input_photo
        input_slot.grid(row=0, column=0)
        output_slot = tk.Label(process_window, image=output_photo)
        output_slot.grid(row=0, column=1)
    else :
        output_slot = tk.Label(process_window, image=output_photo)
        output_slot.grid(row=0, column=0)
        process_window.attributes('-fullscreen', True)
        process_window.rowconfigure(0, weight=1, minsize=50)
        process_window.columnconfigure(0, weight=1, minsize=75)
        
    # Run classic_loop frequently to update images
    process_window.after(1, classic_loop)
    process_window.mainloop()

    # Wait for the keyboard listener thread to finish
    listener_thread.join()
    print("exited loop and cleared thread")
    
def adapter_loop():
    global webcam
    global input_image
    global output_image
    global pipeline
    global looping
    global process_window
    # Capture and process the input image
    input_image = invert_image(webcam.capture_image())
    # Blend last output with new input
    blended_image = blend_images(input_image, output_image.resize((image_size.get(),image_size.get())), float(blend_var.get()))
    # Transform the image with the pipeline
    output_image = pipeline.transform_image(positive_prompt_var.get(), input_image=blended_image)
    image_updater()
    if looping :
        process_window.after(1,adapter_loop)
    else :   
        process_window.destroy()
        
def adapter_handler():
    
    # Get screen size
    global full_width, full_height
    full_width, full_height = fullsize_image(image_size.get(), image_size.get())
    
    # Check if default prompt
    if positive_prompt_var.get() == "":
        positive_prompt_var.set("underwater landscape")
    if adapter_image_var.get() == "":
        adapter_image_var.set("Images/underwater.png")
        adapter_image_slot = Image.open("Images/underwater.png")
        adapter_image_slot = resize_longer_side(adapter_image_slot,64)
        adapter_imagetk_slot = ImageTk.PhotoImage(adapter_image_slot)
        adapter_preview["image"]=adapter_imagetk_slot
    # Begin looping
    global looping
    looping = True

    # Create Pipeline
    global pipeline
    pipeline = SDPipeline()
    pipeline.accelerate_pipe()
    adapter_image = load_image(adapter_image_var.get())
    pipeline.add_ip_adapter(adapter_image)

    # Start the keyboard listener in a separate thread
    listener_thread = threading.Thread(target=keyboard_listener)
    listener_thread.start()

    # Generate the interface first
    global process_window
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
    global input_slot
    global output_slot
    if debug_var.get() :
        input_slot = tk.Label(process_window, image=input_photo)
        input_slot.image = input_photo
        input_slot.grid(row=0, column=0)
        output_slot = tk.Label(process_window, image=output_photo)
        output_slot.grid(row=0, column=1)
    else :
        output_slot = tk.Label(process_window, image=output_photo)
        output_slot.grid(row=0, column=0)
        process_window.attributes('-fullscreen', True)
        process_window.rowconfigure(0, weight=1, minsize=50)
        process_window.columnconfigure(0, weight=1, minsize=75)
        
    # Run classic_loop frequently to update images
    process_window.after(1, adapter_loop)
    process_window.mainloop()

    # Wait for the keyboard listener thread to finish
    listener_thread.join()
    print("exited loop and cleared thread")
    
def background_loop():
    global webcam
    global input_image
    global output_image
    global pipeline
    global looping
    global process_window
    global gif_cacher
    global index
    global max_index
    # Capture and process the input image
    input_image = invert_image(webcam.capture_image())
    # Blend last output with new input
    blended_image = blend_images(input_image, output_image.resize((image_size.get(),image_size.get())), float(blend_var.get()))
    # Get the correct index
    frame_index, index = triangular_scheduler(index, max_index)
    #print(frame_index,index,max_index)
    # Retrieve gif frame
    background_frame = gif_cacher.get_frame(frame_index)
    # Blend frame with the input
    background_input = blend_images(blended_image, background_frame, 0.20)
    # Transform the image with the pipeline
    output_image = pipeline.transform_image(positive_prompt_var.get(), input_image=background_input)
    image_updater()
    if looping :
        process_window.after(1,background_loop)
    else :   
        process_window.destroy()
        
def background_handler():
    
    # Get screen size
    global full_width, full_height
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
    global pipeline
    pipeline = SDPipeline()
    pipeline.accelerate_pipe()
    adapter_image = load_image(adapter_image_var.get())
    pipeline.add_ip_adapter(adapter_image)

    # Start the keyboard listener in a separate thread
    listener_thread = threading.Thread(target=keyboard_listener)
    listener_thread.start()

    global gif_cacher
    global max_index
    global index
    gif_cacher = GifFrameCacher(background_gif_var.get()) # Cache every frame of the gif for easier access later on
    gif_cacher.resize_frames(image_size.get(),image_size.get()) # Resize every frame to the correct size
    max_index = gif_cacher.get_total_frames()-1
    index = 0

    # Generate the interface first
    global process_window
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
    global input_slot
    global output_slot
    if debug_var.get() :
        input_slot = tk.Label(process_window, image=input_photo)
        input_slot.image = input_photo
        input_slot.grid(row=0, column=0)
        output_slot = tk.Label(process_window, image=output_photo)
        output_slot.grid(row=0, column=1)
    else :
        output_slot = tk.Label(process_window, image=output_photo)
        output_slot.grid(row=0, column=0)
        process_window.attributes('-fullscreen', True)
        process_window.rowconfigure(0, weight=1, minsize=50)
        process_window.columnconfigure(0, weight=1, minsize=75)

    # Run classic_loop frequently to update images
    process_window.after(1, background_loop)
    process_window.mainloop()

    # Wait for the keyboard listener thread to finish
    listener_thread.join()
    print("exited loop and cleared thread")

def perspective_loop():
    global webcam
    global input_image
    global output_image
    global pipeline
    global looping
    global process_window
    # Capture and process the input image
    input_image = invert_image(webcam.capture_image())
    # Compute bounding box
    bbox_center_and_size = compute_bounding_box_center_and_size(input_image)
    if bbox_center_and_size:
        center_x, center_y, box_width, box_height = bbox_center_and_size
    # Draw ControlNet depth map
    perspective = drawPerspective(image_size.get(),image_size.get(),box_height,box_width,center_x,center_y)
    # Load the control image in the pipeline
    pipeline.load_control_image(perspective)
    # Transform the image with the pipeline
    output_image = pipeline.transform_image(positive_prompt_var.get())
    output_image = screen_blend(input_image,output_image)
    image_updater()
    if looping :
        process_window.after(1,perspective_loop)
    else :   
        process_window.destroy()
        
def perspective_handler():
    
    # Get screen size
    global full_width, full_height
    full_width, full_height = fullsize_image(image_size.get(), image_size.get())
    
    # Check if default prompt
    if positive_prompt_var.get() == "":
        positive_prompt_var.set("perspective, brick wall, highly detailed")
    
    # Begin looping
    global looping
    looping = True

    # Create Pipeline
    global pipeline
    global adapter_image
    pipeline = CNPipeline()
    pipeline.accelerate_pipe()
    if adapter_image_var.get() != "":
        adapter_image = load_image(adapter_image_var.get())
        pipeline.add_ip_adapter(adapter_image)

    # Start the keyboard listener in a separate thread
    listener_thread = threading.Thread(target=keyboard_listener)
    listener_thread.start()

    # Generate the interface first
    global process_window
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
    global input_slot
    global output_slot
    if debug_var.get() :
        input_slot = tk.Label(process_window, image=input_photo)
        input_slot.image = input_photo
        input_slot.grid(row=0, column=0)
        output_slot = tk.Label(process_window, image=output_photo)
        output_slot.grid(row=0, column=1)
    else :
        output_slot = tk.Label(process_window, image=output_photo)
        output_slot.grid(row=0, column=0)
        process_window.attributes('-fullscreen', True)
        process_window.rowconfigure(0, weight=1, minsize=50)
        process_window.columnconfigure(0, weight=1, minsize=75)
        
    # Run classic_loop frequently to update images
    process_window.after(1, perspective_loop)
    process_window.mainloop()

    # Wait for the keyboard listener thread to finish
    listener_thread.join()
    print("exited loop and cleared thread")

def resize_longer_side(image,size):
    scale_factor = size/max(image.size[0],image.size[1])
    height=int(scale_factor*image.size[0])
    width=int(scale_factor*image.size[1])
    return(image.resize((height,width)))

def crop_and_resize(image: Image.Image, size: int) -> Image.Image:
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

# Presets : positive_prompt, negative_prompt, adapter_image, background_gif, should_invert_image_?, type_of_effect
presets = {
    "0":["","","","",""],
    "1":["abstract, sparks, shiny, electricity, gold","low quality, blur, nsfw, text, watermark","","","True","Standard Effect"],
    "2":["abstract, cobweb, silk, threads, strings","low quality, blur, nsfw, text, watermark","","","True","Standard Effect"],
    "3":["paper origami, abstract","low quality, blur, nsfw, text, watermark","","","True","Standard Effect"],
    "4":["side view of dark modern skyscrapers with an opening to the sky in the middle","low quality, blur, nsfw, text, watermark","Images/building.png","","True","Adapter Effect"],
    "5":["abstract particles","low quality, blur, nsfw, text, watermark","Images/particles.png","","True","Adapter Effect"],
    "6":["abstract, dunes made of sand","low quality, blur, nsfw, text, watermark","Images/dunes.png","","True","Adapter Effect"],
    "7":["waterfall in the middle of a forest","low quality, blur, nsfw, text, watermark","Images/forest.png","Images/forest.gif","True","Background Effect"],
    "8":["underwater landscape","low quality, blur, nsfw, text, watermark","Images/underwater.png","Images/underwater.gif","True","Background Effect"],
    "9":["abstract, galaxy, plasma, fluid shapes","low quality, blur, nsfw, text, watermark","Images/plasma.png","Images/plasma.gif","False","Background Effect"],
    "10":["perspective, brick wall, highly detailed","low quality, blur, nsfw, text, watermark","Images/bricks.png","","True","Perspective Effect"],
    "11":["perspective, abstract wallpaper, bright yellow and purple, highly detailed, intricate patterns","low quality, blur, nsfw, text, watermark","","","True","Perspective Effect"],
    "12":["waterfall in the middle of a rock cliff","low quality, blur, nsfw, text, watermark","Images/rock_cliff.png","Images/rock_cliff.gif","True","Background Effect"],
    "13":["waterfall","low quality, blur, nsfw, text, watermark","","","True","Standard Effect"],
           }

def load_preset():
    global adapter_image_slot
    global adapter_imagetk_slot
    global background_slot
    global backgroundtk_slot
    preset_selection = preset_var.get()
    chosen_preset = presets[preset_selection]
    positive_prompt_var.set(chosen_preset[0])
    negative_prompt_var.set(chosen_preset[1])
    adapter_image_var.set(chosen_preset[2])
    if chosen_preset[2] == "" :
        adapter_image_slot = Image.new("RGB",(128,128))               # Set empty image in the preview
        adapter_imagetk_slot = ImageTk.PhotoImage(adapter_image_slot)
        adapter_preview["image"]=adapter_imagetk_slot
    else :
        adapter_image_slot = Image.open(chosen_preset[2])               # Set the image in the preview
        adapter_image_slot = crop_and_resize(adapter_image_slot,128)
        adapter_imagetk_slot = ImageTk.PhotoImage(adapter_image_slot)
        adapter_preview["image"]=adapter_imagetk_slot
    background_gif_var.set(chosen_preset[3])
    if chosen_preset[3] == "" :
        background_slot = Image.new("RGB",(128,128))               # Set empty gif in the preview
        backgroundtk_slot = ImageTk.PhotoImage(background_slot)
        background_preview["image"]=backgroundtk_slot
    else :
        background_slot = Image.open(chosen_preset[2])               # Set the gif in the preview
        background_slot = crop_and_resize(background_slot,128)
        backgroundtk_slot = ImageTk.PhotoImage(background_slot)
        background_preview["image"]=backgroundtk_slot
    if chosen_preset[4] == "False":
        invert_var.set(False)
    else :
        invert_var.set(True)
    
    effect_type_var.set(chosen_preset[-1]) # Last element is always the effect type.
        
def upload_adapter_image():
    img_name = askopenfilename() # Open explorer window
    global adapter_image_slot
    global adapter_imagetk_slot
    adapter_image_slot = Image.open(img_name)
    adapter_image_slot = crop_and_resize(adapter_image_slot,128)
    adapter_imagetk_slot = ImageTk.PhotoImage(adapter_image_slot)
    adapter_preview["image"]=adapter_imagetk_slot
    adapter_image_var.set(img_name)
    
def cycle_gif():
    global background_slot
    global backgroundtk_slot
    global np_gif
    global gif_index
    global gif_delay
    try :
        gif_index=gif_index+1
        background_slot = Image.fromarray(np_gif[gif_index,:,:,:],'RGB')
        background_slot = resize_longer_side(background_slot,128)
    except :
        gif_index=0
        background_slot = Image.fromarray(np_gif[gif_index,:,:,:],'RGB')
        background_slot = resize_longer_side(background_slot,128)
    backgroundtk_slot=ImageTk.PhotoImage(background_slot)
    background_preview["image"]=backgroundtk_slot

def cycle_gif_loop():
    while True:
        cycle_gif()
        time.sleep(gif_delay/1000)

def upload_gif():
    global background_slot
    global backgroundtk_slot
    global np_gif
    global gif_delay
    
    gif_path = askopenfilename() # Open explorer window
    
    background_gif_var.set(gif_path)
    
    with Image.open(gif_path) as im :
        gif_delay=im.info["duration"]
        np_gif=np.array(im)
        np_gif=np.resize(np_gif,(1,im.size[1],im.size[0],3))
        try:
            while 1:
                im.seek(im.tell() + 1)
                expand_frame=np.expand_dims(np.array(im),axis=0)
                np_gif=np.concatenate((np_gif,expand_frame))
        except EOFError:
            pass
    background_slot = Image.fromarray(np_gif[0,:,:,:],'RGB')
    background_slot = resize_longer_side(background_slot,128)
    backgroundtk_slot = ImageTk.PhotoImage(background_slot)
    background_preview["image"]=backgroundtk_slot
    
    # Open a thread to update the gif in the preview.
    try :
        gif_thread.join() # Make sure we don't duplicate the gif thread and close it if it already exists.
    except :
        pass
    gif_thread = threading.Thread(target=cycle_gif_loop)
    gif_thread.daemon = True
    gif_thread.start()

def send_image_server():
    global output_image
    print("Executing send_image_server")
    while True:
        if output_image is not None:
            numpy_image = np.array(output_image)# Convert from RGB to BGR
            bgr_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', bgr_image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Open Webcam
global webcam
webcam = WebcamCapture(cam_index=0)

# This part creates a Server to view the output using ip/video_feed
"""
# Create server
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to the Video Stream!"
@app.route('/video_feed')        
def video_feed():
    return Response(send_image_server(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Thread to run the Flask app
def run_flask():
    app.run(host='0.0.0.0', port=3142)  # Run Flask on a different port
    
# Start Flask server in a separate thread
flask_thread = threading.Thread(target=run_flask, daemon=True)
flask_thread.start()
"""

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
large_font=font.Font(name="Large", family = "Helvetica", size=font_size*5) #Defining fonts for the interface
medium_font=font.Font(name="Medium", family = "Helvetica", size=font_size*3)

# Frames creation
header_frame = tk.Frame(main_window)
header_frame.grid(row=0,column=0)
parameters_frame = tk.Frame(main_window)
parameters_frame.grid(row=1,column=0)
standard_parameters_frame = tk.Frame(parameters_frame)
standard_parameters_frame.grid(row=0,column=0)
adapter_parameters_frame = tk.Frame(parameters_frame)
adapter_parameters_frame.grid(row=0,column=1)
adapter_filename_frame = tk.Frame(adapter_parameters_frame)
adapter_filename_frame.grid(row=1,column=0)
background_frame = tk.Frame(adapter_parameters_frame)
background_frame.grid(row=2,column=0)
perspective_parameters_frame = tk.Frame(parameters_frame)
perspective_parameters_frame.grid(row=0,column=2)
buttons_frame = tk.Frame(main_window)
buttons_frame.grid(row=2,column=0)

# Standard Parameters
tk.Label(standard_parameters_frame,text="Parameters for Standard FXs",font="Large").grid(row=0,column=0,sticky="nsew")
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

# Adapter Parameters
tk.Label(adapter_parameters_frame,text="Parameters for Adapter FXs",font="Large").grid(row=0,column=0,sticky="nsew")
adapter_image_var = tk.StringVar()
tk.Button(adapter_filename_frame, text="Select Adapter Image", command=upload_adapter_image).grid(row=0,column=0,sticky="nsew")
adapter_image_entry = tk.Entry(adapter_filename_frame, textvariable=adapter_image_var, font="Medium")
adapter_image_entry.grid(row=1,column=0,sticky="nsew")
adapter_image = Image.new("RGB",(128,128))
adapter_tk_image = ImageTk.PhotoImage(adapter_image)
adapter_preview = tk.Label(adapter_parameters_frame,image=adapter_tk_image)
adapter_preview.grid(row=1,column=1,sticky="nsew")
background_gif_var = tk.StringVar()
tk.Button(background_frame, text="Select Background GIF", command=upload_gif).grid(row=0,column=0,sticky="nsew")
background_entry = tk.Entry(background_frame, textvariable=background_gif_var, font="Medium")
background_entry.grid(row=1,column=0,sticky="nsew")
background_preview = tk.Label(adapter_parameters_frame,image=adapter_tk_image)
background_preview.grid(row=2,column=1,sticky="nsew")

# Debug Mode
debug_var = tk.BooleanVar()
tk.Checkbutton(buttons_frame, variable=debug_var, text="Enable Debug ?",font="Medium").grid(row=0,column=0,sticky="nsew")
# Start Buttons
tk.Button(buttons_frame, text="Standard FXs",command=classic_handler,font="Large").grid(row=0,column=1,sticky="nsew")
tk.Button(buttons_frame, text="Adapter FXs",command=adapter_handler,font="Large").grid(row=0,column=2,sticky="nsew")
tk.Button(buttons_frame, text="Background FXs",command=background_handler,font="Large").grid(row=0,column=3,sticky="nsew")
tk.Button(buttons_frame, text="Perspective FXs",command=perspective_handler,font="Large").grid(row=0,column=4,sticky="nsew")

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

print(threading.enumerate())