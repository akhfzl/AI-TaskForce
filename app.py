# Copy this code below, modified this code into add function twibbon (the object inside twibbon must be in the middle) & add new model removal background (if you can't find any, add it into choice none removal background)

import os, re
import zipfile
import shutil
import time
from PIL import Image, ImageDraw
import io
from rembg import remove
import gradio as gr
from concurrent.futures import ThreadPoolExecutor
from transformers import pipeline
from PIL import Image, ImageOps
import numpy as np
import json
import torch
import os

def remove_background_rembg(input_path):
    print(f"Removing background using rembg for image: {input_path}")
    with open(input_path, 'rb') as i:
        input_image = i.read()
    output_image = remove(input_image)
    img = Image.open(io.BytesIO(output_image)).convert("RGBA")
    return img

def remove_background_bria(input_path):
    print(f"Removing background using bria for image: {input_path}")
    device = 0 if torch.cuda.is_available() else -1

    # Load the segmentation model
    pipe = pipeline("image-segmentation", model="briaai/RMBG-1.4", trust_remote_code=True, device=device)

    # Process the image
    result = pipe(input_path)
    return result

def get_bounding_box_with_threshold(image, threshold):
	# Convert image to numpy array
    img_array = np.array(image)

    # Get alpha channel
    alpha = img_array[:,:,3]

    # Find rows and columns where alpha > threshold
    rows = np.any(alpha > threshold, axis=1)
    cols = np.any(alpha > threshold, axis=0)

    # Find the bounding box
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]

    if left < right and top < bottom:
        return (left, top, right, bottom)
    else:
        return None

def position_logic(image_path, canvas_size, padding_top, padding_right, padding_bottom, padding_left, use_threshold=True):
    image = Image.open(image_path)
    image = image.convert("RGBA")

    # Get the bounding box of the non-blank area with threshold
    if use_threshold:
        bbox = get_bounding_box_with_threshold(image, threshold=10)
    else:
        bbox = image.getbbox()
    log = []

    if bbox:
        # Check 1 pixel around the image for non-transparent pixels
        width, height = image.size
        cropped_sides = []

        # Define tolerance for transparency
        tolerance = 30  # Adjust this value as needed

        # Check top edge
        if any(image.getpixel((x, 0))[3] > tolerance for x in range(width)):
            cropped_sides.append("top")

        # Check bottom edge
        if any(image.getpixel((x, height-1))[3] > tolerance for x in range(width)):
            cropped_sides.append("bottom")

        # Check left edge
        if any(image.getpixel((0, y))[3] > tolerance for y in range(height)):
            cropped_sides.append("left")

        # Check right edge
        if any(image.getpixel((width-1, y))[3] > tolerance for y in range(height)):
            cropped_sides.append("right")

        if cropped_sides:
            info_message = f"Info for {os.path.basename(image_path)}: The following sides of the image may contain cropped objects: {', '.join(cropped_sides)}"
            print(info_message)
            log.append({"info": info_message})
        else:
            info_message = f"Info for {os.path.basename(image_path)}: The image is not cropped."
            print(info_message)
            log.append({"info": info_message})

        # Crop the image to the bounding box
        image = image.crop(bbox)
        log.append({"action": "crop", "bbox": [str(bbox[0]), str(bbox[1]), str(bbox[2]), str(bbox[3])]})

        # Calculate the new size to expand the image
        target_width, target_height = canvas_size
        aspect_ratio = image.width / image.height

        if len(cropped_sides) == 4:
            # If the image is cropped on all sides, center crop it to fit the canvas
            if aspect_ratio > 1:  # Landscape
                new_height = target_height
                new_width = int(new_height * aspect_ratio)
                left = (new_width - target_width) // 2
                image = image.resize((new_width, new_height), Image.LANCZOS)
                image = image.crop((left, 0, left + target_width, target_height))
            else:  # Portrait or square
                new_width = target_width
                new_height = int(new_width / aspect_ratio)
                top = (new_height - target_height) // 2
                image = image.resize((new_width, new_height), Image.LANCZOS)
                image = image.crop((0, top, target_width, top + target_height))
            log.append({"action": "center_crop_resize", "new_size": f"{target_width}x{target_height}"})
            x, y = 0, 0
        elif not cropped_sides:
            # If the image is not cropped, expand it from center until it touches the padding
            new_height = target_height - padding_top - padding_bottom
            new_width = int(new_height * aspect_ratio)
        
            if new_width > target_width - padding_left - padding_right:
                # If width exceeds available space, adjust based on width
                new_width = target_width - padding_left - padding_right
                new_height = int(new_width / aspect_ratio)
        
            # Resize the image
            image = image.resize((new_width, new_height), Image.LANCZOS)
            log.append({"action": "resize", "new_width": str(new_width), "new_height": str(new_height)})
        
            x = (target_width - new_width) // 2
            y = target_height - new_height - padding_bottom
        else:
            # New logic for handling cropped top and left, or top and right
            if set(cropped_sides) == {"top", "left"} or set(cropped_sides) == {"top", "right"}:
                new_height = target_height - padding_bottom
                new_width = int(new_height * aspect_ratio)
            
                # If new width exceeds canvas width, adjust based on width
                if new_width > target_width:
                    new_width = target_width
                    new_height = int(new_width / aspect_ratio)
            
                # Resize the image
                image = image.resize((new_width, new_height), Image.LANCZOS)
                log.append({"action": "resize", "new_width": str(new_width), "new_height": str(new_height)})
            
                # Set position
                if "left" in cropped_sides:
                    x = 0
                else:  # right in cropped_sides
                    x = target_width - new_width
                y = 0
            
                # If the resized image is taller than the canvas minus padding, crop from the bottom
                if new_height > target_height - padding_bottom:
                    crop_bottom = new_height - (target_height - padding_bottom)
                    image = image.crop((0, 0, new_width, new_height - crop_bottom))
                    new_height = target_height - padding_bottom
                    log.append({"action": "crop_vertical", "bottom_pixels_removed": str(crop_bottom)})
            
                log.append({"action": "position", "x": str(x), "y": str(y)})
            elif set(cropped_sides) == {"bottom", "left"} or set(cropped_sides) == {"bottom", "right"}:
                # Handle bottom & left or bottom & right cropped images
                new_height = target_height - padding_top
                new_width = int(new_height * aspect_ratio)
            
                # If new width exceeds canvas width, adjust based on width
                if new_width > target_width - padding_left - padding_right:
                    new_width = target_width - padding_left - padding_right
                    new_height = int(new_width / aspect_ratio)
            
                # Resize the image without cropping or stretching
                image = image.resize((new_width, new_height), Image.LANCZOS)
                log.append({"action": "resize", "new_width": str(new_width), "new_height": str(new_height)})
            
                # Set position
                if "left" in cropped_sides:
                    x = 0
                else:  # right in cropped_sides
                    x = target_width - new_width
                y = target_height - new_height
            
                log.append({"action": "position", "x": str(x), "y": str(y)})
            elif set(cropped_sides) == {"bottom", "left", "right"}:
                # Expand the image from the center
                new_width = target_width
                new_height = int(new_width / aspect_ratio)
            
                if new_height < target_height:
                    new_height = target_height
                    new_width = int(new_height * aspect_ratio)
            
                image = image.resize((new_width, new_height), Image.LANCZOS)
            
                # Crop to fit the canvas
                left = (new_width - target_width) // 2
                top = 0
                image = image.crop((left, top, left + target_width, top + target_height))
            
                log.append({"action": "expand_and_crop", "new_size": f"{target_width}x{target_height}"})
                x, y = 0, 0
            elif cropped_sides == ["top"]:
                # New logic for handling only top-cropped images
                if image.width > image.height:
                    new_width = target_width
                    new_height = int(target_width / aspect_ratio)
                else:
                    new_height = target_height - padding_bottom
                    new_width = int(new_height * aspect_ratio)
            
                # Resize the image
                image = image.resize((new_width, new_height), Image.LANCZOS)
                log.append({"action": "resize", "new_width": str(new_width), "new_height": str(new_height)})
            
                x = (target_width - new_width) // 2
                y = 0  # Align to top
            
                # Apply padding only to non-cropped sides
                x = max(padding_left, min(x, target_width - new_width - padding_right))
            elif cropped_sides in [["right"], ["left"]]:
                # New logic for handling only right-cropped or left-cropped images
                if image.width > image.height:
                    new_width = target_width - max(padding_left, padding_right)
                    new_height = int(new_width / aspect_ratio)
                else:
                    new_height = target_height - padding_top - padding_bottom
                    new_width = int(new_height * aspect_ratio)
            
                # Resize the image
                image = image.resize((new_width, new_height), Image.LANCZOS)
                log.append({"action": "resize", "new_width": str(new_width), "new_height": str(new_height)})
            
                if cropped_sides == ["right"]:
                    x = target_width - new_width  # Align to right
                else:  # cropped_sides == ["left"]
                    x = 0  # Align to left
                y = target_height - new_height - padding_bottom  # Respect bottom padding
            
                # Ensure top padding is respected
                if y < padding_top:
                    y = padding_top
                
                log.append({"action": "position", "x": str(x), "y": str(y)})
            elif set(cropped_sides) == {"left", "right"}:
                # Logic for handling images cropped on both left and right sides
                new_width = target_width  # Expand to full width of canvas
            
                # Calculate the aspect ratio of the original image
                aspect_ratio = image.width / image.height
            
                # Calculate the new height while maintaining aspect ratio
                new_height = int(new_width / aspect_ratio)
            
                # Resize the image
                image = image.resize((new_width, new_height), Image.LANCZOS)
                log.append({"action": "resize", "new_width": str(new_width), "new_height": str(new_height)})
            
                # Set horizontal position (always 0 as it spans full width)
                x = 0
            
                # Calculate vertical position to respect bottom padding
                y = target_height - new_height - padding_bottom
            
                # If the resized image is taller than the canvas, crop from the top only
                if new_height > target_height - padding_bottom:
                    crop_top = new_height - (target_height - padding_bottom)
                    image = image.crop((0, crop_top, new_width, new_height))
                    new_height = target_height - padding_bottom
                    y = 0
                    log.append({"action": "crop_vertical", "top_pixels_removed": str(crop_top)})
                else:
                    # Align the image to the bottom with padding
                    y = target_height - new_height - padding_bottom
            
                log.append({"action": "position", "x": str(x), "y": str(y)})
            elif cropped_sides == ["bottom"]:
                # Logic for handling images cropped on the bottom side
                # Calculate the aspect ratio of the original image
                aspect_ratio = image.width / image.height
            
                if aspect_ratio < 1:  # Portrait orientation
                    new_height = target_height - padding_top  # Full height with top padding
                    new_width = int(new_height * aspect_ratio)
                
                    # If the new width exceeds the canvas width, adjust it
                    if new_width > target_width:
                        new_width = target_width
                        new_height = int(new_width / aspect_ratio)
                else:  # Landscape orientation
                    new_width = target_width - padding_left - padding_right
                    new_height = int(new_width / aspect_ratio)
                
                    # If the new height exceeds the canvas height, adjust it
                    if new_height > target_height:
                        new_height = target_height
                        new_width = int(new_height * aspect_ratio)
            
                # Resize the image
                image = image.resize((new_width, new_height), Image.LANCZOS)
                log.append({"action": "resize", "new_width": str(new_width), "new_height": str(new_height)})
            
                # Set horizontal position (centered)
                x = (target_width - new_width) // 2
            
                # Set vertical position (touching bottom edge for all cases)
                y = target_height - new_height
            
                log.append({"action": "position", "x": str(x), "y": str(y)})
            else:
                # Use the original resizing logic for other partially cropped images
                if image.width > image.height:
                    new_width = target_width
                    new_height = int(target_width / aspect_ratio)
                else:
                    new_height = target_height
                    new_width = int(target_height * aspect_ratio)
            
                # Resize the image
                image = image.resize((new_width, new_height), Image.LANCZOS)
                log.append({"action": "resize", "new_width": str(new_width), "new_height": str(new_height)})
            
                # Center horizontally for all images
                x = (target_width - new_width) // 2
                y = target_height - new_height - padding_bottom
            
                # Adjust positions for cropped sides
                if "top" in cropped_sides:
                    y = 0
                elif "bottom" in cropped_sides:
                    y = target_height - new_height
                if "left" in cropped_sides:
                    x = 0
                elif "right" in cropped_sides:
                    x = target_width - new_width
            
                # Apply padding only to non-cropped sides, but keep horizontal centering
                if "left" not in cropped_sides and "right" not in cropped_sides:
                    x = (target_width - new_width) // 2  # Always center horizontally
                if "top" not in cropped_sides and "bottom" not in cropped_sides:
                    y = max(padding_top, min(y, target_height - new_height - padding_bottom))

    return log, image, x, y

def process_single_image(image_path, output_folder, bg_method, canvas_size_name, output_format, bg_choice, custom_color, watermark_path=None):
    add_padding_line = False

    if canvas_size_name == 'Rox':
        canvas_size = (1080, 1080)
        padding_top = 112
        padding_right = 125
        padding_bottom = 116
        padding_left = 125
    elif canvas_size_name == 'Columbia':
        canvas_size = (730, 610)
        padding_top = 30
        padding_right = 105
        padding_bottom = 35
        padding_left = 105
    elif canvas_size_name == 'Zalora':
        canvas_size = (763, 1100)
        padding_top = 50
        padding_right = 50
        padding_bottom = 200
        padding_left = 50


    filename = os.path.basename(image_path)
    try:
        print(f"Processing image: {filename}")
        if bg_method == 'rembg':
            image_with_no_bg = remove_background_rembg(image_path)
        elif bg_method == 'bria':
            image_with_no_bg = remove_background_bria(image_path)
        elif bg_method == None:
            image_with_no_bg = Image.open(image_path)
        
        temp_image_path = os.path.join(output_folder, f"temp_{filename}")
        image_with_no_bg.save(temp_image_path, format='PNG')

        log, new_image, x, y = position_logic(temp_image_path, canvas_size, padding_top, padding_right, padding_bottom, padding_left)

        # Create a new canvas with the appropriate background
        if bg_choice == 'white':
            canvas = Image.new("RGBA", canvas_size, "WHITE")
        elif bg_choice == 'custom':
            canvas = Image.new("RGBA", canvas_size, custom_color)
        else:  # transparent
            canvas = Image.new("RGBA", canvas_size, (0, 0, 0, 0))

        # Paste the resized image onto the canvas
        canvas.paste(new_image, (x, y), new_image)
        log.append({"action": "paste", "position": [str(x), str(y)]})

        # Add visible black line for padding when background is not transparent
        if add_padding_line:
            draw = ImageDraw.Draw(canvas)
            draw.rectangle([padding_left, padding_top, canvas_size[0] - padding_right, canvas_size[1] - padding_bottom], outline="black", width=5)
            log.append({"action": "add_padding_line"})

        output_ext = 'jpg' if output_format == 'JPG' else 'png'
        output_filename = f"{os.path.splitext(filename)[0]}.{output_ext}"
        output_path = os.path.join(output_folder, output_filename)

        # Apply watermark only if the filename ends with "_01" and watermark_path is provided
        if os.path.splitext(filename)[0].endswith("_01") and watermark_path:
            watermark = Image.open(watermark_path).convert("RGBA")
            canvas = canvas.convert("RGBA")
            canvas.paste(watermark, (0, 0), watermark)
            log.append({"action": "add_watermark"})

        if output_format == 'JPG':
            canvas = canvas.convert('RGB')
            canvas.save(output_path, format='JPEG')
        else:
            canvas.save(output_path, format='PNG')

        # if os.path.splitext(filename)[0].endswith("_01") and watermark_path:
        #     with Image.open(watermark_path) as im:
        #         canvas = canvas.convert("RGBA")
        #         canvas.paste(im, (0, 0), im)
        #         log.append({"action": "add_watermark"})
        #         if im.mode in ('RGBA', 'P'):
        #             canvas = canvas.convert('RGB')

        #             canvas.save(output_path, format="JPEG")
        #         elif im.mode in ('RGB', 'JPEG'):
        #             canvas.save(output_path, format='JPEG')
        #         else: 
        #             canvas.save(output_path, format='PNG')

        os.remove(temp_image_path)

        print(f"Processed image path: {output_path}")
        return [(output_path, image_path)], log

    except Exception as e:
        print(f"Error processing {filename}: {e}")
        return None, None

def remove_extension(filename):
    # Regular expression to match any extension at the end of the string
    return re.sub(r'\.[^.]+$', '', filename)

def process_images(input_files, bg_method='rembg', watermark_path=None, canvas_size='Rox', output_format='PNG', bg_choice='transparent', custom_color="#ffffff", num_workers=4, progress=gr.Progress()):
    start_time = time.time()

    output_folder = "processed_images"
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    os.makedirs(output_folder)

    processed_images = []
    original_images = []
    all_logs = []

    if isinstance(input_files, str) and input_files.lower().endswith(('.zip', '.rar')):
        # Handle zip file
        input_folder = "temp_input"
        if os.path.exists(input_folder):
            shutil.rmtree(input_folder)
        os.makedirs(input_folder)

        try:
            with zipfile.ZipFile(input_files, 'r') as zip_ref:
                zip_ref.extractall(input_folder)
        except zipfile.BadZipFile as e:
            print(f"Error extracting zip file: {e}")
            return [], None, 0

        image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp'))]
    elif isinstance(input_files, list):
        # Handle multiple files
        image_files = input_files
    else:
        # Handle single file
        image_files = [input_files]

    total_images = len(image_files)
    print(f"Total images to process: {total_images}")

    avg_processing_time = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_image = {executor.submit(process_single_image, image_path, output_folder, bg_method, canvas_size, output_format, bg_choice, custom_color, watermark_path): image_path for image_path in image_files}
        for idx, future in enumerate(future_to_image):
            try:
                start_time_image = time.time()
                result, log = future.result()
                end_time_image = time.time()
                image_processing_time = end_time_image - start_time_image
            
                # Update average processing time
                avg_processing_time = (avg_processing_time * idx + image_processing_time) / (idx + 1)
                if result:
                    if watermark_path:
                        get_name = future_to_image[future].split('/')
                        get_name = remove_extension(get_name[len(get_name)-1])
                        twibbon_input = f'{get_name}.png' if output_format == 'PNG' else f'{get_name}.jpg'
                        twibbon_output_path = os.path.join(output_folder, f'result_{start_time_image}.png')
                        add_twibbon(f'processed_images/{twibbon_input}', watermark_path, twibbon_output_path)
                        processed_images.append((twibbon_output_path, twibbon_output_path)) 
                    else: 
                        processed_images.extend(result)
                    original_images.append(future_to_image[future])
                    all_logs.append({os.path.basename(future_to_image[future]): log})
            
                # Estimate remaining time
                remaining_images = total_images - (idx + 1)
                estimated_remaining_time = remaining_images * avg_processing_time
            
                progress((idx + 1) / total_images, f"{idx + 1}/{total_images} images processed. Estimated time remaining: {estimated_remaining_time:.2f} seconds")
            except Exception as e:
                print(f"Error processing image {future_to_image[future]}: {e}")

    output_zip_path = "processed_images.zip"
    with zipfile.ZipFile(output_zip_path, 'w') as zipf:
        for file, _ in processed_images:
            zipf.write(file, os.path.basename(file))

    # Write the comprehensive log for all images
    with open(os.path.join(output_folder, 'process_log.json'), 'w') as log_file:
        json.dump(all_logs, log_file, indent=4)
    print("Comprehensive log saved to", os.path.join(output_folder, 'process_log.json'))

    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Processing time: {processing_time} seconds")
    return original_images, processed_images, output_zip_path, processing_time

def gradio_interface(input_files, bg_method, watermark, canvas_size, output_format, bg_choice, custom_color, num_workers):
    progress = gr.Progress()
    watermark_path = watermark.name if watermark else None

    # Check input_files, is it single image, list image, or zip/rar
    if isinstance(input_files, str) and input_files.lower().endswith(('.zip', '.rar')):
            return process_images(input_files, bg_method, watermark_path, canvas_size, output_format, bg_choice, custom_color, num_workers, progress)
    elif isinstance(input_files, list):
        return process_images(input_files, bg_method, watermark_path, canvas_size, output_format, bg_choice, custom_color, num_workers, progress)
    else:
        return process_images(input_files.name, bg_method, watermark_path, canvas_size, output_format, bg_choice, custom_color, num_workers, progress)

def show_color_picker(bg_choice):
    if bg_choice == 'custom':
        return gr.update(visible=True)
    return gr.update(visible=False)

def update_compare(evt: gr.SelectData):
    if isinstance(evt.value, dict) and 'caption' in evt.value:
        input_path = evt.value['caption']
        output_path = evt.value['image']['path']
        input_path = input_path.split("Input: ")[-1]
        # Open the original and processed images
        original_img = Image.open(input_path)
        processed_img = Image.open(output_path)
        
        # Calculate the aspect ratios
        original_ratio = f"{original_img.width}x{original_img.height}"
        processed_ratio = f"{processed_img.width}x{processed_img.height}"
        
        return gr.update(value=input_path), gr.update(value=output_path), gr.update(value=original_ratio), gr.update(value=processed_ratio)
    else:
        print("No caption found in selection")
        return gr.update(value=None), gr.update(value=None), gr.update(value=None), gr.update(value=None)

def process(input_files, bg_method, watermark, canvas_size, output_format, bg_choice, custom_color, num_workers):
	_, processed_images, zip_path, time_taken = gradio_interface(input_files, bg_method, watermark, canvas_size, output_format, bg_choice, custom_color, num_workers)
	processed_images_with_captions = [(img, f"Input: {caption}") for img, caption in processed_images]
	return processed_images_with_captions, zip_path, f"{time_taken:.2f} seconds"

def add_twibbon(image_path, twibbon_path, output_path):
    """
    Adds a twibbon (frame) to an image and centers the original image inside the twibbon.
    
    Parameters:
    image_path (str): The path to the original image.
    twibbon_path (str): The path to the twibbon image.
    output_path (str): The path where the output image with twibbon will be saved.
    """
    # Open the original image and the twibbon
    image = Image.open(image_path)
    twibbon = Image.open(twibbon_path)

    # Get the sizes of both images
    image_width, image_height = image.size
    twibbon_width, twibbon_height = twibbon.size

    # Resize the original image to fit inside the twibbon (optional: resize by aspect ratio)
    aspect_ratio = image_width / image_height
    if twibbon_width / twibbon_height > aspect_ratio:
        new_width = twibbon_width
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = twibbon_height
        new_width = int(new_height * aspect_ratio)

    image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Center the image within the twibbon
    x_offset = (twibbon_width - new_width) // 2
    y_offset = (twibbon_height - new_height) // 2
    combined_image = Image.new('RGBA', (twibbon_width, twibbon_height))
    combined_image.paste(image, (x_offset, y_offset))
    combined_image.paste(twibbon, (0, 0), mask=twibbon)  # Twibbon is pasted over the image

    # Save the result
    combined_image.save(output_path)
    return combined_image

def process_twibbon(image, twibbon):
    output_path = "output_image.png"  # Output sementara
    combined_image = add_twibbon(image.name, twibbon.name, output_path)
    return combined_image

def remove_background(image_path, method="none"):
    image = Image.open(image_path)
    
    if method == "none":
        return image  # Return the original image without any background removal
    elif method == "rembg":
        image = remove_background_rembg(image_path)
    elif method == "bria":
        image = remove_background_bria(image_path)
    
    return image  # Default return in case no valid method is chosen

with gr.Blocks(theme="NoCrypt/miku@1.2.2") as iface:
    gr.Markdown("# Image Background Removal and Resizing with Optional Watermark")
    gr.Markdown("Choose to upload multiple images or a ZIP/RAR file, select the crop mode, optionally upload a watermark image, and choose the output format.")

    with gr.Row():
        input_files = gr.File(label="Upload Image or ZIP/RAR file", file_types=[".zip", ".rar", "image"], interactive=True)
        watermark = gr.File(label="Upload Watermark Image (Optional)", file_types=[".png"])

    with gr.Row():
        canvas_size = gr.Radio(choices=["Rox", "Columbia", "Zalora"], label="Canvas Size", value="Rox")
        output_format = gr.Radio(choices=["PNG", "JPG"], label="Output Format", value="JPG")
        num_workers = gr.Slider(minimum=1, maximum=16, step=1, label="Number of Workers", value=5)

    with gr.Row():
        bg_method = gr.Radio(choices=["bria", "rembg", None], label="Background Removal Method", value="bria")
        bg_choice = gr.Radio(choices=["transparent", "white", "custom"], label="Background Choice", value="white")
        custom_color = gr.ColorPicker(label="Custom Background Color", value="#ffffff", visible=False)

    process_button = gr.Button("Process Images")

    with gr.Row():
        gallery_processed = gr.Gallery(label="Processed Images")
    with gr.Row():
        image_original = gr.Image(label="Original Images", interactive=False)
        image_processed = gr.Image(label="Processed Images", interactive=False)
    with gr.Row():
        original_ratio = gr.Textbox(label="Original Ratio")
        processed_ratio = gr.Textbox(label="Processed Ratio")
    with gr.Row():
        output_zip = gr.File(label="Download Processed Images as ZIP")
        processing_time = gr.Textbox(label="Processing Time (seconds)")

    bg_choice.change(show_color_picker, inputs=bg_choice, outputs=custom_color)
    process_button.click(process, inputs=[input_files, bg_method, watermark, canvas_size, output_format, bg_choice, custom_color, num_workers], outputs=[gallery_processed, output_zip, processing_time])
    gallery_processed.select(update_compare, outputs=[image_original, image_processed, original_ratio, processed_ratio])

iface.launch(share=True)


# add_twibbon('original_image.png', 'twibbon.jpg', 'output_image_with_twibbon.png')
# remove_background('image.png', method="none")
