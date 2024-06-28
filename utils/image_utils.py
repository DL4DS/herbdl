import os
import shutil
from PIL import Image, ImageOps

def get_file_size_in_mb(file_path):
    # Get the size of the file in bytes
    file_size_bytes = os.path.getsize(file_path)
    
    # Convert bytes to megabytes
    file_size_mb = file_size_bytes / (1024 * 1024)
    
    return file_size_mb


def resize_with_aspect_ratio(image_path, output_path, size=(1000, 1000)):
    # Open the original image

    with Image.open(image_path) as img:
        # skip is size is already 1000x1000
        if img.size == size:
            return False

        # Calculate the new size while maintaining the aspect ratio
        img.thumbnail(size, Image.ANTIALIAS)
        
        # Calculate the padding to make the image 1000x1000
        delta_w = size[0] - img.size[0]
        delta_h = size[1] - img.size[1]
        padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
        
        # Add padding to the image
        new_img = ImageOps.expand(img, padding)
        
        # Save the resized image
        new_img.save(output_path)

        return True
