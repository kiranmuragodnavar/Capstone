# from PIL import Image
# import os
# import random

# # Define paths
# input_path = "D:\\capstone\\tstimg"
# output_path = "D:\\capstone\\output_images_folder"

# # Create output directory if it doesn't exist
# if not os.path.exists(output_path):
#     os.makedirs(output_path)

# def apply_rotation(image, angle):
#     return image.rotate(angle, expand=True)

# def generate_images(image_path, total_images):
#     image = Image.open(image_path)
#     base_filename = os.path.splitext(os.path.basename(image_path))[0]

#     # Generate rotated images
#     rotation_angles = [0, 45, 90, 135, 180, 225, 270, 315]
#     num_rotations = len(rotation_angles)
#     images_per_rotation = total_images // (num_rotations + 1)

#     # Generate images with different pixel qualities
#     pixel_qualities = [50, 100, 150, 200, 250]

#     for quality in pixel_qualities:
#         image_quality = image.copy()
#         image_quality.save(
#             os.path.join(output_path, f"{base_filename}_quality_{quality}.jpg"),
#             quality=quality
#         )

#     # Save rotated images
#     for i, angle in enumerate(rotation_angles):
#         rotated_image = apply_rotation(image, angle)
#         rotated_image.save(
#             os.path.join(output_path, f"{base_filename}_rotated_{angle}.jpg"),
#             quality=95  # You can set the quality as you prefer
#         )

#         # Generate additional rotated images to reach the total count
#         for j in range(1, images_per_rotation):
#             random_angle = random.choice(rotation_angles)
#             new_rotated_image = apply_rotation(image, random_angle)
#             new_rotated_image.save(
#                 os.path.join(output_path, f"{base_filename}_rotated_{angle}_{j}.jpg"),
#                 quality=95
#             )

# # Generate images for each image in the input folder
# for image_filename in os.listdir(input_path):
#     if image_filename.endswith(".jpg") or image_filename.endswith(".jpeg"):
#         image_path = os.path.join(input_path, image_filename)
#         generate_images(image_path, total_images=250)

import cv2
import numpy as np
import os

# Input and output directories
input_dir = "D:\\capstone\\forehead\\broad"
output_dir = "D:\\capstone\\forehead\\brodfnl"
os.makedirs(output_dir, exist_ok=True)

# Load input images
input_images = os.listdir(input_dir)

# Define a list of image manipulation techniques
manipulation_techniques = [
    'grayscale',
    'blur',
    'rotate',
    # 'flip',
    'resize',
    'brightness',
    'contrast',
    # 'crop',
    'shear',
    'translate',
    'add_noise',
    # Add more techniques here
]

# Loop through each input image
for input_image_file in input_images:
    input_image_path = os.path.join(input_dir, input_image_file)
    image = cv2.imread(input_image_path)
    
    # Apply each manipulation technique
    for technique in manipulation_techniques:
        for i in range(28):  # Generate 250 variations per technique
            manipulated_image = image.copy()
            
            if technique == 'grayscale':
                manipulated_image = cv2.cvtColor(manipulated_image, cv2.COLOR_BGR2GRAY,)
            elif technique == 'blur':
                kernel_size = (5, 5)
                manipulated_image = cv2.GaussianBlur(manipulated_image, kernel_size, 0)
            elif technique == 'rotate':
                angle = np.random.randint(0, 360)
                rotation_matrix = cv2.getRotationMatrix2D((image.shape[1] / 2, image.shape[0] / 2), angle, 1)
                manipulated_image = cv2.warpAffine(manipulated_image, rotation_matrix, (image.shape[1], image.shape[0]))
            elif technique == 'flip':
                flip_code = np.random.randint(-1, 2)  # Flip horizontally, vertically, or not at all
                manipulated_image = cv2.flip(manipulated_image, flip_code)
            elif technique == 'resize':
                scale_factor = np.random.uniform(0.5, 2.0)  # Resize between 50% and 200% of original size
                new_width = int(image.shape[1] * scale_factor)
                new_height = int(image.shape[0] * scale_factor)
                manipulated_image = cv2.resize(manipulated_image, (new_width, new_height))
            elif technique == 'brightness':
                brightness_factor = np.random.uniform(0.5, 1.5)  # Adjust brightness between 50% and 150%
                manipulated_image = cv2.convertScaleAbs(manipulated_image, alpha=brightness_factor, beta=0)
            elif technique == 'contrast':
                contrast_factor = np.random.uniform(0.5, 1.5)  # Adjust contrast between 50% and 150%
                manipulated_image = cv2.convertScaleAbs(manipulated_image, alpha=1.0, beta=(1.0 - contrast_factor))
            elif technique == 'shear':
                shear_factor = np.random.uniform(-0.3, 0.3)  # Shear between -0.3 and 0.3
                shear_matrix = np.array([[1, shear_factor, 0], [0, 1, 0]], dtype=np.float32)
                manipulated_image = cv2.warpAffine(manipulated_image, shear_matrix, (image.shape[1], image.shape[0]))
            elif technique == 'translate':
                max_translation = 50
                tx = np.random.randint(-max_translation, max_translation)
                ty = np.random.randint(-max_translation, max_translation)
                translation_matrix = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
                manipulated_image = cv2.warpAffine(manipulated_image, translation_matrix, (image.shape[1], image.shape[0]))
            elif technique == 'add_noise':
                noise = np.random.normal(0, 25, image.shape).astype(np.uint8)
                manipulated_image = cv2.add(manipulated_image, noise)
            
            # Save the manipulated image
            output_filename = f"{input_image_file.split('.')[0]}_{technique}_{i}.jpg"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, manipulated_image)
