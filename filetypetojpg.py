from PIL import Image
import os

# Directory containing the input images
input_directory = "D:\\capstone\\archive\\front\\front"  # Replace with the path to your input directory
output_directory = "D:\\capstone\\archivejpg\\front"  # Replace with the desired output directory

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Iterate through all files in the input directory
for filename in os.listdir(input_directory):
    #if filename.endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp",".jfif",".*")):
        input_image_path = os.path.join(input_directory, filename)
        output_image_path = os.path.join(output_directory, filename.split('.')[0] + ".jpg")

        try:
            with Image.open(input_image_path) as img:
                # Convert and save the image as JPG
                img.convert("RGB").save(output_image_path, "JPEG")
            print(f"Converted: {filename}")
        except Exception as e:
            print(f"Error converting {filename}: {e}")

print("All images converted and saved as JPG.")
