import pandas as pd
import shutil
import os

# Define the source folder with images and the destination folder for organization.
source_folder = "D:\\capstone\\4k\\4k"
destination_folder = "D:\\capstone\\organized_images_38"

# Load the Excel sheet with image ID and folder name.
excel_file = "D:\capstone\cap.xlsx"
df = pd.read_excel(excel_file)

# Iterate through each row in the Excel sheet and move images to the corresponding folders.
for index, row in df.iterrows():
    hair_value = str(row["Hair"])
    moustache_value = str(row["Moustache"])
    eyebrow_value = str(row["eyebrows"])

    if hair_value == "White" and moustache_value == "No" and eyebrow_value == "Straight":
        image_id = str(row["ID"])
        source_image_path = os.path.join(source_folder, image_id + ".jpg")
        destination_path = os.path.join(destination_folder, image_id + ".jpg")

        # Check if the source image exists and if the destination folder exists; create it if not.
        if os.path.exists(source_image_path):
            if not os.path.exists(destination_folder):
                os.makedirs(destination_folder)

            # Copy the image to the destination folder.
            print("Copying " + image_id + " to " + destination_folder)
            shutil.copy(source_image_path, destination_path)
        else:
            print(f"Image {image_id} not found in the source folder.")

print("Image organization completed.")
