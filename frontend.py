import streamlit as st
import os
from PIL import Image
import pandas as pd

excel_file="C:\\Users\\Lenovo\\Desktop\\Cap4000.xlsx"
df = pd.read_excel(excel_file)


# Function to get the folder path based on user inputs
def get_image_folder_path(hair, moustache, eyebrows):
    folder_name = f"{hair}_{moustache}_{eyebrows}"
    folder_path = os.path.join("Filtered Images", folder_name)
    return folder_path

# Function to display images in a grid with captions
def display_images_in_grid(image_folder,step1_results):
    # image_files = os.listdir(image_folder)
    num_images = len(step1_results)
    if num_images == 0:
        st.warning("No images found.")
        return

    # Determine the number of columns based on the number of images
    num_columns = min(num_images, 4)

    st.write(f"Displaying {num_images} images:")
    
    columns = st.columns(num_columns)
    for i, image_file in enumerate(step1_results):
        with columns[i % num_columns]:
            image_path = os.path.join(image_folder, image_file+".jpg")
            image = Image.open(image_path)
            st.image(image, use_column_width=True)
            st.caption(image_file)



def display_related_images(selected_folder, selected_image):
    excel_file = selected_folder + "_Excel.xlsx"
    if not os.path.exists(excel_file):
        st.warning("Data file not found.")
        return

    df = pd.read_excel(excel_file, header=None)  # Assuming no header row in Excel

    # Find the row corresponding to the selected image
    selected_row = df[df.iloc[:, 0] == selected_image]
    if not selected_row.empty:
        related_images = selected_row.iloc[0, 1:].dropna().tolist()
        st.header("Related Images:")

        # Display up to 30 related images in a 3-column grid
        related_folder = selected_folder

        num_images_to_display = min(len(related_images), 30)
        num_columns = min(num_images_to_display, 4)
        # temp = related_images[:num_images_to_display]
        related_imageee=related_images[:num_images_to_display]
        columns = st.columns(num_columns)
        for i, related_image in enumerate(related_imageee):
            with columns[i % num_columns]:
                image_path = os.path.join(related_folder, related_image)
                image = Image.open(image_path)
                st.image(image, use_column_width=True)
                st.caption(related_image)





# Streamlit UIimport streamlit as st

st.title("Facial Description Based Suspect Detection")

# Create two columns
st.title("Step - 1 : Image Refinement")
col1, col2 = st.columns(2)

# Column 1
with col1:
    eyes = st.selectbox("Select Eyes", ["Brown", "Blue", "Black", "Green"])
    eyebrows = st.selectbox("Select Eyebrows Style", ["Straight", "Arched"])

# Column 2
with col2:
    hair = st.selectbox("Select Hair Style", ["White", "Black", "Bald"])
    beard = st.selectbox("Select Beard Style", ["No", "Square", "Diamond", "Triangle", "Round"])

# Additional columns as needed
col3, col4 = st.columns(2)

with col3:
    face = st.selectbox("Select Face Shape", ["Round", "Square", "Oval"])
    lips = st.selectbox("Select Lips", ["Thin", "Full"])

with col4:
    moustache = st.selectbox("Select Moustache Style", ["Yes", "No"])
    forehead = st.selectbox("Select Forehead Shape", ["Broad", "M-Shaped", "Narrow"])

# Create another set of columns if needed
col5, col6 = st.columns(2)




step2_path = get_image_folder_path(hair,moustache,eyebrows)
folder_path="C:\\Users\\Lenovo\\Desktop\\Capstone\\Final Demo\\AllImages\\"


step1_results=[]
for index,row in df.iterrows():
    if row["Hair"]==hair and row["Beard"]==beard and row["Lips"]==lips and row["Moustache"]==moustache and row["Forehead"]==forehead and row["eyebrows"]==eyebrows:
        step1_results.append(row["ID"])



# st.write(f"Selected folder: {folder_path}")

if st.button("Show Images"):
    if len(step1_results):
        st.header("Images:")
        display_images_in_grid(folder_path,step1_results)
    else:
        st.error("No Match Found")


st.title("Step - 2 : Refinement based on Image Similarity")


selected_image = st.selectbox("Select an Image", os.listdir(folder_path))

if st.button("Show Related Images"):
    # Display related images from the Excel file
    display_related_images(step2_path, selected_image)





