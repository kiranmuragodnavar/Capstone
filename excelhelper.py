import pandas as pd

# Replace 'your_input_file.xlsx' with the actual name of your input Excel file
input_file = "C:\\Users\\Lenovo\\Desktop\\Capstone\\Final Demo\\Filtered Images\\Bald_Yes_Arched.xlsx"
output_file = "C:\\Users\\Lenovo\\Desktop\\Capstone\\Final Demo\\Filtered Images\\Bald_Yes_Arched_Excel.xlsx"
# "C:\Users\Lenovo\Desktop\Capstone\Final Demo\Filtered Images\Bald_Yes_Arched.xlsx"
# Read the original Excel file
df = pd.read_excel(input_file)
image_dict = {}

# Iterate through the rows and populate the dictionary
for index, row in df.iterrows():
    image1, image2, similarity = row['Image 1'], row['Image 2'], row['Similarity Score']

    # If image1 is not in the dictionary, add it
    if image1 not in image_dict:
        image_dict[image1] = []

    # If image2 is not in the dictionary, add it
    if image2 not in image_dict:
        image_dict[image2] = []

    # Add similarity to both images only if it's not already present
    image_dict[image1].append((image2, similarity))
    image_dict[image2].append((image1, similarity))

# Sort the similarities for each image based on the similarity scores
for key in image_dict:
    image_dict[key] = sorted(image_dict[key], key=lambda x: x[1], reverse=True)

unique_dict = {}

# Iterate through the data and remove duplicates for each key while preserving order
for key, values in image_dict.items():
    unique_values = []
    seen = set()

    for entry in values:
        if entry not in seen:
            unique_values.append(entry)
            seen.add(entry)

    unique_dict[key] = unique_values


# Create a new dictionary with the top similar images for each image
top_dict = {}
for image, similarities in unique_dict.items():
    num_similar_images = min(len(similarities), len(df) - 1)  # Ensure not to exceed the number of files
    top_dict[image] = [similar_image[0] for similar_image in similarities[:num_similar_images]]

# Create a DataFrame for the new Excel file
result_df = pd.DataFrame.from_dict(top_dict, orient='index').reset_index()
result_df.columns = ['Image'] + [f'Similar Image {i+1}' for i in range(len(result_df.columns) - 1)]

# Save the result to a new Excel file
result_df.to_excel(output_file, index=False)
print("Process Done")
