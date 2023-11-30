# import matplotlib.pyplot as plt
# from PIL import Image
# import numpy as np
# from scipy.spatial.distance import cosine
# from mtcnn.mtcnn import MTCNN
# import sys
# from keras.applications.imagenet_utils import preprocess_input
# import tensorflow as tf

# import time
# import os
# import glob
# from mtcnn.mtcnn import MTCNN
# # from vggface import VGGFace
# from keras_vggface import VGGFace
# import openpyxl

# # Function to extract face from an image
# def extract_face(filename, required_size=(224, 224)):
#     img = plt.imread(filename)
#     detector = MTCNN()
#     results = detector.detect_faces(img)
    
#     if len(results) == 0:
#         print(f"No faces found in the image: {filename}")
#         return None
    
#     x1, y1, width, height = results[0]['box']
#     x2, y2 = x1 + width, y1 + height
#     # Extract the face
#     face = img[y1:y2, x1:x2]
    
#     image = Image.fromarray(face)
#     image = image.resize(required_size)
#     face_array = np.asarray(image)
    
#     return face_array




# # Function to get face embeddings from a list of photo files
# def get_embeddings(filenames):
#     # Extract faces
#     faces = [extract_face(f) for f in filenames]
    
#     # Convert into an array of samples
#     samples = np.asarray(faces, 'float32')
    
#     # Prepare the face for the model, e.g., center pixels
#     # samples = preprocess_input(samples, version=2)
#     samples = tf.keras.applications.mobilenet.preprocess_input(samples)


#     model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
#     pred = model.predict(samples)
#     return pred



# # Function to determine if a candidate face is a match for a known face
# def is_match(known_embedding, candidate_embedding, thresh=0.5):
#     # Calculate the distance between embeddings
#     score = cosine(known_embedding, candidate_embedding)
#     return score

# def main():
#     input_folder = "D:\\capstone\\organized_images_601"  # Change this to the folder containing your images
#     excel_filename = "D:\\capstone\\Bald_Yes_Arched.xlsx"

#     # Get a list of image files in the input folder
# #     image_files = glob.glob(os.path.join(input_folder, "*"))
#     image_files = sorted(glob.glob(os.path.join(input_folder, "*")))

#     # Create an Excel workbook
#     workbook = openpyxl.Workbook()
#     worksheet = workbook.active

#     batch_size = 6000  # Adjust this value based on your available memory

#     for i in range(0, len(image_files), batch_size):
#         batch_filenames = image_files[i:i+batch_size]
#         batch_embeddings = get_embeddings(batch_filenames)

#         for j, img1 in enumerate(batch_filenames):
#             for k, img2 in enumerate(batch_filenames):
#                 embeddings = batch_embeddings[j], batch_embeddings[k]
#                 score = is_match(embeddings[0], embeddings[1])

#                 # Append the data to the Excel worksheet
#                 worksheet.append([os.path.basename(img1), os.path.basename(img2), 100 * (1 - score)])
#                 print("Done with", os.path.basename(img1), os.path.basename(img2), i, j)
# #             i+=1

#     # Save the Excel file
#     workbook.save(excel_filename)
#     print(f"Similarity scores saved to {excel_filename}")

# if __name__ == '__main__':
#     t1=time.time()
#     main()
#     t2=time.time()
#     print(t2-t1)


import time
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
import sys
# from utils import preprocess_input



import os
import glob
import numpy as np
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
# from utils import preprocess_input
from keras.applications.vgg16 import preprocess_input
# from tensorflow.keras.applications.vgg16 import preprocess_input
from vggface import VGGFace
import openpyxl
from PIL import Image
import matplotlib.pyplot as plt

# Function to extract face from an image
def extract_face(filename, required_size=(224, 224)):
    img = plt.imread(filename)
    detector = MTCNN()
    results = detector.detect_faces(img)
    
    if len(results) == 0:
        print(f"No faces found in the image: {filename}")
        return None
    
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # Extract the face
    face = img[y1:y2, x1:x2]
    
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = np.asarray(image)
    
    return face_array




# Function to get face embeddings from a list of photo files
def get_embeddings(filenames):
    # Extract faces
    faces = [extract_face(f) for f in filenames]
    
    # Convert into an array of samples
    samples = np.asarray(faces, 'float32')
    
    # Prepare the face for the model, e.g., center pixels
    samples = preprocess_input(samples)

    model = VGGFace(architecture='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    pred = model.predict(samples)
    return pred



# Function to determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # Calculate the distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    return score

def main():
    input_folder = "D:\\capstone\\organized_images_757"  # Change this to the folder containing your images
    excel_filename = "D:\\capstone\\Bald_Yes_Arched.xlsx"

    # Get a list of image files in the input folder
    image_files = sorted(glob.glob(os.path.join(input_folder, "*")))

    # Create an Excel workbook
    workbook = openpyxl.Workbook()
    worksheet = workbook.active
    worksheet.append(["Image 1", "Image 2", "Similarity Score"])

    # Batch processing parameters
    batch_size = 6000  # Adjust this value based on your available memory

    for i in range(0, len(image_files), batch_size):
        batch_filenames = image_files[i:i+batch_size]
        batch_embeddings = get_embeddings(batch_filenames)

        for j, img1 in enumerate(batch_filenames):
            for k, img2 in enumerate(batch_filenames):
                embeddings = batch_embeddings[j], batch_embeddings[k]
                score = is_match(embeddings[0], embeddings[1])

                # Append the data to the Excel worksheet
                worksheet.append([os.path.basename(img1), os.path.basename(img2), 100 * (1 - score)])
                print("Done with", os.path.basename(img1), os.path.basename(img2), i, j)

    # Save the Excel file
    workbook.save(excel_filename)
    print(f"Similarity scores saved to {excel_filename}")

if __name__ == '__main__':
    t1=time.time()
    main()
    t2=time.time()
    print(t2-t1)