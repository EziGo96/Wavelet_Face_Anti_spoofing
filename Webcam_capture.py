'''
Created on 23-Apr-2023

@author: EZIGO
'''
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import model_from_json
import tensorflow_addons as tfa
from utils import WT
# Load your pre-trained Keras model

json_file = open("FASNetSE.json", 'r')
loaded_model_json = json_file.read()
json_file.close()

custom_objects = {'AdaptiveAveragePooling2D': tfa.layers.AdaptiveAveragePooling2D}
model = model_from_json(loaded_model_json, 
                        # custom_objects = custom_objects,
                        )

model.load_weights("FASNetSE.hdf5")

# def crop_center(img, crop_size):
#     h, w = img.shape[:2]
#     y, x = (h - crop_size[1]) // 2, (w - crop_size[0]) // 2
#     return img[y:y + crop_size[1], x:x + crop_size[0]]
#
# def preprocess(img):
#     # Crop out a patch from the center of the image with a 3:4 aspect ratio
#     h, w = img.shape[:2]
#     aspect_ratio = 4 / 3
#     crop_width = int(min(w, h / aspect_ratio))
#     crop_height = int(min(h, w * aspect_ratio))
#     cropped_img = crop_center(img, (crop_width, crop_height))
#
#     # Rescale the crop to 80x80
#     resized_img = cv2.resize(cropped_img, (80, 80))
#
#     # Normalize and expand dimensions for model input
#     normalized_img = resized_img / 255.0
#     return np.expand_dims(normalized_img, axis=0)
#
# cap = cv2.VideoCapture(0)
#
# while True:
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#
#     # Preprocess the frame
#     input_img = preprocess(frame)
#
#     # Pass the preprocessed frame to the Keras model
#     prediction= model.predict(input_img)
#     pred_list=["Fake","Real","Fake"]
#
#     # Display the live output (customize according to your model)
#     output_text = f"Prediction: {pred_list[np.argmax(prediction)]}"
#     cv2.putText(frame, output_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#
#     # Display the resulting frame
#     cv2.imshow('Live Feed', frame)
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the capture and close windows
# cap.release()
# cv2.destroyAllWindows()

def crop_center(img, crop_size):
    h, w = img.shape[:2]
    y, x = (h - crop_size[1]) // 2, (w - crop_size[0]) // 2
    return img[y:y + crop_size[1], x:x + crop_size[0]]

def preprocess(img):
    # Crop out a patch from the center of the image with a 3:4 aspect ratio
    h, w = img.shape[:2]
    aspect_ratio = 3 / 4
    crop_width = int(min(w, h / aspect_ratio))
    crop_height = int(min(h, w * aspect_ratio))
    cropped_img = crop_center(img, (crop_width, crop_height))

    # Rescale the crop to 80x80
    resized_img = cv2.resize(cropped_img, (80, 80))

    # Normalize and expand dimensions for model input
    normalized_img = resized_img / 255.0
    return np.expand_dims(normalized_img, axis=0), resized_img

cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Preprocess the frame
    input_img, resized_img = preprocess(frame)

    WT_img=WT(np.array([resized_img]),name="db14",concat=1).predict((np.array([resized_img])))[0]*255
    # Pass the preprocessed frame to the Keras model
    prediction = model.predict(input_img)
    pred_list=["Real","Real","Fake"]
    # Display the live output (customize according to your model)
    output_text = f"Prediction: {pred_list[np.argmax(prediction)]}"
    cv2.putText(frame, output_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Combine the original frame and the resized image
    h, w = frame.shape[:2]
    combined_img = np.zeros((h, w + 80, 3), dtype=np.uint8)
    combined_img[:h, :w] = frame
    combined_img[:80, w:] = cv2.cvtColor(WT_img, cv2.COLOR_GRAY2BGR)
    # Display the resulting frame
    cv2.imshow('Live Feed', combined_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()