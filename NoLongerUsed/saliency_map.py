import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16 as Model
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend as K
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils import normalize
from vis.utils import utils
import numpy as np
from matplotlib import cm
from PIL import Image
import os

def generate_saliency_map(input_path, output_path):
    # Load model
    model = Model(weights='imagenet', include_top=True)
    
    # Load image
    img = load_img(input_path, target_size=(224, 224))
    img_copy = img.copy()
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    img = preprocess_input(img)
    
    # Make prediction
    yhat = model.predict(img)
    label = decode_predictions(yhat)
    top_class_index = np.argmax(yhat)  # Get the index of the top prediction
    
    # Modify model for visualization
    layer_idx = utils.find_layer_idx(model, 'predictions')
    model.layers[-1].activation = tf.keras.activations.linear
    model = utils.apply_modifications(model)
    
    # Create score object
    score = CategoricalScore([top_class_index])
    
    # Create Saliency object
    saliency = Saliency(model, clone=False)
    
    # Generate saliency map
    saliency_map = saliency(score, img, smooth_samples=20)
    saliency_map = normalize(saliency_map)
    
    # Convert saliency map to RGB image
    saliency_img = np.uint8(cm.jet(saliency_map[0])[..., :3] * 255)
    saliency_img = Image.fromarray(saliency_img)
    
    # Resize saliency image to match original image size
    saliency_img = saliency_img.resize((img_copy.size[0], img_copy.size[1]))
    
    # Save saliency image
    saliency_img.save(output_path, format='JPEG')

if __name__ == '__main__':
    input_image_path = '../images/cat.jpg'  # Change this to your input image path
    output_image_path = '../images/cat_saliency.jpg'  # Change this to your desired output path
    generate_saliency_map(input_image_path, output_image_path)