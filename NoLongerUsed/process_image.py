import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16 as Model
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend as K
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore
from vis.utils import utils
import numpy as np
from matplotlib import cm
from PIL import Image
import os

def process_image(input_path, output_path):
    model = Model(weights='imagenet', include_top=True)
    img = load_img(input_path, target_size=(224, 224))
    img_copy = img.copy()
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    img = preprocess_input(img)
    yhat = model.predict(img)
    label = decode_predictions(yhat)
    top_class_index = np.argmax(yhat)  # Get the index of the top prediction

    layer_idx = utils.find_layer_idx(model, 'predictions')
    model.layers[-1].activation = tf.keras.activations.linear
    model = utils.apply_modifications(model)

    score = CategoricalScore([top_class_index])  # Use the top class index dynamically
    gradcam = Gradcam(model, clone=True)
    cam = gradcam(score, img, penultimate_layer=-1)
    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)  # Remove alpha channel

    heatmap_img = Image.fromarray(heatmap)
    heatmap_img = heatmap_img.resize((img_copy.size[0], img_copy.size[1]))

    if img_copy.mode != 'RGB':
        img_copy = img_copy.convert('RGB')

    overlay = Image.blend(img_copy, heatmap_img, alpha=0.5)
    overlay.save(output_path, format='JPEG')

