import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16 as Model
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import decode_predictions
from tf_keras_vis.utils import normalize
from vis.utils import utils
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.saliency import Saliency
from tensorflow.keras import backend as K
import json
from PIL import Image
import os
from matplotlib import cm


def predict(input_path, outpath_gradCAM, outpath_Saliency):
    model = Model(weights='imagenet', include_top=True)
    img = load_img(input_path, target_size=(224, 224))
    img_copy = img.copy()
    img = img_to_array(img)
    img = img.reshape(1, 224, 224, 3)
    img = preprocess_input(img)
    yhat = model.predict(img)
    class_idxs_sorted = np.argsort(yhat.flatten())[::-1]
    topNclass = 5
    CLASS_INDEX = json.load(open("static/imagenet_class_index.json"))
    classlabel = []
    for i_dict in range(len(CLASS_INDEX)):
        classlabel.append(CLASS_INDEX[str(i_dict)][1])
    predictions = []
    for i, idx in enumerate(class_idxs_sorted[:topNclass]):
        predictions.append({"class": classlabel[idx], "probability": float(yhat[0,idx])})
    
    top_class_index = np.argmax(yhat) 
    
    layer_idx = utils.find_layer_idx(model, 'predictions')
    model.layers[-1].activation = tf.keras.activations.linear
    model = utils.apply_modifications(model)
    score = CategoricalScore([top_class_index])

    #GradCAM
    gradcam = Gradcam(model, clone=True)
    cam = gradcam(score, img, penultimate_layer=-1)
    heatmap = np.uint8(cm.jet(cam[0])[..., :3] * 255)  # Remove alpha channel
    heatmap_img = Image.fromarray(heatmap)
    heatmap_img = heatmap_img.resize((img_copy.size[0], img_copy.size[1]))
    if img_copy.mode != 'RGB':
        img_copy = img_copy.convert('RGB')
    overlay = Image.blend(img_copy, heatmap_img, alpha=0.5)
    overlay.save(outpath_gradCAM, format='JPEG')

    #Saliency
    saliency = Saliency(model, clone=False)
    saliency_map = saliency(score, img, smooth_samples=20)
    saliency_map = normalize(saliency_map)
    saliency_img = np.uint8(cm.jet(saliency_map[0])[..., :3] * 255)
    saliency_img = Image.fromarray(saliency_img)
    saliency_img = saliency_img.resize((img_copy.size[0], img_copy.size[1]))
    saliency_img.save(outpath_Saliency, format='JPEG')

    return predictions
