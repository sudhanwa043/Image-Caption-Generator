import numpy as np
from PIL import Image
import os
from pickle import dump

from keras.applications.xception import Xception

from tqdm.notebook import tqdm
tqdm().pandas()

def extract_features(directory):
    model = Xception(include_top = False, pooling='avg')
    features = {}
    for img in tqdm(os.listdir(directory)):
        filename = directory + "/" + img
        image = Image.open(filename)
        image = image.resize((299,299))
        image = np.expand_dims(image, axis=0)
        image = image/127.5
        image = image - 1.0
        
        feature = model.predict(image)
        features[img] = feature
    return features

features = extract_features("D:\Caption_P2\Flickr8k_Dataset\Flicker8k_Dataset")
dump(features, open("features.p", "wb"))