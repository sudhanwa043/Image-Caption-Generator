import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
import argparse
from pickle import load
from keras.models import load_model
from keras.applications.xception import Xception
import cv2


def extract_features(filename, model):
        # try:
        image = Image.open(filename)

        # except:
            # print("ERROR: Couldn't open image! Make sure the image path and extension is correct")
        image = image.resize((299,299))
        image = np.array(image)
        # for images that has 4 channels, we convert them into 3 channels
        if image.shape[2] == 4: 
            image = image[..., :3]
        image = np.expand_dims(image, axis=0)    
        image = image/127.5
        image = image - 1.0
        feature = model.predict(image)
        return feature

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_desc(model, tokenizer, photo, max_length):
    in_text = 'start'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        pred = model.predict([photo,sequence], verbose=0)
        pred = np.argmax(pred)
        word = word_for_id(pred, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'end':
            break
    return in_text


#path = 'Flicker8k_Dataset/111537222_07e56d5a30.jpg'
max_length = 32
tokenizer = load(open("tokenizer.p","rb"))
model = load_model('models/model_9.h5')
xception_model = Xception(include_top=False, pooling="avg")

# img_path = r"C:\Users\LENOVO\Desktop\11.jpg"

img_path = r"Sample Images\1.jpg"
photo = extract_features(img_path, xception_model)
img = Image.open(img_path)

description = generate_desc(model, tokenizer, photo, max_length)
print("\n\n")

des = description.split(' ')
n=len(des)
des1=""
for i in range (1,n-1):
    des1 = des1 + des[i].capitalize() + " "

description = des1

img1 = cv2.imread(img_path)

h,w,c = img1.shape   
print("Updated Width and Height:", w,"x", h)

cy1 = int(h*9/10)
cy2 = int(h/10)
cx = int(w/25)
cv2.putText(img1, "Caption: "+ description, (cx, cy1), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 2)
cv2.putText(img1, "Caption: "+ description, (cx, cy2), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,255,0), 2)

cv2.imshow("image", img1)
cv2.waitKey(0)

print(description)
