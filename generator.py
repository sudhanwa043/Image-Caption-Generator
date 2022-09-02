from Tokenizing import dict_to_list
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np

#############################
vocab_size = 7577
#############################

def data_generator(descriptions, features, tokenizer, max_length):
    while 1:
        for key, description_list in descriptions.items():
            feature = features[key][0]
            input_image, input_sequence, output_word = create_sequences(tokenizer, max_length, description_list, feature)
            yield[[input_image, input_sequence], output_word]

def create_sequences(tokenizer, max_length, desc_list, feature):
    x1, x2, y = list(), list(), list()

    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]

        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen = max_length)[0]
            out_seq = to_categorical([out_seq], num_classes = vocab_size)[0]
            x1.append(feature)
            x2.append(in_seq)
            y.append(out_seq)
    return np.array(x1), np.array(x2), np.array(y)
