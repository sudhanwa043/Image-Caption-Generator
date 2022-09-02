from Cleaning_sorting import *
from feature_linking import *
from pickle import load
from Tokenizing import *
from generator import *

dataset_text = "D:\Caption_P2\Flickr8k_text"
dataset_images = "D:\Caption_P2\Flickr8k_Dataset"

filename = "D:\Caption_P2\Flickr8k_text\Flickr8k.token.txt"

descriptions = all_img_captions(filename)
print("Length of descriptions is: ", len(descriptions))

clean_descriptions = cleaning_text(descriptions)

vocabulary = text_vocabulary(clean_descriptions)
print("Length of vocabulary: ", len(vocabulary))

save_descriptions(clean_descriptions, "descriptions.txt")
print('Cleaned Descriptions Saved Successfully!')

###################################################################

filename = "D:\Caption_P2\Flickr8k_text\Flickr_8k.trainImages.txt"
train_imgs = load_photos(filename)
train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
train_features = load_features(train_imgs)

print('Done feature linking')

###################################################################

tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.p', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)
print('Done Tokenising')

###################################################################

def max_length(descriptions):
    desc_list = dict_to_list(descriptions)
    return max(len(d.split()) for d in desc_list)
    
max_length = max_length(descriptions)

[a,b], c = next(data_generator(train_descriptions, train_features, tokenizer, max_length))
print('data generated')

###################################################################

