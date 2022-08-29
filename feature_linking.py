from Cleaning_sorting import load_doc
from pickle import load

def load_photos(filename):
    file = load_doc(filename)
    photos = file.split('\n')[:-1]
    return photos

def load_clean_descriptions(filename, photos):
    file = load_doc(filename)
    descriptions = {}
    
    for line in file.split('\n'):
        words = line.split()
        
        if len(words)<1:
            continue
            
        image, image_caption = words[0], words[1:]
        
        if image in photos:
            if image not in descriptions:
                descriptions[image]=[]
            desc = '<start> ' + " ".join(image_caption) + ' <end>'
            descriptions[image].append(desc)
    
    return descriptions

def load_features(photos):
    all_features = load(open("features.p", "rb"))
    features = {k:all_features[k] for k in photos} #key value pair
    return features
