from main import *
from CNN_RNN import *
import os

model = define_model(vocab_size, max_length)
epochs = 10

steps = len(train_descriptions)
os.mkdir('models')

for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
    model.fit_generator(generator, epochs = 1, steps_per_epoch = steps, verbose=1)
    model.save("models/model_"+str(i)+".h5")