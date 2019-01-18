import os
import pickle

for f in os.listdir('data/history'):
    with open(f, 'rb') as file:
        history = pickle.load(file)

