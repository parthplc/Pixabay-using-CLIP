import torch
import clip
from PIL import Image
import torch
import clip
from PIL import Image
import os
import re
from tqdm import tqdm, trange
import random
import requests
import numpy as np
import streamlit as st
global model, preprocess, device
device = 'cpu'
model, preprocess = clip.load("ViT-B/32", device = 'cpu')

images = ['1.jpg', '2.jpg', '3.jpg','11.jpg']

text = "an elephant in the desert"
simScore = []
tokenizedText = clip.tokenize(text).to(device)
path = "./data"
for img in images:
    image = preprocess(Image.open(os.path.join(path, img))).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(tokenizedText)

    # append image name with similarity score
    simScore.append((img, torch.matmul(text_features, image_features.T)[0][0]))

print(simScore)