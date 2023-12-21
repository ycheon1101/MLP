# return image data frame and crop_size
import table_images
from mlp import MLP
import torch.nn as nn
import torch.optim as optim
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from pathlib import Path
import sys
# path
modules_path = Path(__file__).parent
src_path = modules_path.parent

# append path
sys.path.append(src_path)


# check device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# print(device)

# params
learning_rate = 5e-3
num_epochs = 300

# calc loss
criterion = nn.MSELoss()

# instanciation model
model = MLP(in_feature=2, hidden_feature=32, hidden_layers=8, out_feature=3).to(device)


# optimizer
optimizer = optim.Adam(model.parameters(), lr=learning_rate) 

# set target
img_df, crop_size = table_images.make_table()

# train model
def train_model(target, xy_flatten):
    for epoch in range(num_epochs):

        # generate
        generated = model(xy_flatten)

        # loss = criterion(generated, target)
        loss = criterion(generated, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
    

# plot
def plot_img(xy_flatten):
    # reshape the generated tensor to [h, w, c]
    generated_reshape = model(xy_flatten) 
    generated_reshape *= 255.0
    generated_reshape = torch.reshape(generated_reshape, (crop_size, crop_size, 3))

    # change to numpy for plot image
    # generated_reshape = generated_reshape.detach().numpy()
    generated_reshape = generated_reshape.cpu().detach().numpy()


    # save image
    save_img = Image.fromarray(generated_reshape.astype(np.uint8))
    save_img_path = Path(src_path) / 'images' / 'tested_images'
    save_img_filename = save_img_path / 'pure_mlp_test_img.jpg'
    save_img.save(save_img_filename)

# main
def main():

    for image in range(1, len(img_df.index) + 1):
        target = img_df['img_flatten'][image].to(device)
        xy_flatten = img_df['xy_flatten'][image].to(device)

        train_model(target, xy_flatten)
        plot_img(xy_flatten)

main()
            
