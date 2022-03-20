from __future__ import print_function
#%matplotlib inline
import argparse
import os
import random


import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from tqdm import tqdm
from datetime import date
import os
import os.path
from Models import Generator,Discriminator
import pickle
import torchvision
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


## Hyper Parameters
manualSeed = 999
random.seed(manualSeed)
torch.manual_seed(manualSeed)
BATCH_SIZE = 32
DATALOADER_WORKER = 1
SHUFFLE_DATASET = True
IMG_SIZE = 256
NGPU = 4
NUM_EPOCHS = 10000
lr = 2e-4
save_key = "real_image"
folder_key = "5"
beta1 = 0.5
is_resume = False
resume_path_gen = "/common/home/ssa162/CS536/Project1/weights/4/real_image/gen/gen_epoch_11.pth"
resume_path_disc = "/common/home/ssa162/CS536/Project1/weights/4/real_image/disc/disc_epoch_11.pth"
starting_epoch = 0

save_folders = [f"weights/{folder_key}/{save_key}/gen/",f"weights/{folder_key}/{save_key}/disc/"]
for dir in save_folders:
    if not os.path.exists(dir):
        os.makedirs(dir)


transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),]
    )
training_data = dset.ImageFolder(
    root="pizza/realpizza",    
    transform=transform
)
dataloader = torch.utils.data.DataLoader(training_data,batch_size=BATCH_SIZE, shuffle=SHUFFLE_DATASET, num_workers=DATALOADER_WORKER)
device = torch.device("cuda:0" if (torch.cuda.is_available() and NGPU > 0) else "cpu")

gen = Generator().to(device)
if (device.type == 'cuda') and (NGPU > 1):
    gen = nn.DataParallel(gen, list(range(NGPU)))
disc = Discriminator().to(device)
if (device.type == 'cuda') and (NGPU > 1):
    disc = nn.DataParallel(disc, list(range(NGPU)))

if is_resume:
    gen.load_state_dict(torch.load(resume_path_gen))
    disc.load_state_dict(torch.load(resume_path_disc))
    print("Resuming training from epoch {}".format(starting_epoch))
else:   
    gen.apply(weights_init)
    disc.apply(weights_init)

criterion = nn.MSELoss()
real_label = 1
fake_label = 0
optimizer_disc = optim.Adam(disc.parameters(),lr=lr,betas=(beta1,0.999))
optimizer_gen = optim.Adam(gen.parameters(),lr=lr,betas=(beta1,0.999))

img_list = []
G_losses = []
D_losses = []
iters = 0
fixed_noise = torch.randn(BATCH_SIZE, 3, 256, 256, device=device)
with torch.no_grad():
    fake = gen(fixed_noise).detach().cpu()
img_grid = vutils.make_grid(fake, padding=2, normalize=True)
img_list.append(img_grid)
torchvision.utils.save_image(img_grid,f"weights/{folder_key}/{save_key}/gen/img_epoch_{0}.png")
for epoch in range(starting_epoch+1,NUM_EPOCHS):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):       
       
        real_img = data[0].to(device)
        b_size = real_img.size(0)
        real_label = torch.full((b_size, 1,30,30), 1, device=device, dtype=torch.float32)
        fake_label = torch.full((b_size, 1,30,30), 0, device=device, dtype=torch.float32)
        noise = torch.randn(b_size, 3, 256, 256, device=device)       
        
        
        # Train the discriminator
        optimizer_disc.zero_grad()
        real_d_output = disc(real_img)
        loss_d_real = criterion(real_d_output,real_label)
        fake_d_output = disc(gen(noise).detach())
        loss_d_fake = criterion(fake_d_output,fake_label)
        
        loss_d = (loss_d_real+loss_d_fake)/2
        loss_d.backward()
        optimizer_disc.step()

        # Train the generator
        optimizer_gen.zero_grad()
        fake_img = gen(noise)
        fake_g_output = disc(fake_img)
        loss_g = criterion(fake_g_output,real_label)
        loss_g.backward()
        optimizer_gen.step()
        #Output training stats
        if i % 10 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\t'
                  % (epoch, NUM_EPOCHS, i, len(dataloader),
                     loss_d.item(), loss_g.item()))

        # Save Losses for plotting later
        G_losses.append(loss_g.item())
        D_losses.append(loss_d.item())

        iters += 1
    with torch.no_grad():
        fake = gen(fixed_noise).detach().cpu()
    img_grid = vutils.make_grid(fake, padding=2, normalize=True)
    img_list.append(img_grid)
    if epoch % 10 == 0:        
        torchvision.utils.save_image(img_grid,f"weights/{folder_key}/{save_key}/gen/img_epoch_{epoch+1}.png")
        torch.save(gen.state_dict(), f"weights/{folder_key}/{save_key}/gen/gen_epoch_{epoch+1}.pth")
        torch.save(disc.state_dict(), f"weights/{folder_key}/{save_key}/disc/disc_epoch_{epoch+1}.pth")

torch.save(gen.state_dict(), f"weights/{folder_key}/{save_key}/gen/gen_final.pth")
torch.save(disc.state_dict(), f"weights/{folder_key}/{save_key}/disc/disc_epoch_final.pth")

with open('mypickle.pickle', 'wb') as f:
    pickle.dump(img_list, f)