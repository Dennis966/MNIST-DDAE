import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torch.nn import MSELoss
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, random_split
from dset import NoisyCleanTrainSet, NoisyCleanTestSet
from torch.utils.tensorboard import SummaryWriter
import os

# Notes regarding initial commit: poor train val split, no save load model, no tensorboard
# I tried doing "git commit --amend" in Pycharm, and I could not escape the --insert-- mode
# by pressing "esc". ChatGPT couldn't solve this either.


# setup DDAE configurations
# We should? know a priori what the input dimension of our data is? For MNIST, it is 28*28=784.
# Ans: If we use nn.Linear, then we should. If it is convolution layer, then we don't have to.

# ChatGPT says this ensures a more reproducible implementation
# Note that ChatGPT says "full" reproducibility is still not always guaranteed,
# things like cuDNN backend acceleration could still potentially introduce randomness
# beyond the control of PyTorch.
torch.manual_seed(0)
class Encoder(nn.Module):                                             # FCN
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.Conv2d(1, 30, kernel_size=3)          # 784 --enc--> (28-3+1)*(28-3+1)=26*26
        self.maxpool_1 = nn.MaxPool2d(kernel_size=3, stride=1)  # 26*26 --pool--> (26-3+1)*(26-3+1)=24*24

        self.layer_2 = nn.Conv2d(30, 30, kernel_size=3)         # 24*24 --enc--> 22*22
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=1)  # 22*22 --pool--> 20*20

        self.layer_3 = nn.Conv2d(30, 30, kernel_size=3)         # 20*20 --enc--> 18*18
        self.maxpool_3 = nn.MaxPool2d(kernel_size=3, stride=1)  # 18*18 --pool--> 16*16

    def forward(self, x):
        enc_1 = self.maxpool_1(F.relu(self.layer_1(x)))
        enc_2 = self.maxpool_2(F.relu(self.layer_2(enc_1)))
        enc_3 = self.maxpool_3(F.relu(self.layer_3(enc_2)))
        #print("enc_3.size() = {}".format(enc_3.size()))

        return enc_3

class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Conv2d(30, 30, kernel_size=1)      # 16*16 --> (16-1+1)*(16-1+1)=16*16
        self.layer_2 = nn.Conv2d(30, 30, kernel_size=1)      # 16*16 --> (16-1+1)*(16-1+1)=16*16
        self.layer_3 = nn.Conv2d(30, 30, kernel_size=1)      # 16*16 --> (16-1+1)*(16-1+1)=16*16

    def forward(self, x):
        btl_1 = F.relu(self.layer_1(x))
        btl_2 = F.relu(self.layer_2(btl_1))
        btl_3 = F.relu(self.layer_3(btl_2))

        return btl_3


class Decoder(nn.Module):                                             # FCN
    def __init__(self):
        super().__init__()

        self.layer_1 = nn.ConvTranspose2d(30, 30, kernel_size=3)     # 16*16 --dec--> 18*18
        self.layer_2 = nn.ConvTranspose2d(30, 30, kernel_size=3)     # 18*18 --dec--> 20*20
        self.layer_3 = nn.ConvTranspose2d(30, 30, kernel_size=3)     # 20*20 --dec--> 22*22
        self.layer_4 = nn.ConvTranspose2d(30, 30, kernel_size=3)     # 22*22 --dec--> 24*24
        self.layer_5 = nn.ConvTranspose2d(30, 30, kernel_size=3)     # 24*24 --dec--> 26*26
        self.layer_6 = nn.ConvTranspose2d(30, 1, kernel_size=3)      # 26*26 --dec--> 28*28


    def forward(self, x):
        dec_1 = F.relu(self.layer_1(x))
        dec_2 = F.relu(self.layer_2(dec_1))
        dec_3 = F.relu(self.layer_3(dec_2))
        dec_4 = F.relu(self.layer_4(dec_3))
        dec_5 = F.relu(self.layer_5(dec_4))
        dec_6 = F.relu(self.layer_6(dec_5))

        return dec_6

class DDAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.bottleneck = Bottleneck()
        self.decoder = Decoder()

    def forward(self, x):
        enc_3 = self.encoder(x)
        bottleneck_out = self.bottleneck(enc_3)
        dec_3 = self.decoder(bottleneck_out)

        return dec_3

# Set device to GPU if possible
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: {}".format(device))

# Prepare train & test dataset.
total_train_dataset = NoisyCleanTrainSet()              # total training samples
# specify the train_size, val_size
total_train_size = len(total_train_dataset)
train_size = int(0.8 * total_train_size)
val_size = total_train_size - train_size

# train, validation split
train_dataset, val_dataset = random_split(total_train_dataset, [train_size, val_size])
test_dataset = NoisyCleanTestSet()

"""
Common practice: Check the dataset attributes, including the datatype.
"""
#print("len(train_dataset) = {}".format(len(train_dataset)))
#print("type(train_dataset[0][0]) = {}".format(type(train_dataset[0][0])))  #  <class 'numpy.ndarray'>
#print("type(train_dataset[0][1]) = {}".format(type(train_dataset[0][1])))  #  <class 'numpy.ndarray'>
print("train_dataset[0][0].shape = {}".format(train_dataset[0][0].shape))
print("train_dataset[0][1].shape = {}".format(train_dataset[0][1].shape))

# 3 main players:
#    1. Someone that samples from a dataset: DataLoader
#    2. The NN architecture
#    3. The optimizer that tunes the parameters of the NN

# Data sampler: DataLoader
batch_size = 5
train_dataloader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

val_dataloader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False)                # suggested by ChatGPT

test_dataloader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)               # suggested by ChatGPT
# set up the NN structure
model = DDAE().to(device)      # set model on device

"""
Common bug 1: (Datatype incompatibility)
# RuntimeError: Input type (double) and bias type (float) should be the same
"""
model = model.double()           # convert the datatype of model to be of type: torch.float64
#for param in model.parameters():
#    print(param.dtype)           # torch.float64

"""
Common bug 2: (Dimension mismatch)
UserWarning: Using a target size (torch.Siz
e([1, 28, 28])) that is different to the input size (torch.Size([1, 20, 20])). This will likely lead to incorrect results due to broadcasting. Pleas
e ensure they have the same size.
"""
#print(model)

# Common practice 1: Send a random input, check the output
# Common practice 2: print(enc.size()) in forward method
random_input = torch.rand(5, 1, 28, 28, dtype=torch.double).to(device)
random_input = random_input
random_output = model(random_input)
print("random_output.size() = {}".format(random_output.size()))

# Instantiate the loss, gradient calculators
criterion = MSELoss()

# Instantiate the optimizer
learning_rate = 0.01
optimizer = opt.Adam(model.parameters(), lr=learning_rate)

# Create tensorboard
writer = SummaryWriter()
#------------------------------------ Training ------------------------------------
num_epochs = 2
train_loss_tracker = []
val_loss_tracker = []

# Load a previously trained model if it exists
if os.path.exists("./checkpoint.pth"):
    checkpoint = torch.load("./checkpoint.pth")
    model.load_state_dict(checkpoint["model state"])
    optimizer.load_state_dict(checkpoint["optimizer state"])
    prev_epoch = checkpoint["epoch"]
    train_loss_tracker = checkpoint["train losses"]
    val_loss_tracker = checkpoint["val losses"]
    print("-------- Checkpoints loaded -------- ")
else:
    prev_epoch = 0
    print("-------- No previous Checkpoints --------")


for epoch in range(prev_epoch + 1, num_epochs + 1):                                 # for each epoch
    total_train_loss_per_epoch = 0
    total_val_loss_per_epoch = 0

    # train
    model.train()                                                     # reset to train mode
    for batch_idx, train_data in enumerate(train_dataloader):     # for each mini-batch of data
        # training loss
        x = train_data[0].to(device)              # set input tensor to device
        #print("x.size() = {}".format(x.size()))   # x.size() = torch.Size([5, 28, 28])
        x = torch.unsqueeze(x, 1)
        #print("x new.size() = {}".format(x.size()))
        y = train_data[1].to(device)              # set output tensor to device
        #print("y.size() = {}".format(y.size()))   # x.size() = torch.Size([5, 28, 28])
        y = torch.unsqueeze(y, 1)
        #print("y new.size() = {}".format(y.size()))
        NN_output = model(x)
        train_loss = criterion(NN_output, y)
        #print("train_data[0].dtype = {}".format(train_data[0].dtype))
        #print("train_data[1].dtype = {}".format(train_data[1].dtype))
        #print("train_loss.dtype = {}".format(train_loss.dtype))
        print("Batch Number {}: training loss = {}".format(batch_idx, train_loss.item()))
        total_train_loss_per_epoch += train_loss.item()         # call .item() for the numerical value

        optimizer.zero_grad()                                   # clear gradients from previous train_loss.backward()
        train_loss.backward()                                   # calculate gradients on current mini-batch
        optimizer.step()                                        # update parameters of NN

    # validation loss (After each epoch training, validate the newly updated model)
    model.eval()                                # set to eval mode
    with torch.no_grad():                       # Within the context manager: deactivate the grad attribute in the following tensors
        for batch_idx, val_data in enumerate(val_dataloader):
            x = val_data[0].to(device)
            x = torch.unsqueeze(x, 1)
            y = val_data[1].to(device)
            y = torch.unsqueeze(y, 1)
            NN_output = model(x)
            val_loss = criterion(NN_output, y)
            print("Batch Number {}: validation loss = {}".format(batch_idx, val_loss.item()))
            total_val_loss_per_epoch += val_loss.item()


    # print Epoch training and validation MSE
    average_train_loss_per_epoch = total_train_loss_per_epoch/(len(train_dataloader)/batch_size)
    train_loss_tracker.append(average_train_loss_per_epoch)
    print("Epoch {} Training MSE = {}".format(epoch, average_train_loss_per_epoch))

    average_val_loss_per_epoch = total_val_loss_per_epoch / (len(val_dataloader) / batch_size)
    val_loss_tracker.append(average_val_loss_per_epoch)
    print("Epoch {} Validation MSE = {}".format(epoch, average_val_loss_per_epoch))

    writer.add_scalar("Training Loss (MSE)", average_train_loss_per_epoch, epoch)
    writer.add_scalar("Validation Loss (MSE)", average_val_loss_per_epoch, epoch)

    # save checkpoint
    checkpoint = {"model state": model.state_dict(),
                  "optimizer state": optimizer.state_dict(),
                  "epoch": epoch,
                  "train losses": train_loss_tracker,
                  "val losses": val_loss_tracker}
    torch.save(checkpoint, "checkpoint.pth")

writer.close()

#------------------------------------ Testing ------------------------------------
model.eval()                             # set to eval mode
with torch.no_grad():
    total_test_loss = 0
    for batch_idx, test_data in enumerate(test_dataloader):
        x = test_data[0].to(device)
        x = torch.unsqueeze(x, 1)
        y = test_data[1].to(device)
        y = torch.unsqueeze(y, 1)
        NN_output = model(x)
        #print("NN_output.size {}".format(NN_output.size()))
        #print("NN_output[1].size {}".format(NN_output[1].size()))
        if batch_idx == 30:
            print("sample result")
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1)
            # Move from GPU->CPU, detach from computation graph so
            # no gradient calculations will be influenced.
            x_hat = torch.squeeze(x[1], 0)
            x_hat = x_hat.cpu().detach().numpy()

            y_hat = torch.squeeze(NN_output[1], 0)
            y_hat = y_hat.cpu().detach().numpy()

            y_ref_hat = torch.squeeze(y[1], 0)
            y_ref_hat = y_ref_hat.cpu().detach().numpy()

            ax1.imshow(x_hat, cmap='gray')
            ax1.set_title("Noisy")
            #ax1.colorbar()

            ax2.imshow(y_hat, cmap='gray')
            ax2.set_title("Enhanced ({} epochs)".format(num_epochs))
            #ax2.colorbar()

            ax3.imshow(y_ref_hat, cmap='gray')
            ax3.set_title("Clean Reference".format(num_epochs))
            #ax3.colorbar()

            #plt.savefig()
            plt.tight_layout()
            plt.show()


        batch_test_loss = criterion(NN_output, y)

        total_test_loss += batch_test_loss.item()
    average_test_loss = total_test_loss/(len(test_dataloader)/batch_size)
    print("Testing MSE = {}".format(average_test_loss))






