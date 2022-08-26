'''
Created by dingyc.
2022.
'''

from dataclasses import replace
from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import device, optim
import numpy as np
import os
import struct
from tqdm import trange,tqdm
import time

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

epochs = 50
batch_size_train = 64
batch_size_test = 64
learning_rate = 0.001
momentum = 0.5

class PRE():
    '''
    class for pre process
    pre(): deal with the data
    next(): get a batch
    '''
    
    def __init__(self,inputdir,mode="train"):
        self.mode = mode
        self.inputdir = inputdir
        self.batch_index = 0
        if(mode=="test"):
            self.batch_size = batch_size_test
        else:
            self.batch_size = batch_size_train

        self.pre()

    def pre(self):
        '''
        pre process
        for mnist: inputdir: inputdir/xx-ubyte
        '''
        if(self.mode=="test"):
            labels_file = os.path.join(self.inputdir,"t10k-labels.idx1-ubyte")
            images_file = os.path.join(self.inputdir,"t10k-images.idx3-ubyte")
        else:
            labels_file = os.path.join(self.inputdir,"train-labels.idx1-ubyte")
            images_file = os.path.join(self.inputdir,"train-images.idx3-ubyte")

        with open(labels_file,"rb") as l_f:
            magic,n=struct.unpack(">II",l_f.read(8))
            labels = np.fromfile(l_f,dtype=np.uint8)
        with open(images_file,"rb") as i_f:
            magic,num,rows,cols=struct.unpack(">IIII",i_f.read(16))
            images = np.fromfile(i_f,dtype=np.uint8).reshape(len(labels), 28,28)
        
        self.x = images
        self.y = labels
        self.data_size = len(self.y)
        self.batchs_in_one_epoch = self.data_size//self.batch_size
        #when having a remainder, let the rest be one batch
        if(self.data_size%self.batch_size!=0):
            self.batchs_in_one_epoch+=1
    

    def next(self):
        '''
        return a batch_size of data and labels
        if an end of one epoch, return [],[]
        call it and check len(x)<batch_size means an end of one epoch
        if len(x)==0 then break; 
        if 0<len(x)<batch_size then deal with the remaining items and break
        '''
        if(self.batch_index==0):
            state = np.random.get_state()
            np.random.shuffle(self.x)
            np.random.set_state(state)
            np.random.shuffle(self.y)
        
        if(self.batch_index==self.batchs_in_one_epoch-1):
            x = self.x[self.batch_size*self.batch_index:]
            y = self.y[self.batch_size*self.batch_index:]
            self.batch_index = 0
        else:
            x = self.x[self.batch_size*self.batch_index:self.batch_size*(self.batch_index+1)]
            y = self.y[self.batch_size*self.batch_index:self.batch_size*(self.batch_index+1)]
            self.batch_index += 1
        
        return x,y
            

class CNN (nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
        self.con1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0,bias=False),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.con2 = nn.Sequential(
            nn.Conv2d(16, 16, 5, 1, 0,bias=False),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256, 100),
            nn.ReLU()
        )
        self.fc2 = nn.Linear(100, 10)


    def forward(self,x):            # (batch,  1, 28, 28)
        x = self.con1(x)            # (batch, 16, 12, 12)
        x = self.con2(x)            # (batch, 16,  4,  4)
        x = x.view(x.size(0), -1)   # (batch_size, 256)
        x = self.fc1(x)             # (batch_size, 100)
        x = self.fc2(x)             # (batch_size, 10)
        return x


def run(datadir,model,mode="train",run_epochs=epochs):
    '''
    run:
    datadir: data
    model: train mode is modeldir, ortherwise is modelfile
    mode: maybe "train", "continue_train" or "test"
    '''
    net = CNN()     
    optimizer = optim.SGD(net.parameters(),lr=learning_rate,momentum=momentum)
    if(mode!="train"):
        checkpoint = torch.load(model,map_location=device)  
        net.load_state_dict(checkpoint['net'])  
        optimizer.load_state_dict(checkpoint['optimizer'])
    net = net.to(device) 
    checkpoint = {
        "net": net.state_dict(),
        'optimizer':optimizer.state_dict()
    }
    #CrossEntropyLoss要输入的input是模型的分类结果，one-hot形式，batch_szie*n，target是直接的类别，n
    compute_loss=nn.CrossEntropyLoss()
    pre = PRE(datadir,mode)
    losses = []
    acces = []
    print_step = 100
    for e in range(run_epochs):
        if(mode!="test"):
            print(f"epoch [{e+1:>2}/{run_epochs}]")

        sum_loss = 0
        sum_acc = 0

        with tqdm(total=pre.batchs_in_one_epoch) as pbar:
            while(1):
                x,y = pre.next()
                pbar.update(1)
                if(len(x)==0):
                    break

                #x: batch_size*28*28 --> batch_size*1*28*28
                x = torch.tensor(np.float32(x)).unsqueeze(1)
                y = torch.tensor(y)
                x,y = x.to(device),y.to(device)
                out = net(x)
                loss = compute_loss(out+1e-8,y.long())
                
                #train needs backward calculation
                if(mode!="test"):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                sum_loss += loss.item()*y.shape[0]
                _,pred = out.max(1)
                num_correct = (pred == y).sum().item()
                acc = num_correct/y.shape[0]
                sum_acc += num_correct
                
                #show in process bar
                pbar.set_description(f"{mode:^14}")
                pbar.set_postfix(loss=f"{loss.item():8.4f}",acc=f"{acc:8.4f}")

                if(len(x)<pre.batch_size):
                    break
        
        losses.append(sum_loss/pre.data_size)
        acces.append(sum_acc/pre.data_size)
        
        print(f"{mode:^14}: loss={losses[-1]:8.4f}, acc={acces[-1]:8.4f}")
        
        #train needs save models
        if(mode!="test"):
            if(not os.path.exists(model)):
                os.makedirs(model)
            modelpath = f"{model}/mnist_net{e+1}.pth"
            if(mode=="continue_train"):
                modelpath = f"{modelfile[:-4]}+{e+1}.pth"
            torch.save(checkpoint, modelpath)

        #test only needs one epoch
        if(mode=="test"):
            break

        run(datadir,modelpath,mode="test")

def rewrite_model_params(model):
    '''
    rewrite the parameters to C/C++ double type as an '.pt' file
    '''
    net = CNN()     
    checkpoint = torch.load(model,map_location=device)  
    net.load_state_dict(checkpoint['net'])  
    out_model = torch.jit.trace(net, torch.ones(1, 1, 28, 28).to(device))
    torch.jit.save(out_model,model.replace(".pth",".pt"))
    




if __name__ =="__main__":
    inputdir = "D:/vscode/workspace/python/PPML/dataset/mnist_dataset"
    modeldir = "D:/vscode/workspace/python/PPML/models/cnn_mnist"
    modelfile = "D:/vscode/workspace/python/PPML/models/cnn_mnist/mnist_net50.pth"
    #"train", "continue_train", "test"
    # run(inputdir,modeldir,mode="train")
    # print("done")
    run(inputdir,modelfile,mode="continue_train",run_epochs=50)
    # print("done")
    # run(inputdir,modelfile,mode="test")
    # print("done")
    # rewrite_model_params(modelfile)
    # print("done")
