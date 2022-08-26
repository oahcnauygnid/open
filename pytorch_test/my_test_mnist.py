'''
Created by dingyc.
2022.
'''

from random import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torch import device, optim
import numpy as np
import os
import struct
from tqdm import trange,tqdm
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
#epoch [ˈiːpɒk]
epochs = 5
batch_size_train = 64
batch_size_test = 64
learning_rate = 0.001
momentum = 0.9

class MyDataset(Dataset):
    # 构造函数
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
    # 返回数据集大小
    def __len__(self):
        return self.data_tensor.size(0)
    # 返回索引的数据与标签
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]



def get_dataset(inputdir,mode,batch_size):
    '''
    get dataset, train or test
    '''
    #train data
    if(mode=="train"):
        images_file = os.path.join(inputdir,"train-images.idx3-ubyte")
        labels_file = os.path.join(inputdir,"train-labels.idx1-ubyte")
    else:
        images_file = os.path.join(inputdir,"t10k-images.idx3-ubyte")
        labels_file = os.path.join(inputdir,"t10k-labels.idx1-ubyte")
    with open(labels_file,"rb") as l_f:
        magic,n=struct.unpack(">II",l_f.read(8))
        labels = np.fromfile(l_f,dtype=np.uint8)
    with open(images_file,"rb") as i_f:
        magic,num,rows,cols=struct.unpack(">IIII",i_f.read(16))
        images = np.fromfile(i_f,dtype=np.uint8).reshape(len(labels), 28,28)    
    images = torch.tensor(np.float32(images)).unsqueeze(1)
    labels = torch.tensor(labels)
    dataset = MyDataset(images,labels)
    dataloader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
    
    return dataloader


class CNN (nn.Module):

    def __init__(self):
        super(CNN,self).__init__()
        self.con1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.con2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7*7, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self,x):            # (batch,  1, 28, 28)
        x = self.con1(x)            # (batch, 16, 14, 14)
        x = self.con2(x)            # (batch, 32,  7,  7)
        x = x.view(x.size(0), -1)   # (batch_size, 32*7*7)
        x = self.fc(x)             # (batch_size, 10)
        return x

def run(train_dataloader,test_dataloader,model_path,mode="train",run_epochs=epochs):
    '''
    run:
    train_dataloader,test_dataloader : train and test dataloader, when just test, train_dataloader could be ignored
    model_path: train mode is modeldir, ortherwise is modelfile
    mode: maybe "train", "continue_train" or "test"
    '''
    model = CNN()    
    optimizer = optim.SGD(model.parameters(),lr=learning_rate,momentum=momentum)
    loss_fn=nn.CrossEntropyLoss()
    
    if(mode!="train"):
        checkpoint = torch.load(model_path)  
        model.load_state_dict(checkpoint['net'])  
        optimizer.load_state_dict(checkpoint['optimizer'])
    model = model.to(device) 
    checkpoint = {
        "net": model.state_dict(),
        'optimizer':optimizer.state_dict()
    }

    losses = []
    acces = []
    print_step = 100
    if(mode=="test" or mode=="pred"):
        dataloader = test_dataloader
    else:
        dataloader = train_dataloader
    
    if(mode=="pred"):
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            print("predict result: ", pred.argmax(1).item())
            return

    data_size = len(dataloader.dataset)
    num_batches = len(dataloader)
    for e in range(run_epochs):
        if(mode=="train" or mode=="continue_train"):
            print(f"epoch [{e+1:>2}/{run_epochs}]")

        sum_loss = 0
        sum_acc = 0
        
        with tqdm(total=num_batches) as pbar:
            for X, y in dataloader:
                X, y = X.to(device), y.to(device)
                pred = model(X)

                loss = loss_fn(pred+1e-8,y.long())
                pbar.update(1)

                #train needs backward calculation
                if(mode=="train" or mode=="continue_train"):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                sum_loss += loss.item()*y.shape[0]
                num_correct = (pred.argmax(1) == y).type(torch.float).sum().item()
                acc = num_correct/y.shape[0]
                sum_acc += num_correct
                
                #show in process bar
                pbar.set_description(f"{mode:^14}")
                pbar.set_postfix(loss=f"{loss.item():8.4f}",acc=f"{acc:8.4f}")
                       
            losses.append(sum_loss/data_size)
            acces.append(sum_acc/data_size)
            
        print(f"{mode:^14}: loss={losses[-1]:8.4f}, acc={acces[-1]:8.4f}")
        
        #train needs save models
        if(mode=="train" or mode=="continue_train"):
            if(not os.path.exists(model_path)):
                os.makedirs(model_path)
            saved_path = f"{model_path}/mnist_net{e+1}.pth"
            if(mode=="continue_train"):
                saved_path = f"{modelfile[:-4]}+{e+1}.pth"
            torch.save(checkpoint, saved_path)

        #test only needs one epoch
        if(mode=="test" or mode=="pred"):
            break

        run(None,test_dataloader,saved_path,mode="test")

def predict(image_path):
    import cv2
    image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    image = torch.tensor(np.float32(image)).unsqueeze(0).unsqueeze(0)
    
    label = torch.tensor(np.float32([-1]))
    dataset = MyDataset(image,label)
    dataloader = DataLoader(dataset=dataset,batch_size=1)
    run(None,dataloader,modelfile,mode="pred")

if __name__ =="__main__":
    inputdir = "D:/vscode/workspace/python/PPML/dataset/mnist_dataset"
    modeldir = "D:/vscode/workspace/python/PPML/models/cnn_mnist"
    modelfile = "D:/vscode/workspace/python/PPML/models/cnn_mnist/mnist_net5.pth"
    
    train_dataloader = get_dataset(inputdir,mode="train",batch_size=batch_size_train)
    test_dataloader = get_dataset(inputdir,mode="test",batch_size=batch_size_test)
    #"train", "continue_train", "test"
    # run(train_dataloader,test_dataloader,modeldir,mode="train")
    # print("done")
    # run(train_dataloader,test_dataloader,modelfile,mode="continue_train",run_epochs=2)
    # print("done")
    # run(None,test_dataloader,modelfile,mode="test")

    predict("6.png")

    print("Done!")
