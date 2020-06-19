import scipy.io
import torch
import cv2
import numpy as np
from torch.autograd import Variable
import os

SIZE=96
label_len = 36
vocab =  "<,.+:-?$ 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ>"
# start symbol <
# end symbol >
char2token = {"PAD":0}
token2char = {0:"PAD"}
mat_train = scipy.io.loadmat('/content/drive/My Drive/ocr-transformer/IIIT5K-Word_V3.0/IIIT5K/traindata.mat')
mat_test = scipy.io.loadmat('/content/drive/My Drive/ocr-transformer/IIIT5K-Word_V3.0/IIIT5K/testdata.mat')
def resize(img):
    h, w, c = img.shape
    if w > h:
        nw, nh = SIZE, int(h * SIZE/w)
        if nh < 10 : nh = 10
        #print(h, w, nh, nw)
        img = cv2.resize(img, (nw, nh))
        a1 = int((SIZE-nh)/2)
        a2= SIZE-nh-a1
        pad1 = np.zeros((a1, SIZE, c), dtype=np.uint8)
        pad2 = np.zeros((a2, SIZE, c), dtype=np.uint8)
        img = np.concatenate((pad1, img, pad2), axis=0)
    else:
        nw, nh = int(w * SIZE/h), SIZE
        if nw < 10 : nw = 10
        #print(h, w, nh, nw)
        img = cv2.resize(img, (nw, nh))
        a1 = int((SIZE-nw)/2)
        a2= SIZE-nw-a1
        pad1 = np.zeros((SIZE, a1, c), dtype=np.uint8)
        pad2 = np.zeros((SIZE, a2, c), dtype=np.uint8)
        img = np.concatenate((pad1, img, pad2), axis=1)
    return img

label_len = 36
vocab =  "<,.+:-?$ 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ>"
# start symbol <
# end symbol >
char2token = {"PAD":0}
token2char = {0:"PAD"}
for i, c in enumerate(vocab):
    char2token[c] = i+1
    token2char[i+1] = c
def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0
class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, imgs, trg_y, trg, pad=0):
        self.imgs = Variable(imgs.cuda(), requires_grad=False)
        self.src_mask = Variable(torch.from_numpy(np.ones([imgs.size(0), 1, 36], dtype=np.bool)).cuda())
        if trg is not None:
            self.trg = Variable(trg.cuda(), requires_grad=False)
            self.trg_y = Variable(trg_y.cuda(), requires_grad=False)
            self.trg_mask = \
                self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return Variable(tgt_mask.cuda(), requires_grad=False)
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        root = "ocr-transformer/IIIT5K-Word_V3.0/IIIT5K/"
        list_name = self.data['ImgName'][idx]
        file_name = ''.join(list_name)

        img = cv2.imread(os.path.join(root,file_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = resize(img)/255.
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).float().cuda()
        label_y_str = self.data['GroundTruth'][idx][0]
        label = np.zeros(label_len, dtype=int)
        for i, c in enumerate('<'+label_y_str):
            label[i] = char2token[c]
        label = torch.from_numpy(label)
        label_y = np.zeros(label_len, dtype=int)
        for i, c in enumerate(label_y_str+'>'):
            label_y[i] = char2token[c]
      
        label_y = torch.from_numpy(label_y)
  
        return img, label_y, label

if __name__=='__main__':

    mydataset = MyDataset(mat_train["traindata"][0])
    dataloader = torch.utils.data.DataLoader(mydataset, batch_size=2, shuffle=False, num_workers=0)
