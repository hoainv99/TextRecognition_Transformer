import torch
import torch.nn as nn
from torch.autograd import Variable
import time
from dataset import *
from model import *
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        


class LabelSmoothing(nn.Module):
    "Implement label smoothing."
    def __init__(self, size, padding_idx=0, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(size_average=False)
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None
        
    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)

        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
      
        self.true_dist = true_dist
 
        return self.criterion(x, Variable(true_dist, requires_grad=False))

class SimpleLossCompute:
    "A simple loss compute and train function."
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt
        
    def __call__(self, x, y, norm):

        loss = self.criterion(x.contiguous().view(-1, x.size(-1)).log(), 
                              y.contiguous().view(-1)) / norm
        if self.opt is not None:
            loss.backward()
            self.opt.step()
            self.opt.optimizer.zero_grad()

        return loss.data * norm
def calc_accuracy(X,Y,norm):

    _, predicted = torch.max(X,-1)

    mask = (Y!=0).type(torch.FloatTensor).cuda()
 
    out = torch.mul(predicted, mask)
    
    vec_res = torch.mul((out == Y).type(torch.FloatTensor).cuda(),mask)
    train_acc = vec_res.sum()/norm
    return train_acc

def run_epoch(epoch,dataloader, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    total_acc = 0
    total_batch = 0

    for i, (imgs, labels_y, labels) in enumerate(dataloader):
        batch = Batch(imgs, labels_y, labels)
        out = model(batch.imgs, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        _,pred = torch.max(out,dim = -1)
      
        total_acc += calc_accuracy(out,batch.trg_y,batch.ntokens)
        total_batch += 1
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 20 == 1:
            elapsed = time.time() - start
            print("Epoch %d Step: %d Loss: %f Tokens per Sec: %f" %
                    (epoch,i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens, total_acc / total_batch
def train(pretrained = False):
    batch_size = 64
    best_loss=1e+9
    train_dataloader = torch.utils.data.DataLoader(MyDataset(mat_train['traindata'][0]), batch_size=batch_size, shuffle=True, num_workers=0)
    val_dataloader = torch.utils.data.DataLoader(MyDataset(mat_test['testdata'][0]), batch_size=batch_size, shuffle=False, num_workers=0)
    model = model._make_model(len(char2token))
    if pretrained :
      checkpoint = torch.load('checkpoint/best_loss.pth')
      model.load_state_dict(checkpoint['state_dict'])
      best_loss = checkpoint['loss']
      model_opt = checkpoint['optimizer']
    else :
      model_opt = NoamOpt(model.tgt_emb[0].d_model, 1, 2000,torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    model.cuda()
    criterion = LabelSmoothing(size=len(char2token), padding_idx=0, smoothing=0.1)
    criterion.cuda()
    for epoch in range(10000):
        model.train()
        test_train,acc_train = run_epoch(epoch,train_dataloader, model, 
              SimpleLossCompute(model.generator, criterion, model_opt))
        
        model.eval()
        test_loss,test_acc = run_epoch(epoch,val_dataloader, model, 
              SimpleLossCompute(model.generator, criterion, None))
        print(f"test_train: {test_train} acc_train: {acc_train} test_loss: {test_loss} test_acc: {test_acc}")
        if best_loss>test_loss:
          best_loss = test_loss
          print("save checkpoint ...")
          torch.save({                                                                                                                                                                                                 
            'epoch': epoch,
            'loss' : best_loss,                                                                                                                                                                                  
            'state_dict': model.state_dict(),                                                                                                                                                                                
            'optimizer': model_opt},                                                                                                                                                                                     
            'checkpoint/best_loss_1.pth')
            
if __name__=='__main__':
    train(pretrained=False)