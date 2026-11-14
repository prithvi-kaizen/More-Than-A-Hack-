"""Small symmetric, ball-free temporal ownership expert trained on external matches."""
import numpy as np
import torch
from torch import nn
from mtah.external import FROZEN_SPLIT


class ResidualBlock(nn.Module):
    def __init__(self,channels,dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels,channels,5,padding=2*dilation,dilation=dilation)
        self.norm = nn.LayerNorm(channels)

    def forward(self,x):
        y = self.conv(x).transpose(1,2)
        y = torch.nn.functional.gelu(self.norm(y)).transpose(1,2)
        return x+y


class MotionTCN(nn.Module):
    context = 14
    def __init__(self,input_dim=108,channels=32):
        super().__init__()
        if input_dim%2: raise ValueError('INVALID_MOTION_DIMENSION')
        self.input_dim,self.channels = input_dim,channels
        self.network = nn.Sequential(nn.Conv1d(input_dim,channels,1),nn.GELU(),
            *(ResidualBlock(channels,d) for d in (1,2,4)),nn.Conv1d(channels,1,1))

    def forward(self,x):
        # x is batch/time/features; first half is antisymmetric, second invariant.
        flip = torch.cat((-x[:,:,:self.input_dim//2],x[:,:,self.input_dim//2:]),dim=-1)
        scores = self.network(torch.cat((x,flip),dim=0).transpose(1,2)).squeeze(1)
        first,second = scores.chunk(2,dim=0)
        return (first-second)/2


class Windows(torch.utils.data.Dataset):
    def __init__(self,sequences,window=64,stride=32,context=14):
        self.sequences,self.window,self.context = sequences,window,context
        self.index=[]
        for i,s in enumerate(sequences):
            n=len(s['features'])
            starts=sorted(set(list(range(0,max(1,n-window+1),stride))+[max(0,n-window)]))
            self.index.extend((i,start) for start in starts)

    def __len__(self): return len(self.index)

    def __getitem__(self,index):
        i,start=self.index[index];s=self.sequences[i];stop=min(len(s['features']),start+self.window)
        n=stop-start
        x=np.zeros((self.window,s['features'].shape[1]),dtype=np.float32)
        y=np.zeros(self.window,dtype=np.float32);w=np.zeros(self.window,dtype=np.float32)
        x[:n]=s['features'][start:stop]
        labels=s['labels'][start:stop];valid=np.isin(labels,['H','A'])
        y[:n]=labels=='H';w[:n]=s['weights'][start:stop]*valid
        if start>0:w[:self.context]=0
        if stop<len(s['features']):w[n-self.context:n]=0
        return torch.from_numpy(x),torch.from_numpy(y),torch.from_numpy(w)


def train_motion(sequences,match_ids,settings,seed):
    if not match_ids or len(set(match_ids))!=len(match_ids) or not set(match_ids).issubset(FROZEN_SPLIT['train']):
        raise ValueError('UNAUTHORIZED_TEMPORAL_TRAINING_MATCHES')
    if any(s['match_id'] not in match_ids for s in sequences):
        raise ValueError('TEMPORAL_SEQUENCE_SPLIT_LEAKAGE')
    if not sequences or set(np.concatenate([s['labels'] for s in sequences])) & {'H','A'} != {'H','A'}:
        raise ValueError('TEMPORAL_TRAINING_CLASSES_MISSING')
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model=MotionTCN(sequences[0]['features'].shape[1],settings['channels'])
    data=Windows(sequences,settings['window'],settings['window_stride'])
    generator=torch.Generator().manual_seed(seed)
    loader=torch.utils.data.DataLoader(data,batch_size=settings['batch_size'],shuffle=True,
        num_workers=0,generator=generator)
    optimizer=torch.optim.AdamW(model.parameters(),lr=settings['learning_rate'],weight_decay=settings['weight_decay'])
    losses=[]
    for epoch in range(settings['epochs']):
        model.train();total,count=0.,0.
        for x,y,w in loader:
            if not w.sum():continue
            optimizer.zero_grad()
            logits=model(x)
            loss=(torch.nn.functional.binary_cross_entropy_with_logits(logits,y,reduction='none')*w).sum()/w.sum()
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1)
            optimizer.step();total+=float(loss.detach())*float(w.sum());count+=float(w.sum())
        losses.append(total/max(count,1))
        if not np.isfinite(losses[-1]): raise ValueError('NONFINITE_TEMPORAL_LOSS')
        print(f'Temporal epoch {epoch+1}: complete',flush=True)
    model.eval()
    return model,losses


def predict_motion(model,features,n_frames,sample_stride=5,chunk=512):
    model.eval();values=[];n=len(features);halo=model.context
    with torch.inference_mode():
        for start in range(0,n,chunk):
            stop=min(n,start+chunk);a=max(0,start-halo);b=min(n,stop+halo)
            x=torch.from_numpy(np.asarray(features[a:b],dtype=np.float32))[None]
            logits=model(x)[0,start-a:stop-a].numpy()
            values.append(logits)
    logits=np.concatenate(values) if values else np.zeros(0,dtype=np.float32)
    if not len(logits):raise ValueError('EMPTY_TEMPORAL_SEQUENCE')
    full=np.interp(np.arange(n_frames),np.arange(n)*sample_stride,logits)
    return 1/(1+np.exp(-np.clip(full,-30,30)))


def save_motion(model,path,match_ids):
    torch.save(dict(state_dict=model.state_dict(),input_dim=model.input_dim,channels=model.channels,
                    training_matches=list(match_ids)),path)


def load_motion(path):
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=MotionTCN(state['input_dim'],state['channels'])
    model.load_state_dict(state['state_dict']);model.eval()
    return model
