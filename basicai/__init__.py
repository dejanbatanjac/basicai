import torch

def test():
    print('test')

# made for 
def CudaNorm3(t,m=(0.485, 0.456, 0.406),s=(0.229, 0.224, 0.225)):
    'normalization of the batch on cuda'
    mean = torch.Tensor(m).float().reshape(1,3,1,1).cuda()
    std = torch.Tensor(s).float().reshape(1,3,1,1).cuda()
    t = (t-mean)/(std+1e-6)
    return t