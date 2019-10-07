from .Dataset import Dataset
import PIL
import os
import math
import numpy as np
import pathlib

from torchvision import transforms

#################################################
# Directory.py can read items from a directory
################################################# 
 
class ImageDirectory(Dataset):
    'Reads all images from a directory'    
    
    def __init__(self, dir1):        
        self.dir1 = dir1        
        self.imgs = self.read(dir1)

    def __getitem__(self, index):
        path = self.imgs[index]
        img = PIL.Image.open(path).convert('RGB')         
        img = transforms.ToTensor()(img)        
        return img

    def __len__(self):
        return len(self.imgs)
    
    
    def is_image_file(self, filename):
        EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
        return any(filename.endswith(extension) for extension in EXTENSIONS)
    

    def read(self, dir):
        images = []
        assert os.path.isdir(dir), r'{dir} is not a valid directory' 

        for root, _, fnames in sorted(os.walk(dir)):        
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        images.sort()
        return images
    
    def tv(self, percent_t=0.8, percent_v=0.2):
        'just returns indices'
        l  = len(self.imgs)
        cut1 = math.floor(l*percent_t)
        cut2 = math.floor(l*percent_v)
        per = np.random.permutation(l)
        return per[:cut1], per[-cut2:]
    
ID = ImageDirectory    


class ImageLabelDirectory(Dataset):
    'Reads all images with labels from a directory'    
    
    def __init__(self, dir1):        
        self.dir1 = dir1        
        self.imgs = self.read(dir1)
        self.classes=[]

    def __getitem__(self, index):
        path = pathlib.Path(self.imgs[index])
        img = PIL.Image.open(path).convert('RGB')         
        img = transforms.ToTensor()(img)  
        label = path.parts[-2]
        if (label not in self.classes):
            self.classes.append(label)            
        label = self.classes.index(label)    
        return img,label

    def __len__(self):
        return len(self.imgs)
    
    
    def is_image_file(self, filename):
        EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
        return any(filename.endswith(extension) for extension in EXTENSIONS)
    

    def read(self, dir):
        images = []
        assert os.path.isdir(dir), r'{dir} is not a valid directory' 

        for root, _, fnames in sorted(os.walk(dir)):        
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        images.sort()
        return images
    
    def tv(self, percent_t=0.8, percent_v=0.2):
        'just returns indices'
        l  = len(self.imgs)
        cut1 = math.floor(l*percent_t)
        cut2 = math.floor(l*percent_v)
        per = np.random.permutation(l)
        return per[:cut1], per[-cut2:]
    
ILD = ImageLabelDirectory

class ImageImageDirectory(Dataset):
    'Reads pair of images from a directory dir1, and dir2'    
    
    def __init__(self, dir1, dir2):
        self.dir1 = dir1
        self.dir2 = dir2
        self.imgs1 = self.read(dir1)
        self.imgs2 = self.read(dir2)        
        
    def __getitem__(self, index):
        path1 = self.imgs1[index]
        path2 = self.imgs2[index]
        img1 = PIL.Image.open(path1).convert('RGB')         
        img2 = PIL.Image.open(path2).convert('RGB')         
        img1 = transforms.ToTensor()(img1)        
        img2 = transforms.ToTensor()(img2)        
        return img1,img2        

    def __len__(self):
        return len(self.imgs1)    
    
    def is_image_file(self, filename):
        EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP']
        return any(filename.endswith(extension) for extension in EXTENSIONS)    

    def read(self, dir):
        images = []
        assert os.path.isdir(dir), r'{dir} is not a valid directory' 

        for root, _, fnames in sorted(os.walk(dir)):        
            for fname in fnames:
                if self.is_image_file(fname):
                    path = os.path.join(root, fname)
                    images.append(path)
        images.sort()
        return images
    
    def tv(self, percent_t=0.8, percent_v=0.2):
        'just return indices'
        l  = len(self.imgs1)
        cut1 = math.floor(l*percent_t)
        cut2 = math.floor(l*percent_v)
        per = np.random.permutation(l)
        return per[:cut1], per[-cut2:]
    
IID = ImageImageDirectory
