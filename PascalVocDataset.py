import pandas as pd
import os
#from bs4 import BeautifulSoup
#from more_itertools import unique_everseen
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
#import skimage
#from skimage import io

class PascalVOC(Dataset):
    """
    Handle Pascal VOC dataset
    """
    def __init__(self, root_dir,dataset='train',transform=None):
        """
        Summary: 
            Init the class with root dir
        Args:
            root_dir (string): path to your voc dataset
            dataset (string): "train", "val", "train_val", or "test" (if available)
        """
        self.root_dir = root_dir
        self.img_dir =  os.path.join(root_dir, 'JPEGImages/')
        self.ann_dir = os.path.join(root_dir, 'Annotations')
        self.set_dir = os.path.join(root_dir, 'ImageSets', 'Main')
        self.cache_dir = os.path.join(root_dir, 'csvs')
        self.transform = transform
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        img_set = self.get_set(dataset) # set of all image names
        one_hot_label = np.zeros((len(img_set),len(self.list_image_sets())))
        
        #fill in one hot label in fast way
        for i,cat in enumerate(self.list_image_sets()):
            arr = self._imgs_from_category(cat,dataset).values[:,1]
            arr[arr==-1] = 0
            one_hot_label[:,i] = arr
        
        self.img_set = img_set
        self.one_hot_label = one_hot_label
    
    def __getitem__(self,index):
        """
        returns image, one hot label
        """
        
        img_name = os.path.join(self.img_dir,self.img_set[index]+".jpg")
        img = Image.open(img_name).convert('RGB')
        
        if self.transform != None:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)#use default transform
            
        label = self.one_hot_label[index]
        
        return img,label
        
    def __len__(self):
        return len(self.img_set)
    def get_set(self,dataset):
        """
        dataset(string): "train", "val", "train_val", or "test" (if available)
        returns set of train/val/test
        """
        filename = os.path.join(self.set_dir, dataset + ".txt")
        ls = []
        with open(filename,"r") as f:
            for line in f:
                ls.append(line.strip())
        return ls
    
    def list_image_sets(self):
        """
        Summary: 
            List all the image sets from Pascal VOC. Don't bother computing
            this on the fly, just remember it. It's faster.
        """
        return [
            'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair',
            'cow', 'diningtable', 'dog', 'horse',
            'motorbike', 'person', 'pottedplant',
            'sheep', 'sofa', 'train',
            'tvmonitor']

    def _imgs_from_category(self, cat_name, dataset):
        """
        Summary: 
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            pandas dataframe: pandas DataFrame of all filenames from that category
        """
        filename = os.path.join(self.set_dir, cat_name + "_" + dataset + ".txt")
        df = pd.read_csv(
            filename,
            delim_whitespace=True,
            header=None,
            names=['filename', 'true'])
        return df

    def imgs_from_category_as_list(self, cat_name, dataset):
        """
        Summary: 
            Get a list of filenames for images in a particular category
            as a list rather than a pandas dataframe.
        Args:
            cat_name (string): Category name as a string (from list_image_sets())
            dataset (string): "train", "val", "train_val", or "test" (if available)
        Returns:
            list of srings: all filenames from that category
        """
        df = self._imgs_from_category(cat_name, dataset)
        df = df[df['true'] == 1]
        return df['filename'].values

def imshow(tensor):
    plt.imshow(tensor.numpy().transpose(1,2,0))

def show_image_label(dataset,out):
    """
    gets output from dataset and shows image + labels
    """
    ls = dataset.list_image_sets()
    image,labels = out[0],out[1]
    imshow(image)
    for i,label in enumerate(labels):
        if label ==1:
            print(ls[i],end=" ")