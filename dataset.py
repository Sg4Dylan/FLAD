from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from utils import get_file_list


class FLADS(Dataset):

    def __init__(self, ds_root):
        self.ds_root = ds_root
        self.ds_list = []
        self.ds_label = []
        self.img_size = 400
        self.load()


    def load(self):
        self.ds_list = get_file_list(self.ds_root)
        for i in self.ds_list:
            if 'LL' in i:
                self.ds_label.append(0)
            if 'AAC' in i:
                self.ds_label.append(1)
            if 'MP3' in i:
                self.ds_label.append(2)
            if 'OPUS' in i:
                self.ds_label.append(3)


    def __len__(self):
        return len(self.ds_list)
    
    
    def __getitem__(self, index):
        image_path = self.ds_list[index]
        img = Image.open(image_path).resize((self.img_size, self.img_size),Image.BICUBIC)
        img = img.convert('RGBA').convert('RGB')
        ds_fmt = np.array(img).transpose(2, 0, 1)
        img_data = ds_fmt.astype('float32')
        mean_vec = np.array([0.485, 0.456, 0.406])
        stddev_vec = np.array([0.229, 0.224, 0.225])
        norm_img_data = np.zeros(img_data.shape).astype('float32')
        for i in range(img_data.shape[0]):
            norm_img_data[i,:,:] = (img_data[i,:,:]/255 - mean_vec[i]) / stddev_vec[i]
        return norm_img_data, self.ds_label[index]
