import torch
from torchvision import datasets, transforms
import torch.utils.data as data
import os
from PIL import Image
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

base_transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize([0.5] * 3, [0.5] * 3)
])

norm_transform = transforms.Compose([
    transforms.Normalize([0.5] * 3, [0.5] * 3)
])

inverse_transform = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                             std=[1 / 0.5] * 3),
                                        transforms.Normalize(mean=[-0.5] * 3,
                                                             std=[1., 1., 1.]),
                                        ])

upscale_factor = 4

def image_loader(path):
    img = Image.open(path)
    img_tensor = base_transform(img)
    return img_tensor

def lr_image_loader(path, upscale_factor):
    img = Image.open(path)
    width, height = img.size
    tw = width//upscale_factor
    th = height//upscale_factor
    #print((tw,th))
    img = img.resize(size = (tw,th), resample = Image.BILINEAR)
    #print("lr:",img.size)
    img_tensor = base_transform(img)
    #print(img_tensor.shape)
    return img_tensor


class FRDataset(data.Dataset):

    def __init__(self, hr_dir, upscale_factor):
        self.file_hr_dir = hr_dir
        self.transform = base_transform
        self.upscale_factor = upscale_factor
        # self.image_loader = image_loader()
        self.hr_frames_set = os.listdir(self.file_hr_dir)

    def __getitem__(self, index):
        def get_from_set(dir, frame_set):
            frames = frame_set[index]  # 0266
            # print(f'frame is {frames}, typ is {type(frames)}')
            # frame_tensor = torch.Tensor(size=(frame_counter, 3, self.height, self.weight))
            frame_tensor = []

            # file_dir_frames = self.file_dir + frames
            file_dir_frames = os.path.join(dir, frames)
            imgs_path = os.listdir(file_dir_frames)
            imgs_path.sort()
            #import pdb; pdb.set_trace()
            i = 0
            for img in imgs_path:
                final_path = file_dir_frames + "/" + img
                # final_path = '/'.os.listdir(file_dir_frames,img)
                img_tensor = image_loader(final_path)
                # print(img_tensor.size())
                frame_tensor.append(img_tensor)
                i = i + 1
            res = torch.stack(frame_tensor, dim=0)
            # print(f'res has shape {res.shape}')
            return res

        def get_lr_from_set(dir, frame_set, upscale_factor):
            frames = frame_set[index]  # 0266
            # print(f'frame is {frames}, typ is {type(frames)}')
            # frame_tensor = torch.Tensor(size=(frame_counter, 3, self.height, self.weight))
            frame_tensor = []

            # file_dir_frames = self.file_dir + frames
            file_dir_frames = os.path.join(dir, frames)
            imgs_path = os.listdir(file_dir_frames)
            imgs_path.sort()
            i = 0
            for img in imgs_path:
                final_path = file_dir_frames + "/" + img
                # final_path = '/'.os.listdir(file_dir_frames,img)
                img_tensor = lr_image_loader(final_path,upscale_factor)
                # print(img_tensor.size())
                frame_tensor.append(img_tensor)
                i = i + 1
            res = torch.stack(frame_tensor, dim=0)
            # print(f'res has shape {res.shape}')
            return res

        return get_lr_from_set(self.file_hr_dir, self.hr_frames_set,self.upscale_factor), \
               get_from_set(self.file_hr_dir, self.hr_frames_set)

    def __len__(self):
        return len(self.hr_frames_set)

    # # this returns the basic infomation of the dataset.
    # def touch(self):

class loader_wrapper(object):
    def __init__(self, loader):
        self.loader = loader

    def __iter__(self):
        for lr_img, hr_img in self.loader:
            yield lr_img.permute(1, 0, 2, 3, 4), hr_img.permute(1, 0, 2, 3, 4)

    def __len__(self):
        return len(self.loader)


def get_data_loaders(batch, shuffle_dataset=True, dataset_size=0, validation_split=0.2):
    # batch = 4 # batch size of the data every time for training
    # batch_number = 100000  # number of batches, so we totally have batch_number * batch images
    # HR_height = height
    # HR_width = width
    #
    # LR_height = HR_height // SRFactor
    # LR_width = HR_width // SRFactor

    train_dir_HR = 'Data/HR'

    FRData = FRDataset(hr_dir=train_dir_HR,upscale_factor = upscale_factor)
    #import pdb; pdb.set_trace()

    # data_loader_LR = data.DataLoader(FRData_LR, batch_size = batch, shuffle = True)
    # data_loader_HR = data.DataLoader(FRData_HR, batch_size = batch, shuffle = True)

    # print(data_loader[0].size())
    random_seed = 42
    if dataset_size == 0:
        dataset_size = len(FRData)
    #validation_split = 0.2  # Train-Val : 8-2
    print("Dataset size:", len(FRData))
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    #indices = [1]
    train_indices, val_indices = indices[split:], indices[:split]
    print("Training samples chosen:", train_indices)
    train_sampler = SubsetRandomSampler(train_indices)
    print("Training set size:", len(train_sampler))
    valid_sampler = SubsetRandomSampler(val_indices)
    print("Validation set size:", len(valid_sampler))

    train_loader = torch.utils.data.DataLoader(FRData, batch_size=batch, sampler=train_sampler, drop_last=True)
    validation_loader = torch.utils.data.DataLoader(FRData, batch_size=batch, sampler=valid_sampler, drop_last=True)
    train_loader = loader_wrapper(train_loader)
    validation_loader = loader_wrapper(validation_loader)

    return train_loader, validation_loader


if __name__ == "__main__":
    train, val = get_data_loaders(4)
    for lr_img, hr_img in train:
        print(f'lr_img shape is {lr_img.shape}, hr_img shape is {hr_img.shape}')
        break

# class TestFRVSR(unittest.TestCase):
#     def TestGetDataLoader(self):
#


# for i_batch, sample_batched in enumerate(zip(train_loader_LR, train_loader_HR)):
#        #print(sample_batched)
#        #print(data_loader_HR[i_batch].size())
#        permuted_LR_data = sample_batched[0].permute(1, 0, 2, 3, 4)
#        permuted_HR_data = sample_batched[1].permute(1, 0, 2, 3, 4) #labels
#        #print(permuted_data.contiguous())
#        print("LR:",permuted_LR_data.size())
#        print("HR:",permuted_HR_data.size())
#
# for j_batch, sample_batched in enumerate(zip(validation_loader_LR, validation_loader_HR)):
#        #print(sample_batched)
#        #print(data_loader_HR[i_batch].size())
#        permuted_LR_data = sample_batched[0].permute(1, 0, 2, 3, 4)
#        permuted_HR_data = sample_batched[1].permute(1, 0, 2, 3, 4) #labels
#        #print(permuted_data.contiguous())
#        print("LR:",permuted_LR_data.size())
#        print("HR:",permuted_HR_data.size())
