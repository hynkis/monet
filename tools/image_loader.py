import torchvision.datasets as dataset
import torchvision.transforms as transforms
import torch
import numpy as np

import matplotlib.pyplot as plt
import cv2

DATASET_DIR = "../dataset/train/" # end with '/'

# Reference: https://ndb796.tistory.com/372
def custom_imshow(img):
    img = img.numpy()
    plt.imshow(np.transpose(img, (1, 2, 0)))
    plt.show()


img_dataset = dataset.ImageFolder(root=DATASET_DIR,
                                  transform=transforms.Compose([
                                            # transforms.Scale(128),       # scale to 128
                                            # transforms.CenterCrop(128),  # center crop,
                                            transforms.ToTensor(),       # transform to tensor (normalize to 0~1)
                                            # transforms.Normalize((0.5, 0.5, 0.5),  # normalize to -1 ~ 1
                                            #                      (0.5, 0.5, 0.5)), # (c - m)/s
                                            ])
                                 )


dataloader = torch.utils.data.DataLoader(img_dataset,
                                         batch_size=2,
                                         shuffle=True,
                                         num_workers=8)
                                    

for i, data in enumerate(dataloader):
    print(data[0].size())  # input image
    print(data[1])         # class label
    custom_imshow(data[0][0])