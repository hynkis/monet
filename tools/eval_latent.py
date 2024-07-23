import tkinter as tk # Tkinter
from PIL import ImageTk, Image # Pillow
import os
import sys
import time
import argparse
import numpy as np
import cv2
import torch

sys.path.append("../") # for import vae
from torch.utils.data import Dataset, DataLoader
from custom_dataset import customDataset
from vae import VAE_SRL

"""
TODO
    - (V) change brightness of an image by slider
    - (V) load image and encode & decode image
    - check range of latent feature and slide it
"""

parser = argparse.ArgumentParser(description="Pytorch VAE")

parser.add_argument('--cpu',           action="store_true", help='run on CPU (default: False)')
parser.add_argument('--gpu_num',          type=int, default=0,     metavar='N', help='GPU number for training')
parser.add_argument('--seed',           type=int, default=123456, metavar='N',  help='random seed (default: 123456)')

parser.add_argument('--batch_size',    type=int, default=1,     metavar='N', help='batch size (default: 128)')
parser.add_argument('--lr',             type=float, default=1e-4, metavar='G', help='learning rate. (default: 0.0003)')

parser.add_argument('--fc_hidden1',    type=int, default=1024, metavar='G', help='fc_hidden1 in ResnetVAE (default: 1024)')
parser.add_argument('--fc_hidden2',    type=int,  default=768, metavar='G', help='fc_hidden2 in ResnetVAE (default: 768)')
parser.add_argument('--drop_p',        type=float, default=0.3, metavar='G', help='drop_p in ResnetVAE (default: 0.3)')
parser.add_argument('--CNN_embed_dim', type=int, default=256, metavar='G', help='CNN_embed_dim in ResnetVAE / state dim in VAE_SRL (default: 256)')

parser.add_argument('--vae_model',     default="VAE_SRL", help='Model type of VAE (VAE, VAE_SRL, ...) (default: VAE)')
parser.add_argument('--loss_function',     default="MSE", help='loss_function type of VAE (ELBO, MMD, ...) (default: ELBO)')


args = parser.parse_args()

IMAGE_SHAPE = (224,224)
device = torch.device("cuda:" + str(args.gpu_num))
if args.cpu:
    device = torch.device("cpu")

class GUI(object):
    def __init__(self,
                 window_title="Evaluate Latent Space",
                 geometry="920x640+50+50",
                 resizable=False,
                 dataset_dir = "",
                 img_name_txt = "",
                 trained_model_path = "",
                ):
        # GUI variables
        self.slider1_value = 0
        self.latent_state_ratio = 0
        self.torch_img_input = None
        self.torch_img_output = None
        self.pil_img_input = None
        self.pil_img_output = None

        # Root GUI window
        self.window = tk.Tk()
        self.window.title(window_title)
        self.window.geometry(geometry)
        self.window.resizable(resizable, resizable)

        # Slidebar
        self.slider1_vertical = tk.Scale(self.window, from_=0, to=1000,
                                        command=self.update_slider1,
                                        orient='horizontal')
        self.slider1_vertical.place(x=(50+224+6), y=50)
        
        # Button
        self.button_sample = tk.Button(self.window, text="Sample", command=self.button_task_sample_image)
        self.button_sample.place(x=50, y=(50+224+20))
        self.button_inference = tk.Button(self.window, text="Inference", command=self.button_task_inference)
        self.button_inference.place(x=50+100, y=(50+224+20))


        print("GUI setting is defined.")

        # Dataset & Dataloader
        self.dataset = customDataset(dataset_dir=dataset_dir,
                                     img_name_txt=img_name_txt,
                                     label_txt=None,
                                     loss_function= args.loss_function,
                                     transform=None,
                                     )
        self.dataloader = DataLoader(dataset=self.dataset,
                                     batch_size=1,
                                     shuffle=True,
                                     num_workers=1,
                                     drop_last=True
                                     )
        print("Dataset & Dataloader are loaded.")

        # Trained VAE Model
        self.model = VAE_SRL(img_shape=IMAGE_SHAPE, args=args)
        self.model.load_model(trained_model_path)
        self.model.vae.eval()

        print("VAE model is loaded.")

    def update_gui(self):
        self.window.update()
        # Update variable with slider1
        self.latent_state_ratio = 10 * (2.0*(float(self.slider1_value) * 0.001)-1) # from [0,1000] to [-0.1,+0.1]
    
    def update_slider1(self, value):
        self.slider1_value = value
        print("slider1_value, latent_state_ratio :", self.slider1_value, self.latent_state_ratio)

    def button_task_sample_image(self):
        # Sample image
        data = next(iter(self.dataloader)) # torch_img_input
        self.torch_img_input = data["image"] # (batch,3,224,224) ELBO:[0,1], MSE:[-1,+1]

        if args.loss_function == "ELBO":
            cv_img = self.dataset.torch_to_cv_img(self.torch_img_input[0], denormalize=False) # (1,3,224,224) to (224,224,3), [0,1]
        elif args.loss_function == "MSE":
            cv_img = self.dataset.torch_to_cv_img(self.torch_img_input[0], denormalize=True) # (1,3,224,224) to (224,224,3), [0,1]
        else:
            raise NotImplementedError

        cv_img = np.array(cv_img*255., dtype=np.uint8) # from [0,1] to [0,255]
        self.pil_img_input = self.convert_cv_to_pil(cv_img)
        self.update_window_image_input(self.pil_img_input)

    def button_task_inference(self):
        # Feedforward
        X_recon_new, z, mu, logvar = self.model.vae(self.torch_img_input.to(device))
        # Decoder with perturbed mu, logvar
        # mu[:, 100:150] *= self.latent_state_ratio

        z_new = self.model.vae.reparameterize(mu, logvar)
        X_recon_new = self.model.vae.decode(z_new)

        # Visualize reconstructed image
        if args.loss_function == "ELBO":
            cv_img = self.dataset.torch_to_cv_img(X_recon_new.detach().cpu().squeeze(), denormalize=False)
        elif args.loss_function == "MSE":
            cv_img = self.dataset.torch_to_cv_img(X_recon_new.detach().cpu().squeeze(), denormalize=True)
        else:
            raise NotImplementedError

        cv_img = np.array(cv_img*255., dtype=np.uint8) # from [0,1] to [0,255]
        self.pil_img_output = self.convert_cv_to_pil(cv_img)
        self.update_window_image_output(self.pil_img_output)
        print("Update recon image")

    def update_window_image_input(self, pil_img):
        window_img = ImageTk.PhotoImage(pil_img)
        label = tk.Label(self.window, image=window_img)
        label.image = window_img
        label.place(x=50, y=50)

    def update_window_image_output(self, pil_img):
        window_img = ImageTk.PhotoImage(pil_img)
        label = tk.Label(self.window, image=window_img)
        label.image = window_img
        label.place(x=920-(50+224), y=50)

    def load_image(self, path, normalize=False):
        # can use any other library too like OpenCV as long as you are consistent with it
        raw_img = Image.open(path) # RGB image
        resized_img = raw_img.resize(IMAGE_SHAPE) # resizes to (224,224)
        if normalize:
            norm_img = np.transpose(resized_img, (2,1,0)) # (H,W,C) to (C,W,H)
            imx_t = np.array(norm_img, dtype=np.float32) / 255.
            imx_t -= 0.5
            imx_t *= 2.
            return imx_t
        else:
            return resized_img

    def convert_pil_to_cv(self, pil_img):
        cv_img = np.array(pil_img)
        cv_img = cv_img[:, :, ::-1].copy()
        return cv_img

    def convert_cv_to_pil(self, cv_img):
        pil_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(pil_img)
        return pil_img


def main():
    dataset_dir  = "./test_image/img_data"
    img_name_txt = "./test_image/img_data.txt"
    label_txt    = "./test_image/img_data.txt"
    LOAD_MODEL_PATH = "./test_model/vae_model_VAE_SRL_MSE__combined_addTanh_batch256_2021-01-26_15-25-18_epoch_4232_iteration_304000.pth"

    gui = GUI(window_title="Evaluate Latent State",
              geometry="920x640+50+50",
              resizable=False,
              dataset_dir=dataset_dir,
              img_name_txt=img_name_txt,
              trained_model_path=LOAD_MODEL_PATH,
              )

    while True:
        gui.update_gui()

if __name__ == "__main__":
    main()
