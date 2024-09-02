import os
import numpy as np

class Config_Generative_Model:
    def __init__(self):
        # project parameters
        self.seed = 2023
        self.root_path = '.'
        self.nsd_path = '/media/test/Cui/NSD/nsddata_stimuli/stimuli/nsd'
        self.patch_size = 16

        # self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains/ldm/semantic')
        self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains/ldm/label2img')
        # self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains/ldm/text2img-large')
        # self.pretrain_gm_path = os.path.join(self.root_path, 'pretrains/ldm/layout2img')
        
        self.dataset = 'NSD' # GOD or BOLD5000

        self.pretrain_sphericalmae_path = os.path.join(self.root_path, f'pretrains/sphericalmae/1024_642x1024.pth')

        self.img_size = 256

        np.random.seed(self.seed)
        # finetune parameters
        self.batch_size = 25
        self.lr = 5.3e-5
        self.num_epoch = 500
        
        self.precision = 32
        self.accumulate_grad = 1
        self.crop_ratio = 0.2
        self.global_pool = False
        self.use_time_cond = True
        self.eval_avg = True

        # diffusion sampling parameters
        self.num_samples = 5
        self.ddim_steps = 250
        self.HW = None
        # resume check util
        self.model_meta = None
        self.checkpoint_path = None#os.path.join(self.root_path, 'results/generation/642,1024_1/checkpoint.pth')
