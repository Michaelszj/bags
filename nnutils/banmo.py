# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
import os
import os.path as osp
import sys
sys.path.insert(0, 'third_party')
import cv2, numpy as np, time, torch, torchvision
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random
# import trimesh, pytorch3d, pytorch3d.loss, pdb
import pytorch3d, pytorch3d.loss
from pytorch3d import transforms
import configparser
import argparse
from arguments import ModelParams, PipelineParams, OptimizationParams
from nnutils.nerf import Embedding, NeRF, RTHead, SE3head, RTExplicit, Encoder,\
                    ScoreHead, Transhead, NeRFUnc, MLP,\
                    grab_xyz_weights, FrameCode, RTExpMLP
from nnutils.geom_utils import K2mat, mat2K, Kmatinv, K2inv, raycast, sample_xy,\
                                chunk_rays, generate_bones,\
                                canonical2ndc, obj_to_cam, vec_to_sim3, \
                                near_far_to_bound, compute_flow_geodist, \
                                compute_flow_cse, fb_flow_check, pinhole_cam, \
                                render_color, mask_aug, bbox_dp2rnd, resample_dp, \
                                vrender_flo, get_near_far, array2tensor, rot_angle, \
                                rtk_invert, rtk_compose, bone_transform, correct_bones,\
                                correct_rest_pose, fid_reindex, skinning, lbs,get_T,orbit_camera
from nnutils.rendering import render_rays
from nnutils.loss_utils import *
from tqdm import tqdm
from utils.io import draw_pts,save_vid,draw_cams
from utils.loss_utils import l1_loss, ssim, sdf_loss
from scene.gaussian_model import GaussianModel
from scene.cameras import Camera, rtk_from_angles, angles_from_rtk
from gaussian_renderer import render_auto,render,render_points
import torch.nn.functional as F
# from diffusers import DDIMScheduler
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
# distributed data parallel
flags.DEFINE_integer('local_rank', 0, 'for distributed training')
flags.DEFINE_integer('ngpu', 1, 'number of gpus to use')

# data io
flags.DEFINE_integer('accu_steps', 1, 'how many steps to do gradient accumulation')
flags.DEFINE_string('seqname', 'syn-spot-40', 'name of the sequence')
flags.DEFINE_string('logname', 'exp_name', 'Experiment Name')
flags.DEFINE_string('checkpoint_dir', 'logdir/', 'Root directory for output files')
flags.DEFINE_string('model_path', '', 'load model path')
flags.DEFINE_string('pose_cnn_path', '', 'path to pre-trained pose cnn')
flags.DEFINE_string('rtk_path', '', 'path to rtk files')
flags.DEFINE_string('config', '', 'path to config files')
flags.DEFINE_bool('lineload',False,'whether to use pre-computed data per line')
flags.DEFINE_integer('n_data_workers', 1, 'Number of data loading workers')
flags.DEFINE_boolean('use_rtk_file', False, 'whether to use input rtk files')
flags.DEFINE_boolean('preload_pose', False, 'whether to use pre_estimated rtk files')
flags.DEFINE_boolean('debug', False, 'deubg')

# model: shape, appearance, and feature
flags.DEFINE_bool('use_human', False, 'whether to use human cse model')
flags.DEFINE_bool('symm_shape', False, 'whether to set geometry to x-symmetry')
flags.DEFINE_bool('env_code', True, 'whether to use environment code for each video')
flags.DEFINE_bool('env_fourier', True, 'whether to use fourier basis for env')
flags.DEFINE_bool('use_unc',False, 'whether to use uncertainty sampling')
flags.DEFINE_bool('nerf_vis', True, 'use visibility volume')
flags.DEFINE_bool('anneal_freq', False, 'whether to use frequency annealing')
flags.DEFINE_integer('alpha', 10, 'maximum frequency for fourier features')
flags.DEFINE_bool('use_cc', True, 'whether to use connected component for mesh')

# model: motion
flags.DEFINE_bool('lbs', True, 'use lbs for backward warping 3d flow')
flags.DEFINE_integer('num_bones', 50, 'maximum number of bones')
flags.DEFINE_bool('nerf_skin', True, 'use mlp skinning function')
flags.DEFINE_integer('t_embed_dim', 128, 'dimension of the pose code')
flags.DEFINE_bool('frame_code', True, 'whether to use frame code')
flags.DEFINE_bool('flowbw', False, 'use backward warping 3d flow')
flags.DEFINE_bool('se3_flow', False, 'whether to use se3 field for 3d flow')

# model: cameras
flags.DEFINE_bool('use_cam', False, 'whether to use pre-defined camera pose')
flags.DEFINE_string('root_basis', 'expmlp', 'which root pose basis to use {mlp, cnn, exp}')
flags.DEFINE_bool('root_opt', True, 'whether to optimize root body poses')
flags.DEFINE_bool('ks_opt', True,   'whether to optimize camera intrinsics')

# optimization: hyperparams
flags.DEFINE_integer('num_epochs', 1000, 'Number of epochs to train')
flags.DEFINE_float('learning_rate', 5e-4, 'learning rate')
flags.DEFINE_integer('batch_size', 2, 'size of minibatches')
flags.DEFINE_integer('img_size', 512, 'image size for optimization')
flags.DEFINE_integer('nsample', 6, 'num of samples per image at optimization time')
flags.DEFINE_float('perturb',   1.0, 'factor to perturb depth sampling points')
flags.DEFINE_float('noise_std', 0., 'std dev of noise added to regularize sigma')
flags.DEFINE_float('nactive', 0.5, 'num of samples per image at optimization time')
flags.DEFINE_integer('ndepth', 128, 'num of depth samples per px at optimization time')
flags.DEFINE_float('clip_scale', 100, 'grad clip scale')
flags.DEFINE_float('warmup_steps', 0.4, 'steps used to increase sil loss')
flags.DEFINE_float('reinit_bone_steps', 0.667, 'steps to initialize bones')
flags.DEFINE_float('dskin_steps', 0.8, 'steps to add delta skinning weights')
flags.DEFINE_float('init_beta', 0.1, 'initial value for transparency beta')
flags.DEFINE_bool('reset_beta', False, 'reset volsdf beta to 0.1')
flags.DEFINE_float('fine_steps', 1.1, 'by default, not using fine samples')
flags.DEFINE_float('nf_reset', 0.5, 'by default, start reseting near-far plane at 50%')
flags.DEFINE_float('bound_reset', 0.5, 'by default, start reseting bound from 50%')
flags.DEFINE_float('bound_factor', 2, 'by default, use a loose bound')

# optimization: initialization 
flags.DEFINE_bool('init_ellips', False, 'whether to init shape as ellips')
flags.DEFINE_integer('warmup_pose_ep', 0, 'epochs to pre-train cnn pose predictor')
flags.DEFINE_integer('warmup_shape_ep', 0, 'epochs to pre-train nerf')
flags.DEFINE_bool('warmup_rootmlp', False, 'whether to preset base root pose (compatible with expmlp root basis only)')
flags.DEFINE_bool('unc_filter', False, 'whether to filter root poses init with low uncertainty')

# optimization: fine-tuning
flags.DEFINE_bool('keep_pose_basis', True, 'keep pose basis when loading models at train time')
flags.DEFINE_bool('freeze_coarse', False, 'whether to freeze coarse posec of MLP')
flags.DEFINE_bool('freeze_root', False, 'whether to freeze root body pose')
flags.DEFINE_bool('root_stab', True, 'whether to stablize root at ft')
flags.DEFINE_bool('freeze_cvf',  False, 'whether to freeze canonical features')
flags.DEFINE_bool('freeze_shape',False, 'whether to freeze canonical shape')
flags.DEFINE_bool('freeze_proj', False, 'whether to freeze some params w/ proj loss')
flags.DEFINE_bool('freeze_body_mlp', False, 'whether to freeze body pose mlp')
flags.DEFINE_float('proj_start', 0.0, 'steps to strat projection opt')
flags.DEFINE_float('frzroot_start', 0.0, 'steps to strat fixing root pose')
flags.DEFINE_float('frzbody_end', 0.0,   'steps to end fixing body pose')
flags.DEFINE_float('proj_end', 0.2,  'steps to end projection opt')

# CSE fine-tuning (turned off by default)
flags.DEFINE_bool('ft_cse', False, 'whether to fine-tune cse features')
flags.DEFINE_bool('mt_cse', True,  'whether to maintain cse features')
flags.DEFINE_float('mtcse_steps', 0.0, 'only distill cse before several epochs')
flags.DEFINE_float('ftcse_steps', 0.0, 'finetune cse after several epochs')

# render / eval
flags.DEFINE_integer('render_size', 64, 'size used for eval visualizations')
flags.DEFINE_integer('frame_chunk', 20, 'chunk size to split the input frames')
flags.DEFINE_integer('chunk', 32*1024, 'chunk size to split the input to avoid OOM')
flags.DEFINE_integer('rnd_frame_chunk', 3, 'chunk size to render eval images')
flags.DEFINE_bool('queryfw', True, 'use forward warping to query deformed shape')
flags.DEFINE_float('mc_threshold', -0.002, 'marching cubes threshold')
flags.DEFINE_bool('full_mesh', False, 'extract surface without visibility check')
flags.DEFINE_bool('ce_color', True, 'assign mesh color as canonical surface mapping or radiance')
flags.DEFINE_integer('sample_grid3d', 64, 'resolution for mesh extraction from nerf')
flags.DEFINE_string('test_frames', '9', 'a list of video index or num of frames, {0,1,2}, 30')

# losses
flags.DEFINE_bool('use_embed', False, 'whether to use feature consistency losses')
flags.DEFINE_bool('use_proj', False, 'whether to use reprojection loss')
flags.DEFINE_bool('use_corresp', True, 'whether to render and compare correspondence')
flags.DEFINE_bool('dist_corresp', True, 'whether to render distributed corresp')
flags.DEFINE_float('total_wt', 1, 'by default, multiple total loss by 1')
flags.DEFINE_float('sil_wt', 0.1, 'weight for silhouette loss')
flags.DEFINE_float('img_wt',  0.1, 'weight for silhouette loss')
flags.DEFINE_float('feat_wt', 0., 'by default, multiple feat loss by 1')
flags.DEFINE_float('frnd_wt', 1., 'by default, multiple feat loss by 1')
flags.DEFINE_float('proj_wt', 0.02, 'by default, multiple proj loss by 1')
flags.DEFINE_float('flow_wt', 1, 'by default, multiple flow loss by 1')
flags.DEFINE_float('cyc_wt', 1, 'by default, multiple cyc loss by 1')
flags.DEFINE_bool('rig_loss', False,'whether to use globally rigid loss')
flags.DEFINE_bool('root_sm', False, 'whether to use smooth loss for root pose')
flags.DEFINE_float('eikonal_wt', 0., 'weight of eikonal loss')
flags.DEFINE_float('bone_loc_reg', 0.1, 'use bone location regularization')
flags.DEFINE_bool('loss_flt', True, 'whether to use loss filter')
flags.DEFINE_bool('rm_novp', True,'whether to remove loss on non-overlapping pxs')

# for scripts/visualize/match.py
flags.DEFINE_string('match_frames', '0 1', 'a list of frame index')

class banmo(nn.Module):
    writer: SummaryWriter
    save_dir : str
    camera_radius : float
    target : int
    def __init__(self, opts, data_info, extra_opt):
        super(banmo, self).__init__()
        self.opts = opts
        self.device = torch.device("cuda:%d"%opts.local_rank)
        self.config = configparser.RawConfigParser()
        self.config.read('configs/%s.config'%opts.seqname)
        self.alpha=torch.Tensor([opts.alpha])
        self.alpha=nn.Parameter(self.alpha)
        self.loss_select = 1 # by default,  use all losses
        self.root_update = 1 # by default, update root pose
        self.body_update = 1 # by default, update body pose
        self.shape_update = 0 # by default, update all
        self.cvf_update = 0 # by default, update all
        self.progress = 0. # also reseted in optimizer
        self.counter_frz_rebone = 0. # counter to freeze params for reinit bones
        self.use_fine = False # by default not using fine samples
        #self.ndepth_bk = opts.ndepth # original ndepth
        self.root_basis = opts.root_basis
        self.use_cam = opts.use_cam
        self.is_warmup_pose = False # by default not warming up
        self.img_size = opts.img_size # current rendering size, 
                                      # have to be consistent with dataloader, 
                                      # eval/train has different size
        self.num_epochs = opts.num_epochs
        embed_net = nn.Embedding
        
        # multi-video mode
        self.num_vid =  len(self.config.sections())-1
        self.data_offset = data_info['offset']
        self.num_fr=self.data_offset[-1]  
        self.max_ts = (self.data_offset[1:] - self.data_offset[:-1]).max()
        self.impath      = data_info['impath']
        self.latest_vars = {}
        self.latest_vars['rtk'] = np.zeros((self.data_offset[-1], 4,4))
        
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
        for param in self.lpips.parameters(): 
            param.requires_grad=False
        
        self.bound = .5
        self.obj_scale = 10.
        # set shape/appearancce model
        self.num_freqs = 10
        in_channels_xyz=3+3*self.num_freqs*2
        in_channels_dir=27
        self.gs_code_dim=9
        self.use_delta=False
            
        self.gaussians = GaussianModel(0,self.bound,code_dim=self.gs_code_dim)
        self.gaussians.active_sh_degree = 0
        # parser = argparse.ArgumentParser()
        # op = OptimizationParams(parser)
        # op.iterations = self.num_epochs*self.num_fr
        # op.densify_until_iter = self.num_epochs*self.num_fr*(2/4)
        op = extra_opt
        # self.gaussians.random_init(sphere=True,disturb=False)
        self.gaussians.initialize(num_pts=op.num_pts*10,radius=self.bound)
        self.gaussians.training_setup(op)
        self.optim = op
        self.camera_radius = op.radius
        self.networks= {}
        
        self.use_diffusion = True
        if self.use_diffusion:
            from nnutils.zero123_utils import Zero123
            self.guidance_zero123 = Zero123(self.device)
            # from nnutils.imagedream_utils import ImageDream
            # self.guidance_zero123 = ImageDream(self.device)
            # self.neg_prompt = "ugly, bad anatomy, blurry, pixelated obscure, unnatural colors, poor lighting, dull, and unclear, \
            # cropped, lowres, low quality, artifacts, duplicate, morbid, mutilated, poorly drawn face, deformed, dehydrated, bad proportions"
        
        # import pdb;pdb.set_trace()
        # env embedding
        
        if opts.env_code:
            # add video-speficit environment lighting embedding
            env_code_dim = 64
            if opts.env_fourier:
                self.env_code = FrameCode(self.num_freqs, env_code_dim, self.data_offset, scale=1)
            else:
                self.env_code = embed_net(self.num_fr, env_code_dim)
        else:
            env_code_dim = 0
            
        # pose embedding
        t_embed_dim = opts.t_embed_dim
        if opts.frame_code:
            self.pose_code = FrameCode(self.num_freqs, t_embed_dim, self.data_offset)
        else:
            self.pose_code = embed_net(self.num_fr, t_embed_dim)
        
        # bone transform
        if opts.lbs:
            self.num_bones = opts.num_bones
            print('num_bones:',self.num_bones)
            bones= generate_bones(self.num_bones, self.num_bones, 0, self.device)
            self.bones = nn.Parameter(bones)
            self.num_bone_used = self.num_bones # bones used in the model
            self.nerf_body_rts = nn.Sequential(self.pose_code,
                                RTHead(use_quat=False, 
                                #D=5,W=128,
                                in_channels_xyz=t_embed_dim,in_channels_dir=0,
                                out_channels=6*self.num_bones, raw_feat=True))
            self.networks['bones'] = self.bones
            self.networks['nerf_body_rts'] = self.nerf_body_rts
            self.bone_colors = torch.rand(self.num_bones,3).to(self.device)
            self.bones_rts_frame = nn.Parameter(torch.zeros(self.num_fr,self.num_bones,6),requires_grad=True)
            self.bone_optimizer = torch.optim.Adam([self.bones_rts_frame],lr=0.0005)
            
        # skinning weights
        if opts.nerf_skin:
            self.nerf_skin =  MLP(code_dim=self.gs_code_dim,pose_dim=t_embed_dim,output_dim=self.num_bones)
            self.nerf_skin = NeRF(in_channels_xyz=in_channels_xyz+t_embed_dim,
#                                    D=5,W=128,
                                D=4,W=64,
                    in_channels_dir=0, out_channels=self.num_bones, 
                    raw_feat=True, in_channels_code=t_embed_dim)
            self.rest_pose_code = embed_net(1, t_embed_dim)
            self.networks['nerf_skin'] = self.nerf_skin
            self.networks['rest_pose_code'] = self.rest_pose_code
            skin_aux = torch.Tensor([0,self.obj_scale]) 
            self.skin_aux = nn.Parameter(skin_aux)
            self.networks['skin_aux'] = self.skin_aux

        # delta color, opacity...
        if True:
            self.delta_net = MLP(code_dim=self.gs_code_dim,pose_dim=env_code_dim,output_dim=11)
            self.networks['delta_net'] = self.delta_net
            self.use_delta_scale=False
            
        # optimize camera
        if opts.root_opt:
            # train a cnn pose predictor for warmup
            cnn_in_channels = 16

            cnn_head = RTHead(use_quat=True, D=1,
                        in_channels_xyz=128,in_channels_dir=0,
                        out_channels=7, raw_feat=True)
            self.dp_root_rts = nn.Sequential(
                            Encoder((112,112), in_channels=cnn_in_channels,
                                out_channels=128), cnn_head)
            self.nerf_root_rts = RTExpMLP(self.num_fr, 
                                self.num_freqs,t_embed_dim,self.data_offset,
                                delta=self.use_cam)
            self.networks['nerf_root_rts'] = self.nerf_root_rts

        # intrinsics
        ks_list = []
        for i in range(self.num_vid):
            fx,fy,px,py=[float(i) for i in \
                    self.config.get('data_%d'%i, 'ks').split(' ')]
            ks_list.append([fx,fy,px,py])
        self.ks_param = torch.Tensor(ks_list).to(self.device)
        if opts.ks_opt:
            self.ks_param = nn.Parameter(self.ks_param)
        


        self.resnet_transform = torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])

    def save_bones(self):
        torch.save(self.bones_rts_frame.detach().cpu(),osp.join(self.save_dir,'bones_rts_frame.pth'))
        torch.save(self.bones.detach().cpu(),osp.join(self.save_dir,'bones.pth'))
        torch.save(self.bone_colors.detach().cpu(),osp.join(self.save_dir,'bone_colors.pth'))
        
    def load_bones(self,path):
        self.bones_rts_frame = nn.Parameter(torch.load(osp.join(path,'bones_rts_frame.pth')).to(self.device))
        self.bones = nn.Parameter(torch.load(osp.join(path,'bones.pth')).to(self.device))
        self.bone_colors = torch.load(osp.join(path,'bone_colors.pth')).to(self.device)

    def main_render(self,rtk,img_size,embedid,use_dskin=False,background=torch.tensor([0.,0.,0.]).cuda(),canonical=False,render_xyz=False,get_rigid_loss=False,bone_color=False):
        # rtk: (4,4)
        H,W = img_size
        R = rtk[:3,:3]
        T = rtk[:3,3:]
        K = rtk[3]
        FovX = 2. * torch.arctan(W / (2. * K[0]))
        FovY = 2. * torch.arctan(H / (2. * K[1]))
        cam = Camera(0, R, T, FovX, FovY,
                 None, None, '', 0, width=W, height=H)
        
        
        gaussian_xyz =self.gaussians._xyz
        # gaussian_code = self.gaussians._code
        try:
            color_replace=self.color_replace
        except:
            color_replace=None
        if bone_color:
            bones_rst = self.bones.clone()
            rest_pose_code =  self.rest_pose_code
            rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(self.device))
            rts_head = self.nerf_body_rts[1]
            bone_rts_rst = rts_head(rest_pose_code)[0]
            bones_rst = bone_transform(bones_rst, bone_rts_rst, is_vec=True)[0] 
            bone_rts_fw = self.flatten(self.bones_rts_frame[embedid][0])+self.nerf_body_rts(embedid)/100.
            bone_fw_trans = correct_rest_pose(self.num_bones, bone_rts_fw, bone_rts_rst)
            skin = skinning(bones_rst, gaussian_xyz[None,...], skin_aux=self.skin_aux).squeeze()
            
            color_replace = (skin.squeeze()[...,None]*self.bone_colors).sum(dim=1).detach()
            if canonical:
                bone_new = bones_rst
            else:
                bone_new = bone_transform(self.bones, bone_rts_fw, is_vec=True)[0]
            color_new = self.bone_colors
            # bone_vis = render_points(cam, bone_new, color_new, background)
        
        if canonical:
            # output = render(cam,self.gaussians,background)
            if bone_color:
                output = render_auto(cam,self.gaussians._xyz,None,
                                self.gaussians,background,color_replace=color_replace,rot_delta=None,append_points=bone_new,append_color=color_new)
                # output['bone_vis'] = bone_vis['render']
            else:
                output = render(cam,self.gaussians,background)
            return output
        
        
        if not bone_color:
            bones_rst = self.bones.clone()
            rest_pose_code =  self.rest_pose_code
            rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(self.device))
            rts_head = self.nerf_body_rts[1]
            bone_rts_rst = rts_head(rest_pose_code)[0]
            bones_rst = bone_transform(bones_rst, bone_rts_rst, is_vec=True)[0] 
            bone_rts_fw = self.flatten(self.bones_rts_frame[embedid][0])+self.nerf_body_rts(embedid)/100.
            bone_fw_trans = correct_rest_pose(self.num_bones, bone_rts_fw, bone_rts_rst)
            skin = skinning(bones_rst, gaussian_xyz[None,...], skin_aux=self.skin_aux).squeeze()
            
        
        gaussian_xyz_dfm, rots = lbs(bones_rst[None,...], bone_fw_trans[None,...], skin[None,...], 
                gaussian_xyz[None,...],backward=False)
        gaussian_xyz_dfm = gaussian_xyz_dfm[0]
        
        # deltas = self.delta_net(gaussian_code,env_code)
        
        colors = self.gaussians._features_dc
        f_rest = self.gaussians._features_rest
        
        # d_color, d_scale, d_rotation, d_opacity = torch.split(deltas,[3,3,4,1],dim=-1)
        # colors_t = colors+d_color.unsqueeze(1)
        colors_t = colors
    
        
            
        jac = get_jacobian(gaussian_xyz_dfm,self.gaussians._xyz)
        
        append_points = None
        append_color  = None
        if bone_color:
            append_points = bone_new
            append_color = color_new
        # import pdb;pdb.set_trace()
        output = render_auto(cam,gaussian_xyz_dfm,torch.cat([colors_t,f_rest],dim=-2),
                             self.gaussians,background,color_replace=color_replace,rot_delta=jac,append_points=append_points,append_color=append_color)#.transpose(1,2))
        
        
        if get_rigid_loss:
            rig_loss = rigid_loss(jac)
            output['rigid_loss'] = rig_loss
        
        # if bone_color:
        #     output['bone_vis'] = bone_vis['render']
        
        if render_xyz:
            output_xyz = render_auto(cam,gaussian_xyz_dfm,torch.cat([colors_t,f_rest],dim=-2),
                             self.gaussians,background,color_replace=self.gaussians._xyz,rot_delta=jac)
            output['location'] = output_xyz['render'].moveaxis(0,-1)
            alpha = output_xyz['mask'].squeeze()
            output['location'][alpha>0.] /= alpha[alpha>0.][...,None]
        return output
        
        
    def flatten(self,rts):
        tmat= rts[:,0:3] *0.1
        rot=rts[:,3:6]
        rmat = transforms.so3_exponential_map(rot)
        rmat = rmat.view(-1,9)
        rts = torch.cat([rmat,tmat],-1)
        rts = rts.view(1,1,-1)
        return rts
    
    def bone_render(self,rtk,img_size,embedid,background=torch.tensor([0.,0.,0.]).cuda(),canonical=False,render_xyz=False,get_rigid_loss=False,bone_color=False):
        # rtk: (4,4)
        H,W = img_size
        R = rtk[:3,:3]
        T = rtk[:3,3:]
        K = rtk[3]
        FovX = 2. * torch.arctan(W / (2. * K[0]))
        FovY = 2. * torch.arctan(H / (2. * K[1]))
        cam = Camera(0, R, T, FovX, FovY,
                 None, None, '', 0, width=W, height=H)
        
        
        gaussian_xyz =self.gaussians._xyz
        # gaussian_code = self.gaussians._code
        try:
            color_replace=self.color_replace
        except:
            color_replace=None
        if bone_color:
            bones_rst = self.bones.clone()
            rest_pose_code =  self.rest_pose_code
            rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(self.device))
            rts_head = self.nerf_body_rts[1]
            bone_rts_rst = rts_head(rest_pose_code)[0]
            bones_rst = bone_transform(bones_rst, bone_rts_rst, is_vec=True)[0] 
            bone_rts_fw = self.flatten(self.bones_rts_frame[embedid][0])
            bone_fw_trans = correct_rest_pose(self.num_bones, bone_rts_fw, bone_rts_rst)
            skin = skinning(bones_rst, gaussian_xyz[None,...], skin_aux=self.skin_aux).squeeze()
            
            color_replace = (skin.squeeze()[...,None]*self.bone_colors).sum(dim=1).detach()
            if canonical:
                bone_new = bones_rst
            else:
                bone_new = bone_transform(self.bones, bone_rts_fw, is_vec=True)[0]
            color_new = self.bone_colors
            # bone_vis = render_points(cam, bone_new, color_new, background)
        
        if canonical:
            # output = render(cam,self.gaussians,background)
            if bone_color:
                output = render_auto(cam,self.gaussians._xyz,None,
                                self.gaussians,background,color_replace=color_replace,rot_delta=None,append_points=bone_new,append_color=color_new)
                # output['bone_vis'] = bone_vis['render']
            else:
                output = render(cam,self.gaussians,background)
            return output
        
        
        if not bone_color:
            bones_rst = self.bones.clone()
            rest_pose_code =  self.rest_pose_code
            rest_pose_code = rest_pose_code(torch.Tensor([0]).long().to(self.device))
            rts_head = self.nerf_body_rts[1]
            bone_rts_rst = rts_head(rest_pose_code)[0]
            bones_rst = bone_transform(bones_rst, bone_rts_rst, is_vec=True)[0] 
            bone_rts_fw = self.flatten(self.bones_rts_frame[embedid][0])
            bone_fw_trans = correct_rest_pose(self.num_bones, bone_rts_fw, bone_rts_rst)
            skin = skinning(bones_rst, gaussian_xyz[None,...], skin_aux=self.skin_aux).squeeze()
            
        
        gaussian_xyz_dfm, rots = lbs(bones_rst[None,...], bone_fw_trans[None,...], skin[None,...], 
                gaussian_xyz[None,...],backward=False)
        gaussian_xyz_dfm = gaussian_xyz_dfm[0]
        
        
        colors = self.gaussians._features_dc
        f_rest = self.gaussians._features_rest
        
        colors_t = colors
    
        
            
        jac = get_jacobian(gaussian_xyz_dfm,self.gaussians._xyz)
        
        append_points = None
        append_color  = None
        if bone_color:
            append_points = bone_new
            append_color = color_new
        # import pdb;pdb.set_trace()
        # print(rots.shape,jac.shape)
        output = render_auto(cam,gaussian_xyz_dfm,torch.cat([colors_t,f_rest],dim=-2),
                             self.gaussians,background,color_replace=color_replace,rot_delta=jac,append_points=append_points,append_color=append_color)#.transpose(1,2))
        
        
        if get_rigid_loss:
            rig_loss = rigid_loss(jac)
            output['rigid_loss'] = rig_loss
        
        
        if render_xyz:
            output_xyz = render_auto(cam,gaussian_xyz_dfm,torch.cat([colors_t,f_rest],dim=-2),
                             self.gaussians,background,color_replace=self.gaussians._xyz,rot_delta=jac)
            output['location'] = output_xyz['render'].moveaxis(0,-1)
            alpha = output_xyz['mask'].squeeze()
            output['location'][alpha>0.] /= alpha[alpha>0.][...,None]
        return output
        
        
    def visualize(self,fid,theta=np.pi,fai=0.,canonical=False,bone_color=False):
        angle = theta
        fai = fai
        Rmat = torch.tensor([[-np.sin(angle),0.,-np.cos(angle)],
                                [-np.cos(angle)*np.sin(fai),-np.cos(fai),np.sin(angle)*np.sin(fai)],    
                                [-np.cos(angle)*np.cos(fai),np.sin(fai),np.sin(angle)*np.cos(fai)]])
        Tmat = torch.tensor([0.,0.,1.])*2.
        K = torch.tensor([512.,512.,256.,256.])
        H=W=512
        rtk = torch.zeros(4,4).cuda()
        rtk[:3,:3] = Rmat
        rtk[:3,3] = Tmat
        rtk[3,:] = K
        background = torch.tensor([1.,1.,1.]).cuda()
        id = torch.tensor([fid],device=rtk.device)
        results = self.bone_render(rtk,(H,W),id,
                                background=background,canonical=canonical,bone_color=bone_color)
        rgb = results['render'].clamp(0,1)
        rgb = rgb.moveaxis(0,-1).detach().cpu().numpy()
        return rgb
        
        
    def get_delta_loss(self):
        if not self.use_delta:
            return 0
        embedid = torch.range(0,self.num_fr).cuda().long()
        env_code = self.env_code(embedid)
        deltas = self.delta_net(self.gaussians._code,env_code).squeeze()
        delta_loss = (deltas.mean(dim=0)**2).mean()
        delta_weight = 0.1
        return delta_loss*delta_weight
        
        
        
    def warmup_canonical(self, batch, save_dir):

        # import pdb;pdb.set_trace()
        # torch.cuda.empty_cache()
        opts = self.opts
        rtk = self.compute_rts(torch.Tensor([batch['frameid']]).to(self.device).long())[0]
        rtk = torch.cat([rtk,self.ks_param[torch.Tensor([batch['dataid']]).to(self.device).long()]],dim=0).detach()
        gt_theta, gt_phi = angles_from_rtk(rtk)
        gt_theta = 0.
        gt_phi = 0.
        rtk = rtk_from_angles(gt_theta, gt_phi,radius = self.camera_radius)
        rtk[3] = self.ks_param[torch.Tensor([batch['dataid']]).to(self.device).long()]
        rtk[3,3] = 960.
        rtk[3,2] = 960.
        embedid=torch.Tensor([batch['frameid']]).to(self.device).long()+self.data_offset[torch.Tensor([batch['dataid']]).long()]
        embedid.requires_grad=False
        H = batch['img'].shape[-2]
        W = batch['img'].shape[-1]
        background = torch.tensor([1.,1.,1.],requires_grad=False).cuda().float()
        # import pdb;pdb.set_trace()
        # gt_image = gt_image*gt_mask+(1.-gt_mask)*(background[None,...,None,None])
        
        # input_img = gt_image[None,...]* 2. - 1.
        
        # import pdb;pdb.set_trace()
        img_size=(1920,1920)
        with torch.no_grad():
            batch_img = torch.tensor(batch['img'],requires_grad=False).cuda().squeeze().float()
            batch_mask = torch.tensor(batch['mask'],requires_grad=False).cuda().squeeze().float()
            batch_img = batch_img*batch_mask+(1-batch_mask)
            if W>H:
                img_input = torch.cat([torch.ones([3,420,1920],dtype=torch.float32,device=self.device),batch_img,torch.ones([3,420,1920],dtype=torch.float32,device=self.device)],dim=1)[None,...]
                mask_input = torch.cat([torch.zeros([1,420,1920],dtype=torch.float32,device=self.device),batch_mask[None,...],torch.zeros([1,420,1920],dtype=torch.float32,device=self.device)],dim=1)[None,...]
            elif W<H:
                img_input = torch.cat([torch.ones([3,1920,420],dtype=torch.float32,device=self.device),batch_img,torch.ones([3,1920,420],dtype=torch.float32,device=self.device)],dim=2)[None,...]
                mask_input = torch.cat([torch.zeros([1,1920,420],dtype=torch.float32,device=self.device),batch_mask[None,...],torch.zeros([1,1920,420],dtype=torch.float32,device=self.device)],dim=2)[None,...]
            else:
                img_input = batch_img[None,...]
                mask_input = batch_mask[None,None,...]
                img_size=(H,W)
            print("W=%d,H=%d"%(W,H))
            print("img_input.shape",img_input.shape)
            print("mask_input.shape",mask_input.shape)
            # batch_sdf = torch.from_numpy(batch['sdf']).squeeze().cuda()
            input_im = F.interpolate(img_input, (self.optim.ref_size, self.optim.ref_size), mode="bilinear", align_corners=False)
            input_mask = F.interpolate(mask_input, (self.optim.ref_size, self.optim.ref_size), mode="bilinear", align_corners=False)
            # get input embedding
            # self.diffusion_model.clip_emb = self.diffusion_model.model.get_learned_conditioning(input_im.float()).tile(1,1,1).detach()
            # self.diffusion_model.vae_emb = self.diffusion_model.model.encode_first_stage(input_im.float()).mode().detach()
            # self.guidance_zero123.get_image_text_embeds(input_im, ["dog"], [self.neg_prompt])
            c_list, v_list = [], []
            c, v = self.guidance_zero123.get_img_embeds(input_im)
            for _ in range(self.optim.batch_size):
                c_list.append(c)
                v_list.append(v)
            if self.use_diffusion:
                self.guidance_zero123.embeddings = [torch.cat(c_list, 0), torch.cat(v_list, 0)]
            
        n_steps = self.optim.iters
        
        self.warmup_step = 0
        # import pdb;pdb.set_trace()
        for i in tqdm(range(n_steps)):
            # import pdb;pdb.set_trace()
            self.warmup_step += 1
            step_ratio = min(1, self.warmup_step / self.optim.iters)
            self.gaussians.update_learning_rate(self.warmup_step)
            loss = 0.
            
            rendered = self.main_render(rtk,img_size,embedid,background=torch.tensor([1.,1.,1.],requires_grad=False).cuda().float(),canonical=True)
            image = rendered['render']
            mask = rendered['mask']
            depth = rendered['depth']
            image_loss = 10000 * step_ratio * F.mse_loss(image[None,...], img_input)
            mask_loss = 1000 * step_ratio * F.mse_loss(mask[None,...], mask_input)
            input_smooth_loss = 0.# 1000 * step_ratio * depth_smooth_loss(depth.squeeze())
            loss += image_loss+mask_loss+input_smooth_loss
            
            render_resolution = 128 if step_ratio < 0.3 else (256 if step_ratio < 0.6 else 512)
            images = []
            poses = []
            vers, hors, radii = [], [], []
            min_ver = max(min(-30, -30 - gt_phi/np.pi*180.), -80 - gt_phi/np.pi*180.)
            max_ver = min(max(30, 30 - gt_phi/np.pi*180.), 80 - gt_phi/np.pi*180.)
            for _ in range(self.optim.batch_size):

                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                # radius = 0
                radius = np.random.uniform(-0.5, 0.5) 

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = rtk_from_angles(hor/180.*np.pi+gt_theta,ver/180.*np.pi+gt_phi,render_resolution,self.camera_radius+radius)
                poses.append(orbit_camera(ver+gt_phi/np.pi*180.,hor+gt_theta/np.pi*180.,render_resolution,self.camera_radius+radius))
                # poses.append(pose)

                # cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.optim.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                out = self.main_render(pose,(render_resolution,render_resolution),embedid,background=bg_color,canonical=True)
                # out = self.renderer.render(cur_cam, bg_color=bg_color)

                image = out["render"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                images.append(image)
                
                # if self.use_diffusion:
                #     for view_i in range(1, 4):
                #         pose = rtk_from_angles(hor/180.*np.pi+np.pi/2.*view_i+gt_theta+np.pi,ver/180.*np.pi+gt_phi,render_resolution,self.camera_radius+radius)
                #         poses.append(orbit_camera(ver+gt_phi/np.pi*180.,hor+gt_theta/np.pi*180+view_i*90.,render_resolution,self.camera_radius+radius))
                #         # pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                #         # poses.append(pose_i)

                #         out = self.main_render(pose,(render_resolution,render_resolution),embedid,background=bg_color,canonical=False)

                #         image = out["render"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                #         images.append(image)
            images = torch.cat(images, dim=0)
            poses = torch.from_numpy(np.stack(poses, axis=0)).to(self.device)
                    
            zero123_loss = self.optim.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio) / self.optim.batch_size
            # zero123_loss = self.optim.lambda_zero123 * self.guidance_zero123.train_step(images, poses, step_ratio=step_ratio)
            loss += zero123_loss*0.1
            loss.backward()
            self.gaussians.optimizer.step()
            self.gaussians.optimizer.zero_grad()
            if self.warmup_step >= self.optim.density_start_iter and self.warmup_step <= self.optim.density_end_iter:
                viewspace_point_tensor, visibility_filter, radii = out["viewspace_points"], out["visibility_filter"], out["radii"]
                self.gaussians.max_radii2D[visibility_filter] = torch.max(self.gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                self.gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if self.warmup_step % self.optim.densification_interval == 0:
                    # size_threshold = 20 if self.warmup_step > self.optim.opacity_reset_interval else None
                    pnum = self.gaussians.densify_and_prune(self.optim.densify_grad_threshold, min_opacity=0.01, extent=0.2, max_screen_size=1)
                
                if self.warmup_step % self.optim.opacity_reset_interval == 0:
                    self.gaussians.reset_opacity()
            if i%100==0:
                with torch.no_grad():
                    self.save_imgs(save_dir,i,0,use_deform=False,novel_cam=True,save_img=False)
            if i%10==0:
                cv2.imwrite('test.png',images[0].squeeze().moveaxis(0,-1)[:,:,[2,1,0]].detach().cpu().numpy()*255.)
                cv2.imwrite('origin.png',rendered['render'].squeeze().moveaxis(0,-1)[:,:,[2,1,0]].detach().cpu().numpy()*255.)
        with torch.no_grad():
            self.save_imgs(save_dir,n_steps,0,use_deform=False,novel_cam=True,save_img=False)
                # import pdb;pdb.set_trace()
            # if i == 5000:
            #     self.gaussians.prune_points((self.gaussians.get_opacity <= 0.005).squeeze())
            # torch.cuda.empty_cache()
        # import pdb;pdb.set_trace()
        
        
    
    
    def forward_default(self, batch, frame_bone=False):
        # import pdb;pdb.set_trace()
        opts = self.opts
        # get root poses
        step_ratio = 0.98
        
        
        rtk_all = self.compute_rts()
        self.rtk_all = rtk_all
        try:
            rtk = self.compute_rts(batch['frameid'].to(self.device).long())[0]
        except:
            import pdb;pdb.set_trace()
        rtk = torch.cat([rtk,self.ks_param[batch['dataid'].to(self.device).long()]],dim=0)
        gt_theta, gt_phi = angles_from_rtk(rtk)
        gt_theta = 0.
        gt_phi = 0.
        rtk = rtk_from_angles(gt_theta, gt_phi,radius = self.camera_radius)
        embedid=batch['frameid'].to(self.device)+self.data_offset[batch['dataid'].long()]
        H = batch['img'].shape[-2]
        W = batch['img'].shape[-1]
        S = max(H,W)
        
        use_flow = False
        # Render
        background = torch.tensor([1.,1.,1.],requires_grad=False).cuda()
        
        # import pdb;pdb.set_trace()
        if frame_bone:
            rendered = self.bone_render(rtk,(S,S),embedid,background=background,canonical=False,render_xyz=use_flow,get_rigid_loss=True)
        else:
            rendered = self.main_render(rtk,(S,S),embedid,background=background,canonical=False,render_xyz=use_flow,get_rigid_loss=True)
            
        aux_out = rendered
        if W>H:
            image = rendered['render'].squeeze()[:,420:-420]
            mask = rendered['mask'].squeeze()[420:-420]
        elif W<H:
            image = rendered['render'].squeeze()[:,:,420:-420]
            mask = rendered['mask'].squeeze()[:,420:-420]
        else:
            image = rendered['render'].squeeze()
            mask = rendered['mask'].squeeze()
        depth = rendered['depth'].squeeze()
        gt_image = batch['img'].squeeze().cuda().float()
        gt_mask = batch['mask'].squeeze().cuda().float()
        gt_sdf = batch['sdf'].squeeze().cuda().float()
        gt_flow = batch['flow'].squeeze().cuda().float()
        # import pdb;pdb.set_trace()
        
        
        # import pdb;pdb.set_trace()
        gt_image = gt_image*gt_mask+(1.-gt_mask)*(background[...,None,None])
        if W>H:
            img_input = torch.cat([torch.ones([3,420,1920],dtype=torch.float32,device=self.device),gt_image,torch.ones([3,420,1920],dtype=torch.float32,device=self.device)],dim=1)[None,...]
        elif W<H:
            img_input = torch.cat([torch.ones([3,1920,420],dtype=torch.float32,device=self.device),gt_image,torch.ones([3,1920,420],dtype=torch.float32,device=self.device)],dim=2)[None,...]
        # Ll1 = l1_loss(image, gt_image)
        # lambda_dssim = 0.2
        
        
        # # image loss
        # import pdb;pdb.set_trace()
        # img_loss = (1.0 - lambda_dssim) * Ll1 + lambda_dssim * (1.0 - ssim((image)[None,...], (gt_image)[None,...]))
        img_weight = 1
        mask_weight = 0.1
        lpips_weight = 0.1
        rig_weight = 0.4
        
        img_loss = F.mse_loss(image[None,...], (gt_image)[None,...])
        aux_out['img_loss'] = img_loss.data
        
        total_loss = img_loss*img_weight# * 10000  * step_ratio
        
        lpips_loss = self.lpips(image[None,...].clip(0.,1.), gt_image[None,...].clip(0.,1.))
        # 1000. * step_ratio
        total_loss += lpips_loss*lpips_weight
        
        #depth smooth loss
        # input_smooth_loss = depth_smooth_loss(depth.squeeze())
        # smooth_weight = 1000 * step_ratio
        # total_loss += input_smooth_loss*smooth_weight
        # aux_out['input_smooth_loss'] = input_smooth_loss.data
        
        # mask_loss = F.mse_loss(mask[None,...], gt_mask[None,...])
        mask_loss = l1_loss(mask,gt_mask)
        # mask_loss = sdf_loss(gt_mask,mask,gt_sdf)*0.1
        # 1000  * step_ratio
        total_loss += mask_loss*mask_weight
        aux_out['mask_loss'] = mask_loss.data
        
        if self.use_diffusion:
            total_loss *=1
        
        # rigid loss
        rig_loss = rendered['rigid_loss']
        
        total_loss += rig_loss*rig_weight
        # aux_out['rig_loss'] = rig_loss.data
        # import pdb;pdb.set_trace()
        
            
        # novel view diffusion preparing
        rtk_novel = None
        if self.use_diffusion:
            with torch.no_grad():
                # img_input = torch.cat([torch.ones([3,420,1920],dtype=torch.float32,device=self.device),gt_image,torch.ones([3,420,1920],dtype=torch.float32,device=self.device)],dim=1)[None,...]
                input_im = F.interpolate(img_input, (self.optim.ref_size, self.optim.ref_size), mode="bilinear", align_corners=False)
                c_list, v_list = [], []
                c, v = self.guidance_zero123.get_img_embeds(input_im)
                # import pdb;pdb.set_trace()
                for _ in range(self.optim.batch_size_dynamic):
                    c_list.append(c)
                    v_list.append(v)
                self.guidance_zero123.embeddings = [torch.cat(c_list, 0), torch.cat(v_list, 0)]
        # diffusion loss
        if self.use_diffusion:
            images = []
            poses = []
            vers, hors, radii = [], [], []
            min_ver = max(min(-30, -30 - gt_phi/np.pi*180.), -80 - gt_phi/np.pi*180.)
            max_ver = min(max(30, 30 - gt_phi/np.pi*180.), 80 - gt_phi/np.pi*180.)
            
            render_resolution = 512
            for _ in range(self.optim.batch_size_dynamic):

                # render random view
                ver = np.random.randint(min_ver, max_ver)
                hor = np.random.randint(-180, 180)
                # hor = -180
                # radius = 0
                radius = np.random.uniform(-0.5, 0.5) 

                vers.append(ver)
                hors.append(hor)
                radii.append(radius)

                pose = rtk_from_angles(hor/180.*np.pi+gt_theta,ver/180.*np.pi+gt_phi,render_resolution,self.camera_radius+radius)
                poses.append(orbit_camera(ver/180.*np.pi+gt_phi,hor/180.*np.pi+gt_theta,render_resolution,self.camera_radius+radius))

                # cur_cam = MiniCam(pose, render_resolution, render_resolution, self.cam.fovy, self.cam.fovx, self.cam.near, self.cam.far)

                bg_color = torch.tensor([1, 1, 1] if np.random.rand() > self.optim.invert_bg_prob else [0, 0, 0], dtype=torch.float32, device="cuda")
                if frame_bone:
                    out = self.bone_render(pose,(render_resolution,render_resolution),embedid,background=bg_color,canonical=False)
                else:
                    out = self.main_render(pose,(render_resolution,render_resolution),embedid,background=bg_color,canonical=False)
                # out = self.renderer.render(cur_cam, bg_color=bg_color)

                image = out["render"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                if random.random() < 0.02:
                    cv2.imwrite('test1.png',image.squeeze().moveaxis(0,-1)[:,:,[2,1,0]].detach().cpu().numpy()*255.)
                images.append(image)
                
                # if self.use_diffusion:
                #     for view_i in range(1, 4):
                #         pose = rtk_from_angles(hor/180.*np.pi+np.pi/2.*view_i+gt_theta,ver/180.*np.pi+gt_phi,render_resolution,self.camera_radius+radius)
                #         poses.append(orbit_camera(ver/180.*np.pi+gt_phi,hor/180.*np.pi+np.pi/2.*view_i+gt_theta,render_resolution,self.camera_radius+radius))
                #         # pose_i = orbit_camera(self.opt.elevation + ver, hor + 90 * view_i, self.opt.radius + radius)
                #         # poses.append(pose_i)

                #         if frame_bone:
                #             out = self.bone_render(pose,(render_resolution,render_resolution),embedid,background=bg_color,canonical=False)
                #         else:
                #             out = self.main_render(pose,(render_resolution,render_resolution),embedid,background=bg_color,canonical=False)

                #         image = out["image"].unsqueeze(0) # [1, 3, H, W] in [0, 1]
                #         images.append(image)
                        
                        
            images = torch.cat(images, dim=0)
            # poses = torch.stack(poses, axis=0).to(self.device)
                    
            zero123_loss = self.optim.lambda_zero123 * self.guidance_zero123.train_step(images, vers, hors, radii, step_ratio) / self.optim.batch_size_dynamic
            aux_out['zero123_loss'] = zero123_loss
            total_loss += zero123_loss/1000.
            
        # save some variables
        if opts.lbs:
            aux_out['skin_scale'] = self.skin_aux[0].clone().detach()
            aux_out['skin_const'] = self.skin_aux[1].clone().detach()

        
          
          
  
        # flow loss
        # if opts.use_corresp:
        #     flo_loss_samp = rendered['flo_loss_samp']
        #     if opts.rm_novp:
        #         flo_loss_samp = flo_loss_samp * rendered['sil_coarse'].detach()

        #     # eval on valid pts
        #     flo_loss = flo_loss_samp[sil_at_samp_flo[...,0]].mean() * 2
        #     #flo_loss = flo_loss_samp[sil_at_samp_flo[...,0]].mean()
        #     flo_loss = flo_loss * opts.flow_wt
    
        #     # warm up by only using flow loss to optimize root pose
        #     if self.loss_select == 0:
        #         total_loss = total_loss*0. + flo_loss
        #     else:
        #         total_loss = total_loss + flo_loss
        #     aux_out['flo_loss'] = flo_loss
        
        
        if opts.lbs and opts.bone_loc_reg>0 and not frame_bone:
            bones_rst = self.bones
            bones_rst,_ = correct_bones(self, bones_rst)
            inds = np.random.choice(range(self.gaussians._xyz.shape[0]),size=(1000,),replace=False)
            shape_samp = self.gaussians._xyz[inds].detach()
            
            # shape_samp = shape_samp[0].to(self.device)
            from geomloss import SamplesLoss
            samploss = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
            bone_loc_loss = samploss(bones_rst[:,:3]*10, shape_samp*10)
            bone_loc_loss = opts.bone_loc_reg*bone_loc_loss
            total_loss = total_loss + bone_loc_loss
            aux_out['bone_loc_loss'] = bone_loc_loss


        #     # elastic energy for se3 field / translation field
        #     if 'elastic_loss' in rendered.keys():
        #         elastic_loss = rendered['elastic_loss'].mean() * 1e-3
        #         total_loss = total_loss + elastic_loss
        #         aux_out['elastic_loss'] = elastic_loss
        
        #     # elastic energy for se3 field / translation field
        #     if 'elastic_loss' in rendered.keys():
        #         elastic_loss = rendered['elastic_loss'].mean() * 1e-3
        #         total_loss = total_loss + elastic_loss
        #         aux_out['elastic_loss'] = elastic_loss

        # regularization of root poses
        # if opts.root_sm:
        #     # root_sm_loss_1st = compute_root_sm_loss(rtk_all, self.data_offset)
        #     root_sm_loss_2nd = compute_root_sm_2nd_loss(rtk_all, self.data_offset)
        #     # aux_out['root_sm_1st_loss'] = root_sm_loss_1st
        #     aux_out['root_sm_2nd_loss'] = root_sm_loss_2nd
        #     total_loss = total_loss + root_sm_loss_2nd



        
        
        # total_loss = total_loss * opts.total_wt
        # aux_out['total_loss'] = total_loss
        # aux_out['beta'] = self.nerf_coarse.beta.clone().detach()[0]
        # if opts.debug:
        #     torch.cuda.synchronize()
        #     print('set input + render + loss time:%.2f'%(time.time()-start_time))
        
        
        
        # total_loss = total_loss * opts.total_wt
        # aux_out['total_loss'] = total_loss
        # aux_out['beta'] = self.nerf_coarse.beta.clone().detach()[0]
        # if opts.debug:
        #     torch.cuda.synchronize()
        #     print('set input + render + loss time:%.2f'%(time.time()-start_time))
        return total_loss, aux_out


    
    

    @staticmethod
    def create_base_se3(bs, device, radius=1.0):
        """
        create a base se3 based on near-far plane
        """
        rt = torch.zeros(bs,3,4).to(device)
        rt[:,:3,:3] = torch.eye(3)[None].repeat(bs,1,1).to(device)
        rt[:,:2,3] = 0.
        rt[:,2,3] = radius
        return rt

    @staticmethod
    def prepare_ray_cams(rtk, kaug):
        """ 
        in: rtk, kaug
        out: Rmat, Tmat, Kinv
        """
        Rmat = rtk[:,:3,:3]
        Tmat = rtk[:,:3,3]
        Kmat = K2mat(rtk[:,3,:])
        Kaug = K2inv(kaug) # p = Kaug Kmat P
        Kinv = Kmatinv(Kaug.matmul(Kmat))
        return Rmat, Tmat, Kinv

        
    def update_delta_rts(self, rays):
        """
        change bone_rts_fw to delta fw
        """
        opts = self.opts
        bones_rst, bone_rts_rst = correct_bones(self, self.networks['bones'])
        self.networks['bones_rst']=bones_rst

        # delta rts
        rays['bone_rts'] = correct_rest_pose(opts, rays['bone_rts'], bone_rts_rst)

        if 'bone_rts_target' in rays.keys():       
            rays['bone_rts_target'] = correct_rest_pose(opts, 
                                            rays['bone_rts_target'], bone_rts_rst)

        if 'bone_rts_dentrg' in rays.keys():
            rays['bone_rts_dentrg'] = correct_rest_pose(opts, 
                                            rays['bone_rts_dentrg'], bone_rts_rst)


    def convert_batch_input(self, batch):
        device = self.device
        opts = self.opts
        if batch['img'].dim()==4:
            bs,_,h,w = batch['img'].shape
        else:
            bs,_,_,h,w = batch['img'].shape
        # convert to float
        for k,v in batch.items():
            try:
                batch[k] = batch[k].float()
            except:
                pass

        img_tensor = batch['img'].view(bs,-1,3,h,w).permute(1,0,2,3,4).reshape(-1,3,h,w)
        input_img_tensor = img_tensor.clone()
        for b in range(input_img_tensor.size(0)):
            input_img_tensor[b] = self.resnet_transform(input_img_tensor[b])
        
        self.input_imgs   = input_img_tensor.to(device)
        self.imgs         = img_tensor.to(device)
        self.masks        = batch['mask']        .view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)      .to(device)
        self.vis2d        = batch['vis2d']        .view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)     .to(device)
        self.dps          = batch['dp']          .view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)      .to(device)
        dpfd = 16
        dpfs = 112
        self.dp_feats     = batch['dp_feat']     .view(bs,-1,dpfd,dpfs,dpfs).permute(1,0,2,3,4).reshape(-1,dpfd,dpfs,dpfs).to(device)
        self.dp_bbox      = batch['dp_bbox']     .view(bs,-1,4).permute(1,0,2).reshape(-1,4)          .to(device)
        if opts.use_embed and opts.ft_cse and (not self.is_warmup_pose):
            self.dp_feats_mask = self.dp_feats.abs().sum(1)>0
            self.csepre_feats = self.dp_feats.clone()
            # unnormalized features
            self.csenet_feats, self.dps = self.csenet(self.imgs, self.masks)
            # for visualization
            self.dps = self.dps * self.dp_feats_mask.float()
            if self.progress > opts.ftcse_steps:
                self.dp_feats = self.csenet_feats
            else:
                self.dp_feats = self.csenet_feats.detach()
        self.dp_feats     = F.normalize(self.dp_feats, 2,1)
        self.rtk          = batch['rtk']         .view(bs,-1,4,4).permute(1,0,2,3).reshape(-1,4,4)    .to(device)
        self.kaug         = batch['kaug']        .view(bs,-1,4).permute(1,0,2).reshape(-1,4)          .to(device)
        self.frameid      = batch['frameid']     .view(bs,-1).permute(1,0).reshape(-1).cpu()
        self.dataid       = batch['dataid']      .view(bs,-1).permute(1,0).reshape(-1).cpu()
      
        self.frameid_sub = self.frameid.clone() # id within a video
        self.embedid = self.frameid + self.data_offset[self.dataid.long()]
        self.frameid = self.frameid + self.data_offset[self.dataid.long()]
        self.errid = self.frameid # for err filter
        self.rt_raw  = self.rtk.clone()[:,:3]

        # process silhouette
        self.masks = (self.masks*self.vis2d)>0
        self.masks = self.masks.float()
        
        self.flow = batch['flow'].view(bs,-1,2,h,w).permute(1,0,2,3,4).reshape(-1,2,h,w).to(device)
        self.occ  = batch['occ'].view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)     .to(device)
        self.lineid = None
        
    def convert_feat_input(self, batch):
        device = self.device
        opts = self.opts
        bs = batch['frameid'].shape[0]
        # convert to float
        for k,v in batch.items():
            batch[k] = batch[k].float()

        # self.dps          = batch['dp']          .view(bs,-1,h,w).permute(1,0,2,3).reshape(-1,h,w)      .to(device)
        dpfd = 16
        dpfs = 112
        self.dp_feats     = batch['dp_feat']     .view(bs,-1,dpfd,dpfs,dpfs).permute(1,0,2,3,4).reshape(-1,dpfd,dpfs,dpfs).to(device)
        self.dp_bbox      = batch['dp_bbox']     .view(bs,-1,4).permute(1,0,2).reshape(-1,4)          .to(device)
        self.dp_feats     = F.normalize(self.dp_feats, 2,1)
        self.rtk          = batch['rtk']         .view(bs,-1,4,4).permute(1,0,2,3).reshape(-1,4,4)    .to(device)
        self.kaug         = batch['kaug']        .view(bs,-1,4).permute(1,0,2).reshape(-1,4)          .to(device)
        self.frameid      = batch['frameid']     .view(bs,-1).permute(1,0).reshape(-1).cpu()
        self.dataid       = batch['dataid']      .view(bs,-1).permute(1,0).reshape(-1).cpu()
      
        self.frameid_sub = self.frameid.clone() # id within a video
        self.embedid = self.frameid + self.data_offset[self.dataid.long()]
        self.frameid = self.frameid + self.data_offset[self.dataid.long()]
        self.errid = self.frameid # for err filter
        self.rt_raw  = self.rtk.clone()[:,:3]

        self.lineid = None
    
    def convert_root_pose(self):
        """
        assumes has self.
        {rtk, frameid, dp_feats, dps, masks, kaug }
        produces self.
        """
        opts = self.opts
        bs = self.rtk.shape[0]
        device = self.device

        # scale initial poses
        if self.use_cam:
            self.rtk[:,:3,3] = self.rtk[:,:3,3] / self.obj_scale
        else:
            self.rtk[:,:3] = self.create_base_se3(bs, device)
  
        # compute delta pose
        if self.opts.root_opt:
            if self.root_basis == 'cnn':
                frame_code = self.dp_feats
            elif self.root_basis == 'mlp' or self.root_basis == 'exp'\
              or self.root_basis == 'expmlp':
                frame_code = self.frameid.long().to(device)
            else: print('error'); exit()
            root_rts = self.nerf_root_rts(frame_code)
            self.rtk = self.refine_rt(self.rtk, root_rts)

        self.rtk[:,3,:] = self.ks_param[self.dataid.long()] #TODO kmat

    @staticmethod
    def refine_rt(rt_raw, root_rts):
        """
        input:  rt_raw representing the initial root poses (after scaling)
        input:  root_rts representing delta se3
        output: current estimate of rtks for all frames
        """
        rt_raw = rt_raw.clone()
        root_rmat = root_rts[:,0,:9].view(-1,3,3)
        root_tmat = root_rts[:,0,9:12]
    
        rmat = rt_raw[:,:3,:3].clone()
        tmat = rt_raw[:,:3,3].clone()
        tmat = tmat + rmat.matmul(root_tmat[...,None])[...,0]
        rmat = rmat.matmul(root_rmat)
        rt_raw[:,:3,:3] = rmat
        rt_raw[:,:3,3] = tmat
        return rt_raw
      
    def compute_rts(self,frameid=None):
        """
        Assumpions
        - use_cam
        - use mlp or exp root pose 
        input:  rt_raw representing the initial root poses
        output: current estimate of rtks for all frames
        """
        device = self.device
        opts = self.opts
        if frameid is None:
            frameid = torch.Tensor(range(self.num_fr)).to(device).long()

        if self.use_cam:
            # scale initial poses
            rt_raw = torch.Tensor(self.latest_vars['rt_raw']).to(device)
            rt_raw[:,:3,3] = rt_raw[:,:3,3] / self.obj_scale
        else:
            rt_raw = self.create_base_se3(frameid.shape[0], device, self.camera_radius)
        
        # compute mlp rts
        if opts.root_opt:
            if self.root_basis == 'mlp' or self.root_basis == 'exp'\
            or self.root_basis == 'expmlp':
                root_rts = self.nerf_root_rts(frameid)
            else: print('error'); exit()
            rt_raw = self.refine_rt(rt_raw, root_rts)
             
        return rt_raw

    def save_latest_vars(self):
        """
        in: self.
        {rtk, kaug, frameid, vis2d}
        out: self.
        {latest_vars}
        these are only used in get_near_far_plane and compute_visibility
        """
        rtk = self.rtk.clone().detach()
        Kmat = K2mat(rtk[:,3])
        Kaug = K2inv(self.kaug) # p = Kaug Kmat P
        rtk[:,3] = mat2K(Kaug.matmul(Kmat))

        # TODO don't want to save k at eval time (due to different intrinsics)
        self.latest_vars['rtk'][self.frameid.long()] = rtk.cpu().numpy()
        self.latest_vars['rt_raw'][self.frameid.long()] = self.rt_raw.cpu().numpy()
        self.latest_vars['kaug'][self.frameid.long()] = self.kaug.cpu().numpy()
        self.latest_vars['idk'][self.frameid.long()] = 1

    def set_input(self, batch, load_line=False):
        device = self.device
        opts = self.opts

        
        self.convert_batch_input(batch)
        bs = self.imgs.shape[0]
        
        self.convert_root_pose()
      
        self.save_latest_vars()
        

        return bs

    def save_imgs(self,savedir,epoch,vid=5,use_deform=True,novel_cam=False,fixed_cam=False,
                  save_img=True,save_video=True,random_color=False,bone_color=False,fixed_frame=-1,
                  frame_bone=False,camera_axis=np.pi,fai=0.,ease_rot=False,postfix='',rp=False,seqname=''):
        if vid ==-1: vid=5
        opts = self.opts
        rtks = self.compute_rts()[self.data_offset[vid]:self.data_offset[vid+1],:,:].cuda()
        # kaugs = self.latest_vars['kaug'][self.data_offset[vid]:self.data_offset[vid+1],:]
        # rtk = torch.Tensor(rtk).to(self.device)
        sample_idx = np.linspace(0,len(rtks)-1, len(rtks)).astype(int)
        img_size = self.img_size
        bs = len(rtks)
        # near_far = self.near_far[self.data_offset[vid]:self.data_offset[vid+1],:]
        # embedid = torch.Tensor(sample_idx).to(self.device).long() + \
        #         self.data_offset[vid]
        rgbs = []
        masks = []
        others = []
        prefix = 'origin'
        if novel_cam:
            prefix = 'novel'
        if fixed_cam:
            prefix = 'fix'
        if random_color:
            prefix=prefix+'_random'
        if bone_color:
            prefix=prefix+'_bone'
        prefix = prefix+postfix
        #import pdb;pdb.set_trace()
        # target = [random.randint(0,bs-1) for i in range(5)]
        H=1024
        W=1024
        if random_color:
            self.color_replace = torch.rand_like(self.gaussians._xyz).cuda()
        else:
            self.color_replace = None
        
        
        frames = {'camel':(0,25),'bailang':(8,31),'zongxiong':(0,bs),'snail':(0,bs),'penguin_n':(0,bs),'littlelion':(5,bs)}
        target = list(range(bs))
        # print(target)
        rotate_num = 2
        final_angle = 2*np.pi
        if rp:
            a,b = frames[seqname]
            target = list(range(a,b))
            target = target*10
            target = target[:80]
            final_angle = final_angle*rotate_num
        for i in tqdm(range(len(target))):
            rtk = torch.cat([rtks[target[i]],self.ks_param[vid][None,...]],dim=0)
            cam_r = self.camera_radius
            gt_theta, gt_phi = angles_from_rtk(rtk)
            gt_theta = 0.
            gt_phi=0.
            rtk = rtk_from_angles(gt_theta, gt_phi,radius = cam_r)
            Rmat = rtk[:3,:3]
            Tmat = rtk[:3,3]
            K = rtk[3,:]
            
            if fixed_cam:
                angle = camera_axis
                
                Rmat = torch.tensor([[-np.sin(angle),0.,-np.cos(angle)],
                                        [-np.cos(angle)*np.sin(fai),-np.cos(fai),np.sin(angle)*np.sin(fai)],    
                                        [-np.cos(angle)*np.cos(fai),np.sin(fai),np.sin(angle)*np.cos(fai)]])
                Tmat = torch.tensor([0.,0.,1.])*cam_r
                # Rmat = torch.inverse(torch.tensor([[-np.sin(angle),0.,-np.cos(angle)],
                #                         [0.,-1.,0.],
                #                         [-np.cos(angle),0.,np.sin(angle)]]))
                # Tmat = torch.tensor([0.,0.,1.])*cam_r
                K = torch.tensor([512.,512.,256.,256.])
                H=W=512
            
            if novel_cam:
                angle = final_angle*i/len(target)
                if ease_rot:
                    angle /= 6.
                
                Rmat = torch.tensor([[-np.sin(angle),0.,-np.cos(angle)],
                                        [-np.cos(angle)*np.sin(fai),-np.cos(fai),np.sin(angle)*np.sin(fai)],    
                                        [-np.cos(angle)*np.cos(fai),np.sin(fai),np.sin(angle)*np.cos(fai)]])
                Tmat = torch.tensor([0.,0.,1.])*cam_r
                # Rmat = torch.inverse(torch.tensor([[-np.sin(angle),0.,-np.cos(angle)],
                #                         [0.,-1.,0.],
                #                         [-np.cos(angle),0.,np.sin(angle)]]))
                # Tmat = torch.tensor([0.,0.,1.])*cam_r
                K = torch.tensor([512.,512.,256.,256.])
                H=W=512
            
            rtk[:3,:3] = Rmat
            rtk[:3,3] = Tmat
            rtk[3,:] = K
            background = torch.tensor([1.,1.,1.]).cuda()
            id = torch.tensor([target[i]+self.data_offset[vid]],device=rtk.device)
            if fixed_frame != -1:
                id = torch.tensor([fixed_frame],device=rtk.device)
                use_deform = True
            if frame_bone:
                results = self.bone_render(rtk,(H,W),id,
                                       background=background,canonical=not use_deform,bone_color=bone_color)
            else:
                results = self.main_render(rtk,(H,W),id,
                                        background=background,canonical=not use_deform,bone_color=bone_color)
            rgb = results['render'].clamp(0,1)
            # rgb[rgb>1.]=1.
            rgb = rgb.moveaxis(0,-1).detach().cpu().numpy()
            rgbs.append(rgb)
            if random_color:
                mask = results['mask']
                # import pdb;pdb.set_trace()
                masks.append(mask[0][...,None][:,:,[0,0,0]].detach().cpu().numpy())
            # if bone_color:
            #     bone_vis = results['bone_vis'].clamp(0,1)
            #     others.append(bone_vis.moveaxis(0,-1).detach().cpu().numpy())
            # if save_img:
            #     cv2.imwrite('%s/%s_%03d_%05d.png'%(savedir,prefix,epoch,i), rgb[...,::-1]*255)
        if save_video:
            save_vid('%s/%s_%03d'%(savedir,prefix,epoch), rgbs, suffix='.mp4',upsample_frame=0)
            if random_color:
                save_vid('%s/%s_%03d'%(savedir,prefix[:-6]+'mask',epoch), masks, suffix='.mp4',upsample_frame=0)
            # if bone_color:
            #     save_vid('%s/%s_%03d'%(savedir,prefix[:-5]+'bone_vis',epoch), others, suffix='.mp4',upsample_frame=0)
        return rgbs