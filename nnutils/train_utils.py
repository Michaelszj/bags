# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch
import torch.nn as nn
import os
import os.path as osp
import sys
sys.path.insert(0,'third_party')
import time
import pdb
import numpy as np
from absl import flags
import cv2
import time

import mcubes
from nnutils import banmo
import subprocess
from torch.utils.tensorboard import SummaryWriter
from kmeans_pytorch import kmeans
import torch.distributed as dist
import torch.nn.functional as F
import trimesh
import torchvision
from torch.autograd import Variable
from collections import defaultdict
from pytorch3d import transforms
from torch.nn.utils import clip_grad_norm_
from matplotlib.pyplot import cm
from tqdm import tqdm
from nnutils.geom_utils import lbs, reinit_bones, warp_bw, warp_fw, vec_to_sim3,\
                               obj_to_cam, get_near_far, near_far_to_bound, \
                               compute_point_visibility, process_so3_seq, \
                               ood_check_cse, align_sfm_sim3, gauss_mlp_skinning, \
                               correct_bones
from nnutils.nerf import grab_xyz_weights
# from ext_utils.flowlib import flow_to_image
from utils.io import mkdir_p
from nnutils.vis_utils import image_grid
from utils.general_utils import inverse_sigmoid
from dataloader import frameloader
from utils.io import save_vid, draw_cams, extract_data_info, merge_dict,\
        render_root_txt, save_bones, draw_cams_pair, get_vertex_colors
from utils.colors import label_colormap
from torch.utils.tensorboard import SummaryWriter
class DataParallelPassthrough(torch.nn.parallel.DistributedDataParallel):
    """
    for multi-gpu access
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
    
    def __delattr__(self, name):
        try:
            return super().__delattr__(name)
        except AttributeError:
            return delattr(self.module, name)
    
class v2s_trainer():
    def __init__(self, opts, is_eval=False):
        self.opts = opts
        self.is_eval=is_eval
        self.local_rank = opts.local_rank
        self.save_dir = os.path.join(opts.checkpoint_dir, opts.logname)
        
        self.accu_steps = 5 # opts.accu_steps
        name = "GSMo"
        tb_writer_summary_path = os.path.join(opts.checkpoint_dir, "runs")
        current_time = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        log_dir = os.path.join(tb_writer_summary_path, current_time)
        self.writer = SummaryWriter(log_dir=log_dir, comment=name)
        # write logs
        if opts.local_rank==0:
            if not os.path.exists(self.save_dir): os.makedirs(self.save_dir)
            log_file = os.path.join(self.save_dir, 'opts.log')
            if not self.is_eval:
                if os.path.exists(log_file):
                    os.remove(log_file)
                opts.append_flags_into_file(log_file)

    def define_model(self, data_info, extra_opt):
        opts = self.opts
        self.device = torch.device('cuda:{}'.format(opts.local_rank))
        self.model = banmo.banmo(opts, data_info, extra_opt)
        self.model.save_dir = self.save_dir
        self.model.forward = self.model.forward_default
        self.num_epochs = opts.num_epochs

        # load model
        if opts.model_path!='':
            self.load_network(opts.model_path, is_eval=self.is_eval)

        self.model = self.model.to(self.device)
        return
    
    def init_dataset(self):
        opts = self.opts
        opts_dict = {}
        opts_dict['n_data_workers'] = opts.n_data_workers
        opts_dict['batch_size'] = opts.batch_size
        opts_dict['seqname'] = opts.seqname
        opts_dict['img_size'] = opts.img_size
        opts_dict['ngpu'] = opts.ngpu
        opts_dict['local_rank'] = opts.local_rank
        opts_dict['rtk_path'] = opts.rtk_path
        opts_dict['preload']= False
        opts_dict['accu_steps'] = opts.accu_steps

        if self.is_eval and opts.rtk_path=='' and opts.model_path!='':
            # automatically load cameras in the logdir
            model_dir = opts.model_path.rsplit('/',1)[0]
            cam_dir = '%s/init-cam/'%model_dir
            if os.path.isdir(cam_dir):
                opts_dict['rtk_path'] = cam_dir

        self.dataloader = frameloader.data_loader(opts_dict,shuffle=False)
        # opts_dict['multiply'] = True
        self.trainloader = frameloader.data_loader(opts_dict)
        # del opts_dict['multiply']
        # opts_dict['img_size'] = opts.render_size
        self.evalloader = frameloader.eval_loader(opts_dict)

        # compute data offset
        data_info = extract_data_info(self.evalloader)
        return data_info
    
    def init_training(self):
        opts = self.opts
        # set as module attributes since they do not change across gpus
        self.model.final_steps = self.num_epochs * \
                                min(200,len(self.trainloader)) * opts.accu_steps
        # ideally should be greater than 200 batches

        # params_nerf_coarse=[]
        # params_nerf_beta=[]
        # params_nerf_feat=[]
        # params_nerf_beta_feat=[]
        # params_nerf_fine=[]
        # params_nerf_unc=[]
        # params_nerf_flowbw=[]
        params_nerf_skin=[]
        params_delta_net=[]
        # params_nerf_vis=[]
        params_nerf_root_rts=[]
        params_nerf_body_rts=[]
        params_root_code=[]
        params_pose_code=[]
        params_env_code=[]
        params_vid_code=[]
        params_bones=[]
        params_skin_aux=[]
        params_ks=[]
        # params_nerf_dp=[]
        # params_csenet=[]
        for name,p in self.model.named_parameters():
            if 'nerf_skin' in name:
                params_nerf_skin.append(p)
            elif 'delta_net' in name:
                params_delta_net.append(p)
            elif 'nerf_root_rts' in name:
                params_nerf_root_rts.append(p)
            elif 'nerf_body_rts' in name:
                params_nerf_body_rts.append(p)
            elif 'root_code' in name:
                params_root_code.append(p)
            elif 'pose_code' in name or 'rest_pose_code' in name:
                params_pose_code.append(p)
            elif 'env_code' in name:
                params_env_code.append(p)
            elif 'vid_code' in name:
                params_vid_code.append(p)
            elif 'bones' == name:
                params_bones.append(p)
            elif 'skin_aux' == name:
                params_skin_aux.append(p)
            elif 'ks_param' == name:
                params_ks.append(p)
            else: continue
            if opts.local_rank==0:
                print('optimized params: %s'%name)

        self.optimizer = torch.optim.AdamW(
            [# {'params': params_nerf_coarse},
            #  {'params': params_nerf_beta},
            #  {'params': params_nerf_feat},
            #  {'params': params_nerf_beta_feat},
            #  {'params': params_nerf_fine},
            #  {'params': params_nerf_unc},
            #  {'params': params_nerf_flowbw},
             {'params': params_nerf_skin},
            #  {'params': params_nerf_vis},
             {'params': params_delta_net},
             {'params': params_nerf_root_rts},
             {'params': params_nerf_body_rts,'lr':opts.learning_rate*0.1},
             {'params': params_root_code},
             {'params': params_pose_code},
             {'params': params_env_code},
             {'params': params_vid_code},
             {'params': params_bones},
             {'params': params_skin_aux,'lr':opts.learning_rate},
             {'params': params_ks},
            #  {'params': params_nerf_dp},
            #  {'params': params_csenet},
            ],
            lr=opts.learning_rate,betas=(0.9, 0.999),weight_decay=1e-4)

        if self.model.root_basis=='exp':
            lr_nerf_root_rts = 10
        elif self.model.root_basis=='cnn':
            lr_nerf_root_rts = 0.2
        elif self.model.root_basis=='mlp':
            lr_nerf_root_rts = 1 
        elif self.model.root_basis=='expmlp':
            lr_nerf_root_rts = 1 
        else: print('error'); exit()
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(self.optimizer,\
                        [# opts.learning_rate, # params_nerf_coarse
                    #      opts.learning_rate, # params_nerf_beta
                    #      opts.learning_rate, # params_nerf_feat
                    #   10*opts.learning_rate, # params_nerf_beta_feat
                    #      opts.learning_rate, # params_nerf_fine
                    #      opts.learning_rate, # params_nerf_unc
                    #      opts.learning_rate, # params_nerf_flowbw
                         opts.learning_rate, # params_nerf_skin
                         10*opts.learning_rate, # params_delta_net
                        #  opts.learning_rate, # params_nerf_vis
        lr_nerf_root_rts*opts.learning_rate, # params_nerf_root_rts
                         10*opts.learning_rate, # params_nerf_body_rts
        lr_nerf_root_rts*opts.learning_rate, # params_root_code
                         opts.learning_rate, # params_pose_code
                         opts.learning_rate, # params_env_code
                         opts.learning_rate, # params_vid_code
                         opts.learning_rate, # params_bones
                      10*opts.learning_rate, # params_skin_aux
                      10*opts.learning_rate, # params_ks
                        #  opts.learning_rate, # params_nerf_dp
                        #  opts.learning_rate, # params_csenet
            ],
            int(self.model.final_steps/self.accu_steps),
            pct_start=2./self.num_epochs, # use 2 epochs to warm up
            cycle_momentum=False, 
            anneal_strategy='linear',
            final_div_factor=1./5, div_factor = 25,
            )
    
    def save_network(self, epoch_label, prefix=''):
        if self.opts.local_rank==0:
            param_path = '%s/%sparams_%s.pth'%(self.save_dir,prefix,epoch_label)
            save_dict = self.model.state_dict()
            torch.save(save_dict, param_path)

            var_path = '%s/%svars_%s.npy'%(self.save_dir,prefix,epoch_label)
            latest_vars = self.model.latest_vars.copy()
            del latest_vars['fp_err']  
            del latest_vars['flo_err']   
            del latest_vars['sil_err'] 
            del latest_vars['flo_err_hist']
            np.save(var_path, latest_vars)
            return
    
    @staticmethod
    def rm_module_prefix(states, prefix='module'):
        new_dict = {}
        for i in states.keys():
            v = states[i]
            if i[:len(prefix)] == prefix:
                i = i[len(prefix)+1:]
            new_dict[i] = v
        return new_dict

    def load_network(self,model_path=None, is_eval=True, rm_prefix=True):
        opts = self.opts
        states = torch.load(model_path,map_location='cpu')
        if rm_prefix: states = self.rm_module_prefix(states)
        var_path = model_path.replace('params', 'vars').replace('.pth', '.npy')
        latest_vars = np.load(var_path,allow_pickle=True)[()]
        
        if is_eval:
            # load variables
            self.model.latest_vars = latest_vars
        
        # if size mismatch, delete all related variables
        if rm_prefix and states['near_far'].shape[0] != self.model.near_far.shape[0]:
            print('!!!deleting video specific dicts due to size mismatch!!!')
            self.del_key( states, 'near_far') 
            self.del_key( states, 'root_code.weight') # only applies to root_basis=mlp
            self.del_key( states, 'pose_code.weight')
            self.del_key( states, 'pose_code.basis_mlp.weight')
            self.del_key( states, 'nerf_body_rts.0.weight')
            self.del_key( states, 'nerf_body_rts.0.basis_mlp.weight')
            self.del_key( states, 'nerf_root_rts.0.weight')
            self.del_key( states, 'nerf_root_rts.root_code.weight')
            self.del_key( states, 'nerf_root_rts.root_code.basis_mlp.weight')
            self.del_key( states, 'nerf_root_rts.delta_rt.0.basis_mlp.weight')
            self.del_key( states, 'nerf_root_rts.base_rt.se3')
            self.del_key( states, 'nerf_root_rts.delta_rt.0.weight')
            self.del_key( states, 'env_code.weight')
            self.del_key( states, 'env_code.basis_mlp.weight')
            if 'vid_code.weight' in states.keys():
                self.del_key( states, 'vid_code.weight')
            if 'ks_param' in states.keys():
                self.del_key( states, 'ks_param')

            # delete pose basis(backbones)
            if not opts.keep_pose_basis:
                del_key_list = []
                for k in states.keys():
                    if 'nerf_body_rts' in k or 'nerf_root_rts' in k:
                        del_key_list.append(k)
                for k in del_key_list:
                    print(k)
                    self.del_key( states, k)
    
        if rm_prefix and opts.lbs and states['bones'].shape[0] != self.model.bones.shape[0]:
            self.del_key(states, 'bones')
            states = self.rm_module_prefix(states, prefix='nerf_skin')
            states = self.rm_module_prefix(states, prefix='nerf_body_rts')


        # load some variables
        # this is important for volume matching
        if latest_vars['obj_bound'].size==1:
            latest_vars['obj_bound'] = latest_vars['obj_bound'] * np.ones(3)
        self.model.latest_vars['obj_bound'] = latest_vars['obj_bound'] 

        # load nerf_coarse, nerf_bone/root (not code), nerf_vis, nerf_feat, nerf_unc
        #TODO somehow, this will reset the batch stats for 
        # a pretrained cse model, to keep those, we want to manually copy to states
        if opts.ft_cse and \
          'csenet.net.backbone.fpn_lateral2.weight' not in states.keys():
            self.add_cse_to_states(self.model, states)
        self.model.load_state_dict(states, strict=False)

        return

    @staticmethod 
    def add_cse_to_states(model, states):
        states_init = model.state_dict()
        for k in states_init.keys():
            v = states_init[k]
            if 'csenet' in k:
                states[k] = v

    def eval_cam(self, idx_render=None): 
        """
        idx_render: list of frame index to render
        """
        opts = self.opts
        
        with torch.no_grad():
            self.model.eval()
            # load data
            for dataset in self.evalloader.dataset.datasets:
                dataset.load_pair = False
                dataset.feat_only = True
            batch = []
            for i in idx_render:
                batch.append( self.evalloader.dataset[i] )
            # import pdb;pdb.set_trace()
            batch = self.evalloader.collate_fn(batch)
            for dataset in self.evalloader.dataset.datasets:
                dataset.load_pair = True
                dataset.feat_only = False

            #TODO can be further accelerated
            self.model.convert_feat_input(batch)

            if opts.unc_filter:
                # process densepoe feature
                valid_list, error_list = ood_check_cse(self.model.dp_feats, 
                                        self.model.dp_embed, 
                                        self.model.dps.long())
                valid_list = valid_list.cpu().numpy()
                error_list = error_list.cpu().numpy()
            else:
                valid_list = np.ones( len(idx_render))
                error_list = np.zeros(len(idx_render))

            self.model.convert_root_pose()
            rtk = self.model.rtk
            kaug = self.model.kaug

            #TODO may need to recompute after removing the invalid predictions
            # need to keep this to compute near-far planes
            # self.model.save_latest_vars()
                
            # extract mesh sequences
            aux_seq = {
                       'is_valid':[],
                       'err_valid':[],
                       'rtk':[],
                       'kaug':[],
                       'impath':[],
                       'masks':[],
                       }
            if opts.local_rank==0: 
                    print(f'extracting frame {idx_render[0]} to {idx_render[-1]}')
            for idx,_ in enumerate(idx_render):
                frameid=self.model.frameid[idx]
                aux_seq['rtk'].append(rtk[idx].cpu().numpy())
                aux_seq['kaug'].append(kaug[idx].cpu().numpy())
                # aux_seq['masks'].append(self.model.masks[idx].cpu().numpy())
                aux_seq['is_valid'].append(valid_list[idx])
                aux_seq['err_valid'].append(error_list[idx])
                
                impath = self.model.impath[frameid.long()]
                aux_seq['impath'].append(impath)
        return aux_seq
  
    def eval(self, idx_render=None, dynamic_mesh=False): 
        """
        idx_render: list of frame index to render
        dynamic_mesh: whether to extract canonical shape, or dynamic shape
        """
        opts = self.opts
        with torch.no_grad():
            self.model.eval()

            # run marching cubes on canonical shape
            mesh_dict_rest = self.extract_mesh(self.model, opts.chunk, \
                                         opts.sample_grid3d, opts.mc_threshold)

            # choose a grid image or the whold video
            if idx_render is None: # render 9 frames
                idx_render = np.linspace(0,len(self.evalloader)-1, 9, dtype=int)

            # render
            chunk=opts.rnd_frame_chunk
            rendered_seq = defaultdict(list)
            aux_seq = {'mesh_rest': mesh_dict_rest['mesh'],
                       'mesh':[],
                       'rtk':[],
                       'impath':[],
                       'bone':[],}
            for j in range(0, len(idx_render), chunk):
                batch = []
                idx_chunk = idx_render[j:j+chunk]
                for i in idx_chunk:
                    batch.append( self.evalloader.dataset[i] )
                batch = self.evalloader.collate_fn(batch)
                rendered = self.render_vid(self.model, batch) 
            
                for k, v in rendered.items():
                    rendered_seq[k] += [v]
                    
                hbs=len(idx_chunk)
                sil_rszd = F.interpolate(self.model.masks[:hbs,None], 
                            (opts.render_size, opts.render_size))[:,0,...,None]
                rendered_seq['img'] += [self.model.imgs.permute(0,2,3,1)[:hbs]]
                rendered_seq['sil'] += [self.model.masks[...,None]      [:hbs]]
                rendered_seq['flo'] += [self.model.flow.permute(0,2,3,1)[:hbs]]
                rendered_seq['dpc'] += [self.model.dp_vis[self.model.dps.long()][:hbs]]
                rendered_seq['occ'] += [self.model.occ[...,None]      [:hbs]]
                rendered_seq['feat']+= [self.model.dp_feats.std(1)[...,None][:hbs]]
                rendered_seq['flo_coarse'][-1]       *= sil_rszd 
                rendered_seq['img_loss_samp'][-1]    *= sil_rszd 
                if 'frame_cyc_dis' in rendered_seq.keys() and \
                    len(rendered_seq['frame_cyc_dis'])>0:
                    rendered_seq['frame_cyc_dis'][-1] *= 255/rendered_seq['frame_cyc_dis'][-1].max()
                    rendered_seq['frame_rigloss'][-1] *= 255/rendered_seq['frame_rigloss'][-1].max()
                if opts.use_embed:
                    rendered_seq['pts_pred'][-1] *= sil_rszd 
                    rendered_seq['pts_exp'] [-1] *= rendered_seq['sil_coarse'][-1]
                    rendered_seq['feat_err'][-1] *= sil_rszd
                    rendered_seq['feat_err'][-1] *= 255/rendered_seq['feat_err'][-1].max()
                if opts.use_proj:
                    rendered_seq['proj_err'][-1] *= sil_rszd
                    rendered_seq['proj_err'][-1] *= 255/rendered_seq['proj_err'][-1].max()
                if opts.use_unc:
                    rendered_seq['unc_pred'][-1] -= rendered_seq['unc_pred'][-1].min()
                    rendered_seq['unc_pred'][-1] *= 255/rendered_seq['unc_pred'][-1].max()

                # extract mesh sequences
                for idx in range(len(idx_chunk)):
                    frameid=self.model.frameid[idx].long()
                    embedid=self.model.embedid[idx].long()
                    print('extracting frame %d'%(frameid.cpu().numpy()))
                    # run marching cubes
                    if dynamic_mesh:
                        if not opts.queryfw:
                           mesh_dict_rest=None 
                        mesh_dict = self.extract_mesh(self.model,opts.chunk,
                                            opts.sample_grid3d, opts.mc_threshold,
                                        embedid=embedid, mesh_dict_in=mesh_dict_rest)
                        mesh=mesh_dict['mesh']
                        if mesh_dict_rest is not None and opts.ce_color:
                            mesh.visual.vertex_colors = mesh_dict_rest['mesh'].\
                                   visual.vertex_colors # assign rest surface color
                        else:
                            # get view direction 
                            obj_center = self.model.rtk[idx][:3,3:4]
                            cam_center = -self.model.rtk[idx][:3,:3].T.matmul(obj_center)[:,0]
                            view_dir = torch.cuda.FloatTensor(mesh.vertices, device=self.device) \
                                            - cam_center[None]
                            vis = get_vertex_colors(self.model, mesh_dict_rest['mesh'], 
                                                    frame_idx=idx, view_dir=view_dir)
                            mesh.visual.vertex_colors[:,:3] = vis*255

                        # save bones
                        if 'bones' in mesh_dict.keys():
                            bone = mesh_dict['bones'][0].cpu().numpy()
                            aux_seq['bone'].append(bone)
                    else:
                        mesh=mesh_dict_rest['mesh']
                    aux_seq['mesh'].append(mesh)

                    # save cams
                    aux_seq['rtk'].append(self.model.rtk[idx].cpu().numpy())
                    
                    # save image list
                    impath = self.model.impath[frameid]
                    aux_seq['impath'].append(impath)

            # save canonical mesh and extract skinning weights
            mesh_rest = aux_seq['mesh_rest']
            if len(mesh_rest.vertices)>100:
                self.model.latest_vars['mesh_rest'] = mesh_rest
            if opts.lbs:
                bones_rst = self.model.bones
                bones_rst,_ = correct_bones(self.model, bones_rst)
                # compute skinning color
                if mesh_rest.vertices.shape[0]>100:
                    rest_verts = torch.Tensor(mesh_rest.vertices).to(self.device)
                    nerf_skin = self.model.nerf_skin if opts.nerf_skin else None
                    rest_pose_code = self.model.rest_pose_code(torch.Tensor([0])\
                                            .long().to(self.device))
                    skins = gauss_mlp_skinning(rest_verts[None], 
                            self.model.embedding_xyz,
                            bones_rst, rest_pose_code, 
                            nerf_skin, skin_aux=self.model.skin_aux)[0]
                    skins = skins.cpu().numpy()
   
                    num_bones = skins.shape[-1]
                    colormap = label_colormap()
                    # TODO use a larger color map
                    colormap = np.repeat(colormap[None],4,axis=0).reshape(-1,3)
                    colormap = colormap[:num_bones]
                    colormap = (colormap[None] * skins[...,None]).sum(1)

                    mesh_rest_skin = mesh_rest.copy()
                    mesh_rest_skin.visual.vertex_colors = colormap
                    aux_seq['mesh_rest_skin'] = mesh_rest_skin

                aux_seq['bone_rest'] = bones_rst.cpu().numpy()
        
            # draw camera trajectory
            suffix_id=0
            if hasattr(self.model, 'epoch'):
                suffix_id = self.model.epoch
            if opts.local_rank==0:
                mesh_cam = draw_cams(aux_seq['rtk'])
                mesh_cam.export('%s/mesh_cam-%02d.obj'%(self.save_dir,suffix_id))
            
                mesh_path = '%s/mesh_rest-%02d.obj'%(self.save_dir,suffix_id)
                mesh_rest.export(mesh_path)

                if opts.lbs:
                    bone_rest = aux_seq['bone_rest']
                    bone_path = '%s/bone_rest-%02d.obj'%(self.save_dir,suffix_id)
                    save_bones(bone_rest, 0.1, bone_path)

            # save images
            for k,v in rendered_seq.items():
                rendered_seq[k] = torch.cat(rendered_seq[k],0)
                ##TODO
                #if opts.local_rank==0:
                #    print('saving %s to gif'%k)
                #    is_flow = self.isflow(k)
                #    upsample_frame = min(30,len(rendered_seq[k]))
                #    save_vid('%s/%s'%(self.save_dir,k), 
                #            rendered_seq[k].cpu().numpy(), 
                #            suffix='.gif', upsample_frame=upsample_frame, 
                #            is_flow=is_flow)

        return rendered_seq, aux_seq


    def eval(self,epoch,bone=False):
        target_vid = 0
        temp_dir = self.save_dir+"/eval"
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        rgbs = self.model.save_imgs(temp_dir,epoch,target_vid,save_img=False,frame_bone=bone)
        self.model.save_imgs(temp_dir,epoch,target_vid,save_img=False,random_color=True,frame_bone=bone)
        bones = self.model.save_imgs(temp_dir,epoch,target_vid,save_img=False,bone_color=True,frame_bone=bone)
        with torch.no_grad():
            self.model.save_imgs(temp_dir,epoch,target_vid,use_deform=False,novel_cam=True,save_img=False,frame_bone=bone)
            self.model.save_imgs(temp_dir,epoch,target_vid,use_deform=False,novel_cam=True,save_img=False,bone_color=True,frame_bone=bone)
        self.model.save_imgs(temp_dir,epoch,target_vid,fixed_cam=True,save_img=False,frame_bone=bone)
        random_frame = torch.randint(max(0,self.center_frame-5),min(len(self.evalloader),self.center_frame+5),size=(1,)).item()
        self.model.save_imgs(temp_dir,epoch,target_vid,novel_cam=True,save_video=False,frame_bone=bone,fixed_frame=random_frame)
        
        # self.save_cameras(self.model.rtk_all,epoch,target_vid)
        # self.cat_videos(self.gts,rgbs,bones,temp_dir+'/cat_%03d'%(epoch))
        
        
        
    def setup(self):
        seqname=self.opts.seqname
        self.center_frame = 21
        temp_dir = self.save_dir+"/checkpoints"
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        # self.model.warmup_canonical(self.evalloader.dataset.datasets[0][self.center_frame],temp_dir)
        # self.model.gaussians.save_ply(temp_dir+'/epoch_init.ply')
        self.model.gaussians.load_ply(temp_dir+'/epoch_init.ply')
        self.model.gaussians.training_setup(self.model.optim)
        
        self.gts = []
        img_path = f'database/DAVIS/JPEGImages/Full-Resolution/{seqname}/'
        mask_path = f'database/DAVIS/Annotations/Full-Resolution/{seqname}/'
        # import pdb;pdb.set_trace()
        for i in range(0, len(self.dataloader)+1):
            j = i+1
            img = cv2.imread(img_path+str(j).zfill(5)+'.jpg')[:,:,::-1]/255.
            try:
                mask = cv2.imread(mask_path+str(j).zfill(5)+'.png')[:,:,:1]/255.
            except:
                mask = cv2.imread(mask_path+str(j).zfill(5)+'.jpg')[:,:,:1]/255.
            self.gts.append(img*mask+(1.-mask))
            
    def train(self):
        opts = self.opts
        log=None
        self.model.total_steps = 0
        self.model.progress = 0
        torch.manual_seed(8)  # do it again
        torch.cuda.manual_seed(1)
        self.model.writer=self.writer
        # disable bones before warmup epochs are finished
    
        # CNN pose warmup or  load CNN
        # import pdb;pdb.set_trace()
        # if True:
        #     self.preload_pose()
        # elif opts.warmup_pose_ep>0 or opts.pose_cnn_path!='':
        #     self.warmup_pose(log, pose_cnn_path=opts.pose_cnn_path)
        
        
        
        seqname=opts.seqname
        self.center_frame = 21
        temp_dir = self.save_dir+"/checkpoints"
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        self.model.warmup_canonical(self.evalloader.dataset.datasets[0][self.center_frame],temp_dir)
        self.model.gaussians.save_ply(temp_dir+'/epoch_init.ply')
        # self.model.gaussians.load_ply(temp_dir+'/epoch_init.ply')
        # self.model.gaussians.training_setup(self.model.optim)
        # self.model.load_bones(self.save_dir)
        # import pdb;pdb.set_trace()
        self.gts = []
        img_path = f'datasource/{seqname}/imgs/'
        mask_path = f'datasource/{seqname}/masks/'
        # import pdb;pdb.set_trace()
        offset = 0
        try:
            _ = cv2.imread(img_path+str(0).zfill(5)+'.jpg')[:,:,::-1]/255.
        except:
            offset = 1
        print('offset:',offset)
        print('length:',len(self.dataloader)+1)
        for i in range(0, len(self.dataloader)+1):
            j = i+offset
            try:
                img = cv2.imread(img_path+str(j).zfill(5)+'.jpg')[:,:,::-1]/255.
            except:
                print('Error reading image:',img_path+str(j).zfill(5)+'.jpg')
            try:
                mask = cv2.imread(mask_path+str(j).zfill(5)+'.png')[:,:,:1]/255.
            except:
                mask = cv2.imread(mask_path+str(j).zfill(5)+'.jpg')[:,:,:1]/255.
            self.gts.append(img*mask+(1.-mask))
        
        # start training
        # import pdb;pdb.set_trace()
        # torch.cuda.empty_cache()
        self.model.use_diffusion = False
        # for epoch in range(0, self.num_epochs):
        #     self.model.epoch = epoch
    
        #     self.model.img_size = opts.img_size
        #     self.reset_hparams(epoch)
        #     self.train_one_epoch(epoch, self.center_frame)
        #     temp_dir = self.save_dir+"/imgs"
        #     if not os.path.isdir(temp_dir):
        #         os.makedirs(temp_dir)
            
        #     if epoch % 30 == 0:
        #         self.eval(epoch)
        #     temp_dir = self.save_dir+"/checkpoints"
        #     if not os.path.isdir(temp_dir):
        #         os.makedirs(temp_dir)
        #     if epoch % 200 == 1:
        #         self.model.gaussians.save_ply(temp_dir+'/epoch_'+str(epoch)+'.ply')
        # import pdb;pdb.set_trace()
        # self.model.save_bones()
        #import pdb;pdb.set_trace()
        
        self.reset_hparams(0)
        t = self.center_frame-1
        while(t>=0):
            self.train_bones(100,t,dup=True)
            t -= 1
        t = self.center_frame+1
        self.model.save_bones()
        self.eval(10000, bone=True)
        while(t<self.model.num_fr):
            self.train_bones(100,t,dup=True)
            t += 1
        self.model.save_bones()
        self.eval(20000, bone=True)
        
        # self.eval(10000, bone=True)
        # self.model.use_diffusion = True
        # self.eval(1000)
        # for epoch in range(0, self.num_epochs):
        #     self.model.epoch = epoch
    
        #     self.model.img_size = opts.img_size
        #     self.train_one_epoch(epoch, self.center_frame)
        #     temp_dir = self.save_dir+"/imgs"
        #     if not os.path.isdir(temp_dir):
        #         os.makedirs(temp_dir)
            
        #     if epoch % 30 == 0:
        #         self.eval(epoch)
        #     temp_dir = self.save_dir+"/checkpoints"
        #     if not os.path.isdir(temp_dir):
        #         os.makedirs(temp_dir)
        #     if epoch % 200 == 1:
        #         self.model.gaussians.save_ply(temp_dir+'/epoch_'+str(epoch)+'.ply')
    
    def cat_videos(self,gt,rgb,bone,path):
        cat = []
        for i in range(len(gt)):
            H = gt[i].shape[-3]
            W = gt[i].shape[-2]
            print("H=",H,"W=",W)
            if W>H:
                cat.append(np.concatenate([gt[i],rgb[i][420:-420],bone[i][420:-420]],axis=1))
            elif W<H:
                cat.append(np.concatenate([gt[i],rgb[i][:,420:-420],bone[i][:,420:-420]],axis=1))
            else:
                cat.append(np.concatenate([gt[i],rgb[i],bone[i]],axis=1))
        save_vid(path, cat, suffix='.mp4',upsample_frame=0)
        
    def save_cameras(self, rtk_all, epoch, vid=5):
        temp_dir = self.save_dir+"/cam"
        if not os.path.isdir(temp_dir):
            os.makedirs(temp_dir)
        rtks = rtk_all[self.model.data_offset[vid]:self.model.data_offset[vid+1]]
        rtks = torch.cat([rtks,self.model.ks_param[vid][None,None,...].repeat(rtks.shape[0],1,1)],dim=1).detach().cpu()
        mesh_cam = draw_cams(rtks)
        mesh_cam.export('%s/mesh_cam-%03d.obj'%(temp_dir,epoch))
        
    @staticmethod
    def save_cams(opts,aux_seq, save_prefix, latest_vars,datasets, evalsets, obj_scale,
            trainloader=None, unc_filter=True):
        """
        save cameras to dir and modify dataset 
        """
        mkdir_p(save_prefix)
        dataset_dict={dataset.imglist[0].split('/')[-2]:dataset for dataset in datasets}
        evalset_dict={dataset.imglist[0].split('/')[-2]:dataset for dataset in evalsets}
        if trainloader is not None:
            line_dict={dataset.imglist[0].split('/')[-2]:dataset for dataset in trainloader}

        length = len(aux_seq['impath'])
        valid_ids = aux_seq['is_valid']
        idx_combine = 0
        for i in range(length):
            impath = aux_seq['impath'][i]
            seqname = impath.split('/')[-2]
            rtk = aux_seq['rtk'][i]
           
            if unc_filter:
                # in the same sequance find the closest valid frame and replace it
                seq_idx = np.asarray([seqname == i.split('/')[-2] \
                        for i in aux_seq['impath']])
                valid_ids_seq = np.where(valid_ids * seq_idx)[0]
                if opts.local_rank==0 and i==0: 
                    print('%s: %d frames are valid'%(seqname, len(valid_ids_seq)))
                if len(valid_ids_seq)>0 and not aux_seq['is_valid'][i]:
                    closest_valid_idx = valid_ids_seq[np.abs(i-valid_ids_seq).argmin()]
                    rtk[:3,:3] = aux_seq['rtk'][closest_valid_idx][:3,:3]

            # rescale translation according to input near-far plane
            rtk[:3,3] = rtk[:3,3]*obj_scale
            rtklist = dataset_dict[seqname].rtklist
            idx = int(impath.split('/')[-1].split('.')[-2])
            save_path = '%s/%s-%05d.txt'%(save_prefix, seqname, idx)
            # import pdb;pdb.set_trace()
            np.savetxt(save_path, rtk)
            rtklist[idx] = save_path
            evalset_dict[seqname].rtklist[idx] = save_path
            if trainloader is not None:
                line_dict[seqname].rtklist[idx] = save_path
            
            #save to rtraw 
            # latest_vars['rt_raw'][idx_combine] = rtk[:3,:4]
            latest_vars['rtk'][idx_combine,:3,:3] = rtk[:3,:3]

            if idx==len(rtklist)-2:
                # to cover the last
                save_path = '%s/%s-%05d.txt'%(save_prefix, seqname, idx+1)
                if opts.local_rank==0: print('writing cam %s'%save_path)
                np.savetxt(save_path, rtk)
                rtklist[idx+1] = save_path
                evalset_dict[seqname].rtklist[idx+1] = save_path
                if trainloader is not None:
                    line_dict[seqname].rtklist[idx+1] = save_path

                idx_combine += 1
                # latest_vars['rt_raw'][idx_combine] = rtk[:3,:4]
                latest_vars['rtk'][idx_combine,:3,:3] = rtk[:3,:3]
            idx_combine += 1
        
        
    def extract_cams(self, full_loader):
        # store cameras
        opts = self.opts
        idx_render = range(len(self.evalloader))
        chunk = 50
        aux_seq = []
        for i in range(0, len(idx_render), chunk):
            aux_seq.append(self.eval_cam(idx_render=idx_render[i:i+chunk]))
        aux_seq = merge_dict(aux_seq)
        aux_seq['rtk'] = np.asarray(aux_seq['rtk'])
        aux_seq['kaug'] = np.asarray(aux_seq['kaug'])
        aux_seq['masks'] = np.asarray(aux_seq['masks'])
        aux_seq['is_valid'] = np.asarray(aux_seq['is_valid'])
        aux_seq['err_valid'] = np.asarray(aux_seq['err_valid'])

        save_prefix = '%s/init-cam'%(self.save_dir)
        trainloader=self.trainloader.dataset.datasets
        self.save_cams(opts,aux_seq, save_prefix,
                    self.model.latest_vars,
                    full_loader.dataset.datasets,
                self.evalloader.dataset.datasets,
                self.model.obj_scale, trainloader=trainloader,
                unc_filter=opts.unc_filter)
        
        # dist.barrier() # wait untail all have finished
        if opts.local_rank==0:
            # draw camera trajectory
            for dataset in full_loader.dataset.datasets:
                seqname = dataset.imglist[0].split('/')[-2]
                render_root_txt('%s/%s-'%(save_prefix,seqname), 0)


    def reset_nf(self):
        opts = self.opts
        # save near-far plane
        shape_verts = self.model.dp_verts_unit / 3 * self.model.near_far.mean()
        shape_verts = shape_verts * 1.2
        # save object bound if first stage
        if opts.model_path=='' and opts.bound_factor>0:
            shape_verts = shape_verts*opts.bound_factor
            self.model.latest_vars['obj_bound'] = \
            shape_verts.abs().max(0)[0].detach().cpu().numpy()

        if self.model.near_far[:,0].sum()==0: # if no valid nf plane loaded
            self.model.near_far.data = get_near_far(self.model.near_far.data,
                                                self.model.latest_vars,
                                         pts=shape_verts.detach().cpu().numpy())
        save_path = '%s/init-nf.txt'%(self.save_dir)
        save_nf = self.model.near_far.data.cpu().numpy() * self.model.obj_scale
        np.savetxt(save_path, save_nf)
    
    def warmup_shape(self, log):
        opts = self.opts

        # force using warmup forward, dataloader, cnn root
        self.model.forward = self.model.forward_warmup_shape
        full_loader = self.trainloader  # store original loader
        self.trainloader = range(200)
        self.num_epochs = opts.warmup_shape_ep

        # training
        self.init_training()
        for epoch in range(0, opts.warmup_shape_ep):
            self.model.epoch = epoch
            self.train_one_epoch(epoch, log, warmup=True)
            self.save_network(str(epoch+1), 'mlp-') 

        # restore dataloader, rts, forward function
        self.model.forward = self.model.forward_default
        self.trainloader = full_loader
        self.num_epochs = opts.num_epochs

        # start from low learning rate again
        self.init_training()
        self.model.total_steps = 0
        self.model.progress = 0.

    def preload_pose(self):
        print('Preloading estimated cameras...')
        load_prefix = '%s/init-cam'%(self.save_dir)
        for i in tqdm(range(self.model.num_fr)):
            impath = self.model.impath[i]
            idx = int(impath.split('/')[-1].split('.')[-2])
            seqname = impath.split('/')[-2]
            load_path = '%s/%s-%05d.txt'%(load_prefix, seqname, idx)
            self.model.latest_vars['rtk'][i] = (np.loadtxt(load_path))
            
    def warmup_pose(self, log, pose_cnn_path):
        opts = self.opts

        # force using warmup forward, dataloader, cnn root
        self.model.root_basis = 'cnn'
        self.model.use_cam = False
        # self.model.forward = self.model.forward_warmup
        full_loader = self.dataloader  # store original loader
        self.dataloader = range(200)
        original_rp = self.model.nerf_root_rts
        self.model.nerf_root_rts = self.model.dp_root_rts
        del self.model.dp_root_rts
        self.num_epochs = opts.warmup_pose_ep
        self.model.is_warmup_pose=True

        
        pose_states = torch.load(opts.pose_cnn_path, map_location='cpu')
        pose_states = self.rm_module_prefix(pose_states, 
                prefix='module.nerf_root_rts')
        self.model.nerf_root_rts.load_state_dict(pose_states, 
                                                    strict=False)

        # extract camera and near far planes
        self.extract_cams(full_loader)
        # import pdb;pdb.set_trace()
        # restore dataloader, rts, forward function
        self.model.root_basis=opts.root_basis
        self.model.use_cam = opts.use_cam
        self.model.forward = self.model.forward_default
        self.dataloader = full_loader
        del self.model.nerf_root_rts
        self.model.nerf_root_rts = original_rp
        self.num_epochs = opts.num_epochs
        self.model.is_warmup_pose=False

        # start from low learning rate again
        self.init_training()
        self.model.total_steps = 0
        self.model.progress = 0.
            
    def train_one_epoch(self, epoch, target):
        """
        training loop in a epoch
        """
        opts = self.opts
        # self.model.train()
        dataloader = self.trainloader
        print(f'start training epoch {epoch}')
        self.model.target = target
        # if not warmup: dataloader.sampler.set_epoch(epoch) # necessary for shuffling
        for i, batch in tqdm(enumerate(dataloader)):
            # if i==200*opts.accu_steps:
            #     break
            # if abs(batch['frameid']-target)/self.model.num_fr > (epoch/self.num_epochs+0.05) :continue
            # if i<21: continue
            opt = self.model.optim
            gaussians = self.model.gaussians
            self.model.total_steps += 1
            iteration = self.model.total_steps
            # if abs(i-target)/self.model.num_fr > (epoch/opts.num_epochs)+0.02:
            #     continue
            # gaussians.update_learning_rate(iteration)
            
            


#            self.optimizer.zero_grad()
            self.accu_steps=5
                
            #for j in range(10000):
                    
            total_loss,aux_out = self.model(batch)
            total_loss = total_loss/self.accu_steps
            total_loss.mean().backward()
            # import pdb;pdb.set_trace()
            # print(j)
            # if j%20==0:
            #     cv2.imwrite('test.png',aux_out['render'].squeeze().moveaxis(0,-1)[:,:,[2,1,0]].detach().cpu().numpy()*255.)
            image, viewspace_point_tensor, visibility_filter, radii = aux_out["render"], aux_out["viewspace_points"], aux_out["visibility_filter"], aux_out["radii"]
            with torch.no_grad():
                if (self.model.total_steps)%self.accu_steps == 0:
                    self.clip_grad(aux_out)
                    # if iteration < self.model.num_epochs*self.model.num_fr*(2/4):
                    #     self.zero_grad(gaussians._xyz)
                    #     self.zero_grad(gaussians._scaling)
                    self.zero_grad(gaussians._xyz)
                    # self.zero_grad(gaussians._scaling)
                    # self.zero_grad(gaussians._opacity)
                    # self.zero_grad(gaussians._rotation)
                    # self.dec_grad(gaussians._xyz)
                    # self.dec_grad(gaussians._scaling)
                    # self.dec_grad(gaussians._opacity)
                    # self.dec_grad(gaussians._rotation)
                    # if iteration > 50:
                    #     pass# self.zero_grad(gaussians._features_dc)
                    # import pdb;pdb.set_trace()
                    # if iteration > self.model.num_epochs*self.model.num_fr*(4/4):
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            # if j%200==0:
            #     temp_dir = self.save_dir+"/imgs"
            #     with torch.no_grad():
            #         self.model.save_imgs(temp_dir,epoch,0,use_deform=False,novel_cam=True,save_img=False)
            #         self.model.save_imgs(temp_dir,epoch,0,use_deform=False,novel_cam=True,save_img=False,bone_color=True)
            
                # if iteration >= opt.densify_until_iter:
                #     self.accu_steps = 1
                #     self.model.use_diffusion = True
                    # gaussians.optimizer.step()
                    # gaussians.optimizer.zero_grad()
                # if iteration > self.model.num_epochs*self.model.num_fr*(0/4):
                #     # Keep track of max radii in image-space for pruning
                #     gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                #     gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                #     if iteration % 2000 == 0:
                #         pnum = gaussians.densify_and_prune(self.model.optim.densify_grad_threshold, min_opacity=0.01, extent=0.5, max_screen_size=1)
                #         print('gaussian num:',gaussians._xyz.shape[0])
                    
                    # if (iteration % 3000) == 0:
                    #     gaussians.reset_opacity()
                # else:
                #     self.model.use_delta_scale=False
                

                # if aux_out['nerf_root_rts_g']>1*opts.clip_scale and \
                #                 self.model.total_steps>200*self.accu_steps:
                #     latest_path = '%s/params_latest.pth'%(self.save_dir)
                #     self.load_network(latest_path, is_eval=False, rm_prefix=False)
                
            for i,param_group in enumerate(self.optimizer.param_groups):
                aux_out['lr_%02d'%i] = param_group['lr']

            
            for item in aux_out.keys():
                if item[-5:] == '_loss':
                    self.writer.add_scalar('train_loss/'+item,aux_out[item],iteration)
            self.writer.add_scalar('train_gaussian/number',self.model.gaussians._xyz.shape[0],iteration)
            if self.model.total_steps%10==0:
                try:
                    cv2.imwrite('test.png',aux_out['rendered_novel'].squeeze().moveaxis(0,-1)[:,:,[2,1,0]].detach().cpu().numpy()*255.)
                except:pass
            # torch.cuda.empty_cache()
            
    def train_bones(self, epoch, target,dup=False):
        """
        training loop in a epoch
        """
        opts = self.opts
        # self.model.train()
        print('training frame',target)
        dataloader = self.dataloader
        if dup:
            origin = -1
            if target > self.center_frame:
                origin = target-1
            else:
                origin = target+1
            self.model.bone_optimizer.zero_grad()
            with torch.no_grad():
                self.model.bones_rts_frame[target] = self.model.bones_rts_frame[origin]
        # if not warmup: dataloader.sampler.set_epoch(epoch) # necessary for shuffling
        for i, batch in enumerate(dataloader):
            if i<target :continue
            for j in tqdm(range(epoch)):
                total_loss,aux_out = self.model(batch,frame_bone=True)
                total_loss.mean().backward()
                # self.zero_grad(self.model.gaussians._xyz)
                # self.zero_grad(self.model.gaussians._scaling)
                # # self.zero_grad(self.model.gaussians._opacity)
                # self.zero_grad(self.model.gaussians._rotation)
                # # self.zero_grad(self.model.gaussians._features_dc)
                # self.model.gaussians.optimizer.step()
                # self.model.gaussians.optimizer.zero_grad()
                with torch.no_grad():
                    self.model.bones_rts_frame.grad[:,:,3:]*=5
                self.model.bone_optimizer.step()
                self.model.bone_optimizer.zero_grad()
            # temp_dir = self.save_dir+"/frames"
            # if not os.path.isdir(temp_dir):
            #     os.makedirs(temp_dir)
            # self.model.save_imgs(temp_dir,target,0,use_deform=True,novel_cam=True,save_img=False,fixed_frame=target,frame_bone=True)
            # self.model.save_imgs(temp_dir,target,0,use_deform=True,novel_cam=True,save_img=False,bone_color=True,fixed_frame=target,frame_bone=True)
            break
                    
    def update_cvf_indicator(self, i):
        """
        whether to update canoical volume features
        0: update all
        1: freeze 
        """
        opts = self.opts

        # during kp reprojection optimization
        if (opts.freeze_proj and self.model.progress >= opts.proj_start and \
               self.model.progress < (opts.proj_start+opts.proj_end)):
            self.model.cvf_update = 1
        else:
            self.model.cvf_update = 0
        
        # freeze shape after rebone        
        if self.model.counter_frz_rebone > 0:
            self.model.cvf_update = 1

        if opts.freeze_cvf:
            self.model.cvf_update = 1
    
    def update_shape_indicator(self, i):
        """
        whether to update shape
        0: update all
        1: freeze shape
        """
        opts = self.opts
        # incremental optimization
        # or during kp reprojection optimization
        if (opts.model_path!='' and \
        self.model.progress < opts.warmup_steps)\
         or (opts.freeze_proj and self.model.progress >= opts.proj_start and \
               self.model.progress <(opts.proj_start + opts.proj_end)):
            self.model.shape_update = 1
        else:
            self.model.shape_update = 0

        # freeze shape after rebone        
        if self.model.counter_frz_rebone > 0:
            self.model.shape_update = 1

        if opts.freeze_shape:
            self.model.shape_update = 1
    
    def update_root_indicator(self, i):
        """
        whether to update root pose
        1: update
        0: freeze
        """
        opts = self.opts
        if (opts.freeze_proj and \
            opts.root_stab and \
           self.model.progress >=(opts.frzroot_start) and \
           self.model.progress <=(opts.proj_start + opts.proj_end+0.01))\
           : # to stablize
            self.model.root_update = 0
        else:
            self.model.root_update = 1
        
        # freeze shape after rebone        
        if self.model.counter_frz_rebone > 0:
            self.model.root_update = 0
        
        if opts.freeze_root: # to stablize
            self.model.root_update = 0
    
    def update_body_indicator(self, i):
        """
        whether to update root pose
        1: update
        0: freeze
        """
        opts = self.opts
        if opts.freeze_proj and \
           self.model.progress <=opts.frzbody_end: 
            self.model.body_update = 0
        else:
            self.model.body_update = 1

        
    def select_loss_indicator(self, i):
        """
        0: flo
        1: flo/sil/rgb
        """
        opts = self.opts
        if not opts.root_opt or \
            self.model.progress > (opts.warmup_steps):
            self.model.loss_select = 1
        elif i%2 == 0:
            self.model.loss_select = 0
        else:
            self.model.loss_select = 1

        #self.model.loss_select=1
        

    def reset_hparams(self, epoch):
        """
        reset hyper-parameters based on current geometry / cameras
        """
        opts = self.opts
        # mesh_rest = self.model.latest_vars['mesh_rest']

        # reset object bound, for feature matching
        # if epoch>int(self.num_epochs*(opts.bound_reset)):
        #     if mesh_rest.vertices.shape[0]>100:
        #         self.model.latest_vars['obj_bound'] = 1.2*np.abs(mesh_rest.vertices).max(0)
        
        # reinit bones based on extracted surface
        # only reinit for the initialization phase
        if opts.lbs and opts.model_path=='' and ((epoch==0)):
            reinit_bones(self.model, self.model.gaussians.get_xyz, opts.num_bones)
            self.init_training() # add new params to optimizer
            if epoch>0:
                # freeze weights of root pose in the following 1% iters
                self.model.counter_frz_rebone = 0.01
            #     #reset error stats
            #     self.model.latest_vars['fp_err']      [:]=0
            #     self.model.latest_vars['flo_err']     [:]=0
            #     self.model.latest_vars['sil_err']     [:]=0
            #     self.model.latest_vars['flo_err_hist'][:]=0

        # # need to add bones back at 2nd opt
        # if opts.model_path!='':
        #     self.model.networks['bones'] = self.model.bones

        # # add nerf-skin when the shape is good
        # if opts.lbs and opts.nerf_skin and \
        #         epoch==int(self.num_epochs*opts.dskin_steps):
        #     self.model.networks['nerf_skin'] = self.model.nerf_skin

        # self.broadcast()

    def broadcast(self):
        """
        broadcast variables to other models
        """
        dist.barrier()
        if self.opts.lbs:
            dist.broadcast_object_list(
                    [self.model.num_bones, 
                    self.model.num_bone_used,],
                    0)
            dist.broadcast(self.model.bones,0)
            dist.broadcast(self.model.nerf_body_rts[1].rgb[0].weight, 0)
            dist.broadcast(self.model.nerf_body_rts[1].rgb[0].bias, 0)

        dist.broadcast(self.model.near_far,0)
   
    def clip_grad(self, aux_out):
        """
        gradient clipping
        """
        is_invalid_grad=False
        grad_nerf_skin=[]
        grad_nerf_root_rts=[]
        grad_nerf_body_rts=[]
        grad_root_code=[]
        grad_pose_code=[]
        grad_env_code=[]
        grad_vid_code=[]
        grad_bones=[]
        grad_skin_aux=[]
        grad_ks=[]
        grad_nerf_dp=[]
        grad_csenet=[]
        for name,p in self.model.named_parameters():
            try: 
                pgrad_nan = p.grad.isnan()
                if pgrad_nan.sum()>0: 
                    print(name)
                    is_invalid_grad=True
            except: pass
            
            if 'nerf_skin' in name:
                grad_nerf_skin.append(p)
            elif 'nerf_root_rts' in name:
                grad_nerf_root_rts.append(p)
            elif 'nerf_body_rts' in name:
                grad_nerf_body_rts.append(p)
            elif 'root_code' in name:
                grad_root_code.append(p)
            elif 'pose_code' in name or 'rest_pose_code' in name:
                grad_pose_code.append(p)
            elif 'env_code' in name:
                grad_env_code.append(p)
            elif 'vid_code' in name:
                grad_vid_code.append(p)
            elif 'module.bones' == name:
                grad_bones.append(p)
            elif 'module.skin_aux' == name:
                grad_skin_aux.append(p)
            elif 'module.ks_param' == name:
                grad_ks.append(p)
            elif 'nerf_dp' in name:
                grad_nerf_dp.append(p)
            elif 'csenet' in name:
                grad_csenet.append(p)
            else: continue
        # import pdb;pdb.set_trace()
        # freeze root pose when using re-projection loss only
        if self.model.root_update == 0:
            self.zero_grad_list(grad_root_code)
            self.zero_grad_list(grad_nerf_root_rts)
        if self.model.body_update == 0:
            self.zero_grad_list(grad_pose_code)
            self.zero_grad_list(grad_nerf_body_rts)
        if self.opts.freeze_body_mlp:
            self.zero_grad_list(grad_nerf_body_rts)
        if self.model.shape_update == 1:
            #TODO add skinning 
            self.zero_grad_list(grad_bones)
            self.zero_grad_list(grad_nerf_skin)
            self.zero_grad_list(grad_skin_aux)
        if self.model.cvf_update == 1:
            self.zero_grad_list(grad_csenet)
        if self.opts.freeze_coarse:
            pass
           
        clip_scale=self.opts.clip_scale
 
        #TODO don't clip root pose
        aux_out['nerf_skin_g']     = clip_grad_norm_(grad_nerf_skin,     .1*clip_scale)
        aux_out['nerf_root_rts_g'] = clip_grad_norm_(grad_nerf_root_rts,100*clip_scale)
        aux_out['nerf_body_rts_g'] = clip_grad_norm_(grad_nerf_body_rts,100*clip_scale)
        aux_out['root_code_g']= clip_grad_norm_(grad_root_code,          .1*clip_scale)
        aux_out['pose_code_g']= clip_grad_norm_(grad_pose_code,         100*clip_scale)
        aux_out['env_code_g']      = clip_grad_norm_(grad_env_code,      .1*clip_scale)
        aux_out['vid_code_g']      = clip_grad_norm_(grad_vid_code,      .1*clip_scale)
        aux_out['bones_g']         = clip_grad_norm_(grad_bones,          1*clip_scale)
        aux_out['skin_aux_g']   = clip_grad_norm_(grad_skin_aux,         .1*clip_scale)
        aux_out['ks_g']            = clip_grad_norm_(grad_ks,            .1*clip_scale)
        aux_out['nerf_dp_g']       = clip_grad_norm_(grad_nerf_dp,       .1*clip_scale)
        aux_out['csenet_g']        = clip_grad_norm_(grad_csenet,        .1*clip_scale)

        #if aux_out['nerf_root_rts_g']>10:
        #    is_invalid_grad = True
        # if is_invalid_grad:
        #     self.zero_grad_list(self.model.parameters())
            
    @staticmethod
    def find_nerf_coarse(nerf_model):
        """
        zero grad for coarse component connected to inputs, 
        and return intermediate params
        """
        param_list = []
        input_layers=[0]+nerf_model.skips

        input_wt_names = []
        for layer in input_layers:
            input_wt_names.append(f"xyz_encoding_{layer+1}.0.weight")

        for name,p in nerf_model.named_parameters():
            if name in input_wt_names:
                # get the weights according to coarse posec
                # 63 = 3 + 60
                # 60 = (num_freqs, 2, 3)
                out_dim = p.shape[0]
                pos_dim = nerf_model.in_channels_xyz-nerf_model.in_channels_code
                # TODO
                num_coarse = 8 # out of 10
                #num_coarse = 10 # out of 10
                #num_coarse = 1 # out of 10
           #     p.grad[:,:3] = 0 # xyz
           #     p.grad[:,3:pos_dim].view(out_dim,-1,6)[:,:num_coarse] = 0 # xyz-coarse
                p.grad[:,pos_dim:] = 0 # others
            else:
                param_list.append(p)
        return param_list

    @staticmethod 
    def render_vid(model, batch):
        opts=model.opts
        model.set_input(batch)
        rtk = model.rtk
        kaug=model.kaug.clone()
        embedid=model.embedid

        rendered, _ = model.nerf_render(rtk, kaug, embedid, ndepth=opts.ndepth)
        if 'xyz_camera_vis' in rendered.keys():    del rendered['xyz_camera_vis']   
        if 'xyz_canonical_vis' in rendered.keys(): del rendered['xyz_canonical_vis']
        if 'pts_exp_vis' in rendered.keys():       del rendered['pts_exp_vis']      
        if 'pts_pred_vis' in rendered.keys():      del rendered['pts_pred_vis']     
        rendered_first = {}
        for k,v in rendered.items():
            if v.dim()>0: 
                bs=v.shape[0]
                rendered_first[k] = v[:bs//2] # remove loss term
        return rendered_first 

    @staticmethod
    def extract_mesh(model,chunk,grid_size,
                      #threshold = -0.005,
                      threshold = -0.002,
                      #threshold = 0.,
                      embedid=None,
                      mesh_dict_in=None):
        opts = model.opts
        mesh_dict = {}
        if model.near_far is not None: 
            bound = model.latest_vars['obj_bound']
        else: bound=1.5*np.asarray([1,1,1])

        if mesh_dict_in is None:
            ptx = np.linspace(-bound[0], bound[0], grid_size).astype(np.float32)
            pty = np.linspace(-bound[1], bound[1], grid_size).astype(np.float32)
            ptz = np.linspace(-bound[2], bound[2], grid_size).astype(np.float32)
            query_yxz = np.stack(np.meshgrid(pty, ptx, ptz), -1)  # (y,x,z)
            #pts = np.linspace(-bound, bound, grid_size).astype(np.float32)
            #query_yxz = np.stack(np.meshgrid(pts, pts, pts), -1)  # (y,x,z)
            query_yxz = torch.Tensor(query_yxz).to(model.device).view(-1, 3)
            query_xyz = torch.cat([query_yxz[:,1:2], query_yxz[:,0:1], query_yxz[:,2:3]],-1)
            query_dir = torch.zeros_like(query_xyz)

            bs_pts = query_xyz.shape[0]
            out_chunks = []
            for i in range(0, bs_pts, chunk):
                query_xyz_chunk = query_xyz[i:i+chunk]
                query_dir_chunk = query_dir[i:i+chunk]

                # backward warping 
                if embedid is not None and not opts.queryfw:
                    query_xyz_chunk, mesh_dict = warp_bw(opts, model, mesh_dict, 
                                                   query_xyz_chunk, embedid)
                if opts.symm_shape: 
                    #TODO set to x-symmetric
                    query_xyz_chunk[...,0] = query_xyz_chunk[...,0].abs()
                xyz_embedded = model.embedding_xyz(query_xyz_chunk) # (N, embed_xyz_channels)
                out_chunks += [model.nerf_coarse(xyz_embedded, sigma_only=True)]
            vol_o = torch.cat(out_chunks, 0)
            vol_o = vol_o.view(grid_size, grid_size, grid_size)
            #vol_o = F.softplus(vol_o)

            if not opts.full_mesh:
                #TODO set density of non-observable points to small value
                if model.latest_vars['idk'].sum()>0:
                    vis_chunks = []
                    for i in range(0, bs_pts, chunk):
                        query_xyz_chunk = query_xyz[i:i+chunk]
                        if opts.nerf_vis:
                            # this leave no room for halucination and is not what we want
                            xyz_embedded = model.embedding_xyz(query_xyz_chunk) # (N, embed_xyz_channels)
                            vis_chunk_nerf = model.nerf_vis(xyz_embedded)
                            vis_chunk = vis_chunk_nerf[...,0].sigmoid()
                        else:
                            #TODO deprecated!
                            vis_chunk = compute_point_visibility(query_xyz_chunk.cpu(),
                                             model.latest_vars, model.device)[None]
                        vis_chunks += [vis_chunk]
                    vol_visi = torch.cat(vis_chunks, 0)
                    vol_visi = vol_visi.view(grid_size, grid_size, grid_size)
                    vol_o[vol_visi<0.5] = -1

            ## save color of sampled points 
            #cmap = cm.get_cmap('cool')
            ##pts_col = cmap(vol_visi.float().view(-1).cpu())
            #pts_col = cmap(vol_o.sigmoid().view(-1).cpu())
            #mesh = trimesh.Trimesh(query_xyz.view(-1,3).cpu(), vertex_colors=pts_col)
            #mesh.export('0.obj')
            #pdb.set_trace()

            print('fraction occupied:', (vol_o > threshold).float().mean())
            vertices, triangles = mcubes.marching_cubes(vol_o.cpu().numpy(), threshold)
            vertices = (vertices - grid_size/2)/grid_size*2*bound[None, :]
            mesh = trimesh.Trimesh(vertices, triangles)

            # mesh post-processing 
            if len(mesh.vertices)>0:
                if opts.use_cc:
                    # keep the largest mesh
                    mesh = [i for i in mesh.split(only_watertight=False)]
                    mesh = sorted(mesh, key=lambda x:x.vertices.shape[0])
                    mesh = mesh[-1]

                # assign color based on canonical location
                vis = mesh.vertices
                try:
                    model.module.vis_min = vis.min(0)[None]
                    model.module.vis_len = vis.max(0)[None] - vis.min(0)[None]
                except: # test time
                    model.vis_min = vis.min(0)[None]
                    model.vis_len = vis.max(0)[None] - vis.min(0)[None]
                vis = vis - model.vis_min
                vis = vis / model.vis_len
                if not opts.ce_color:
                    vis = get_vertex_colors(model, mesh, frame_idx=0)
                mesh.visual.vertex_colors[:,:3] = vis*255

        # forward warping
        if embedid is not None and opts.queryfw:
            mesh = mesh_dict_in['mesh'].copy()
            vertices = mesh.vertices
            vertices, mesh_dict = warp_fw(opts, model, mesh_dict, 
                                           vertices, embedid)
            mesh.vertices = vertices
               
        mesh_dict['mesh'] = mesh
        return mesh_dict

    def save_logs(self, log, aux_output, total_steps, epoch):
        for k,v in aux_output.items():
            self.add_scalar(log, k, aux_output,total_steps)
        
    def add_image_grid(self, rendered_seq, log, epoch):
        for k,v in rendered_seq.items():
            grid_img = image_grid(rendered_seq[k],3,3)
            if k=='depth_rnd':scale=True
            elif k=='occ':scale=True
            elif k=='unc_pred':scale=True
            elif k=='proj_err':scale=True
            elif k=='feat_err':scale=True
            else: scale=False
            self.add_image(log, k, grid_img, epoch, scale=scale)

    def add_image(self, log,tag,timg,step,scale=True):
        """
        timg, h,w,x
        """

        if self.isflow(tag):
            timg = timg.detach().cpu().numpy()
            timg = flow_to_image(timg)
        elif scale:
            timg = (timg-timg.min())/(timg.max()-timg.min())
        else:
            timg = torch.clamp(timg, 0,1)
    
        if len(timg.shape)==2:
            formats='HW'
        elif timg.shape[0]==3:
            formats='CHW'
            print('error'); pdb.set_trace()
        else:
            formats='HWC'

        log.add_image(tag,timg,step,dataformats=formats)

    @staticmethod
    def add_scalar(log,tag,data,step):
        if tag in data.keys():
            log.add_scalar(tag,  data[tag], step)

    @staticmethod
    def del_key(states, key):
        if key in states.keys():
            del states[key]
    
    @staticmethod
    def isflow(tag):
        flolist = ['flo_coarse', 'fdp_coarse', 'flo', 'fdp', 'flo_at_samp']
        if tag in flolist:
           return True
        else:
            return False

    @staticmethod
    def zero_grad_list(paramlist):
        """
        Clears the gradients of all optimized :class:`torch.Tensor` 
        """
        for p in paramlist:
            if p.grad is not None:
                p.grad.detach_()
                p.grad.zero_()

    @staticmethod
    def zero_grad(p):
        """
        Clears the gradients of all optimized :class:`torch.Tensor` 
        """
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()
            
    @staticmethod
    def dec_grad(p,index=10.):
        """
        Clears the gradients of all optimized :class:`torch.Tensor` 
        """
        if p.grad is not None:
            p.grad/=index