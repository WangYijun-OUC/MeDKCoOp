import os.path as osp
import os
import sys
import pdb
import json
import math
import copy
import torch
import random
import numpy as np
import torch.nn as nn
from collections import OrderedDict
from tqdm import tqdm
from torch.nn import functional as F
from torch.nn import Parameter
from torch.cuda.amp import GradScaler, autocast
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler
from trainers.prompt_templates import BIOMEDCOOP_TEMPLATES, CUSTOM_TEMPLATES, IMAGENET_TEMPLATES, CUSTOM_TEMPLATES
from open_clip.src.open_clip import create_model_from_pretrained, get_tokenizer
from MedSAM.segment_anything import sam_model_registry

class TextEncoder(nn.Module):
    def __init__(self, biomedclip_model):
        super().__init__()
        self.model = biomedclip_model
        self.dtype = biomedclip_model.text.transformer.dtype

    def forward(self, prompts, tokenized_prompts):

        x = self.model.encode_text(prompts,True,tokenized_prompts)

        return x



def cal_edge_emb(x, p=2, dim=1, enable_mask=False):   # v1_graph---taking the similairty by 
    ''' 
    x: (n,K)   [m+1, 1000, 1024]
    return: (n^2, K)
    '''
    x = F.normalize(x, p=p, dim=dim)    #[m+1, 1000, 1024], [100, 1024, 101]
    x_c = x
    x = x.transpose(1, 2)  #[1000, m+1, 1024]  [100, 101, 1024]
    x_r = x  # (K, n, 1) #[1000, m+1, 1024]
    # x_c = torch.transpose(x, 1, 2)  # (K, 1, n) #[1000, 1024, m+1]
    # A = torch.bmm(x_r, x_c).permute(1,2,0)  # (n, n, K) 
    edge = torch.bmm(x_r, x_c)     # [1000, m+1, m+1]

    # A = A.view(A.size(0) * A.size(1), A.size(2))  # (n^2, K)
    # print(A.size())

    b, c, p = x.size()  # [2, 3, 512]
    threshold = random.uniform(0.5, 0.6)

    zeros_vec = torch.zeros_like(edge)
    ones_vec = torch.ones_like(edge)    # [2, 3]
    eye_vec = torch.eye(c)      # [3, 3]


    value,idx = edge.max(dim=1)
    value_thre = value.unsqueeze(-1)*threshold  # [2, 3, 1]

    if enable_mask:     # FALSE
        mask = (edge >= value_thre).float()  # [2, 3, 3]
        mask += eye_vec.unsqueeze(0).cuda()  # [2, 3, 3]
        edge = edge * mask
        return edge
    #edge=F.softmax(edge,dim=2)
    #print(mask[0].sum(1))
    else:
        return edge
    # return A

    
def _get_base_text_features(cfg, classnames, biomedclip_model, text_encoder):
    device = next(text_encoder.parameters()).device     # cpu
    dtype = biomedclip_model.text.transformer.dtype     # float32
    if dtype == torch.float32:
        text_encoder = text_encoder.cuda()  

    dataset = cfg.DATASET.NAME


    classnames = [name.replace("_", " ") for name in classnames]
    tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    biomedclip_model_temp = biomedclip_model.float().eval().cuda() 

    with torch.no_grad():
        print(classnames)
        text_embedding = []
        for name in classnames :
            class_speci_prompt = []
            for i in range(cfg.TRAINER.GRAPHDCPL.N_PROMPTS):
                
                # x_tokenized = torch.cat([tokenizer(TEMPLATES.format(BIOMEDCOOP_TEMPLATES[classname][i])) for classname in classnames] )    # [2, 256] int64  
                x_tokenized = torch.cat([tokenizer(BIOMEDCOOP_TEMPLATES[name][i])] )    # [1, 256] int64
                embedding = biomedclip_model_temp.text.transformer.embeddings.word_embeddings(x_tokenized.cuda()).type(dtype)   # [1, 256, 768]

                # text_features = biomedclip_model_temp.encode_text(x_tokenized.cuda())       # [1, 512]  float32

                if dtype == torch.float32:
                    class_speci_prompt.append(text_encoder(embedding, x_tokenized.cuda()))  
                else:
                    class_speci_prompt.append(text_encoder(embedding, x_tokenized.cuda()))  # [1, 512] cuda float32
            class_speci_prompts = torch.stack(class_speci_prompt).mean(0)      # [50, 1, 512] -> [1, 512]
            text_embedding.append(class_speci_prompts)
        text_embeddings = torch.stack(text_embedding).mean(1)
    text_encoder = text_encoder.to(device)
    # biomedclip_model = biomedclip_model.to(device)
    return text_embeddings.to(device)   # [2, 512]

def _get_base_image_features(cfg, classnames, biomedclip_model, img_encoder, train_loader_x):
    device = next(img_encoder.parameters()).device
    ctx_dim = cfg.TRAINER.GRAPHDCPL.hidden_dim   # 512
    n_ctx = cfg.TRAINER.GRAPHDCPL.N_CTX  # 2
    ctx_init = cfg.TRAINER.GRAPHDCPL.CTX_INIT    # a photo of a

    assert cfg.TRAINER.GRAPHDCPL.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
    compound_prompts_depth = cfg.TRAINER.GRAPHDCPL.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts

    dtype = biomedclip_model.text.transformer.dtype
    if dtype == torch.float32:
        img_encoder = img_encoder.cuda()
        # biomedclip_model = biomedclip_model.cuda()

    print('loading medical model')
    MedSAM_CKPT_PATH = "/path/to/your/medsam.pth"
    medical_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)
    medical_model.cuda()

    visual_net = nn.Sequential(OrderedDict([
        ("linear1", nn.Linear(256, 256 // 16)),     # W1=[256, 16], b1=16, features*W1+b1 -> [4, 16]
        ("relu", nn.ReLU(inplace=True)),            
        ("linear2", nn.Linear(256 // 16, 512))      # W2=[16, 512], b2=512, features*X2+b2 -> [4, 512]
    ]))


    with torch.no_grad():
        img_feature = []
        labels = []

        # for epch in range(2):
        for batch_idx, batch in tqdm(enumerate(train_loader_x), 
                                    total=len(train_loader_x), 
                                    desc='Extracting medical image features'):
            image = batch["img"]    # [4, 3, 224, 224]
            image = image.cuda()
            # impath = batch["impath"]
            label = batch["label"]
            label = label.cuda()
 
            medical_feature = medical_model.image_encoder(nn.functional.interpolate(image, (1024, 1024))).cpu()
            mb,mc,mh,mw = medical_feature.shape
            medical_feature = medical_feature.view(mb,mc,mh*mw)
            remote_mae_feature = medical_feature.mean(dim=-1).type(dtype)
            
            image_features = img_encoder(image.type(dtype))     # [4, 512]
            # tensor cuda8 torch.float16 tensor[2, 768],cuda8,torch.float16,grad_fn 8 * [2, 768] cuda8 torch.float16

            image_features_bias = visual_net(remote_mae_feature.to(torch.float32))                        # [4, 512]
            image_features = image_features + image_features_bias.cuda()              # [4, 512]

            image_features = image_features / image_features.norm(dim=-1, keepdim=True) # [4, 512]
            
            img_feature.append(image_features)                                          # []
            labels.append(label)
        
        img_feature_list = torch.cat(img_feature, dim=0)
        label_list = torch.cat(labels, dim=0)

        sorted_labels, indices = torch.sort(label_list)     # 10837     [10837]
        img_feature_sorted = torch.index_select(img_feature_list, 0, indices)

        unique_labels, counts = torch.unique(sorted_labels, return_counts=True)     # [1533, 3154, 3196, 2954]
        num_classes = unique_labels.size(0)

        split_indices = counts.cumsum(dim=0)[:-1].tolist()      # [1533, 4687, 7883]
        features_split = torch.tensor_split(img_feature_sorted, split_indices, dim=0)       # [1533, 512] ... [2954, 512]

        class_means = [feat.mean(dim=0) for feat in features_split]     
        img_feature_means = torch.stack(class_means, dim=0)  # [num_classes, feature_dim]
        img_encoder = img_encoder.to(device)
        # biomedclip_model = biomedclip_model.to(device)
    return img_feature_means.to(device)  # [2, 512]  float32

class GraphConvolution(nn.Module):
    def __init__(self, hidden_dim, name=None, device=None, class_num=None, sparse_inputs=False, act=nn.Tanh, bias=True, dropout=0.0, mode="text"):
        super().__init__()
        self.act = nn.Tanh() if act else None
        self.device=device  # float16
        self.dropout = dropout
        self.sparse_inputs = sparse_inputs  
        self.hidden_dim = hidden_dim
        self.bias = bias
        # self.hidden_dim = 512
        self.class_num = class_num
        self.mode = mode
        self.gcn_weights = nn.Parameter(torch.ones(self.hidden_dim, self.hidden_dim))   # [hidden_dim, hidden_dim]     float32
        if self.bias:
            self.gcn_bias = nn.Parameter(torch.zeros(class_num, self.hidden_dim))       # [class_num, hidden_dim]
        self.reset_parameters_txt()

        if self.dropout>0.01:
            self.drop_flag = True
            self.dropout_layer = nn.Dropout(p=dropout)
        else:
            self.drop_flag = False

        # else:
        #     self.register_parameter('bias', None)

    def reset_parameters_txt(self):
        stdv = 1. / math.sqrt(self.gcn_weights.size(1))     # gcn_weights
        self.gcn_weights.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            nn.init.kaiming_normal_(self.gcn_weights, a=0)
            nn.init.constant_(self.gcn_bias, 0.0)
    
    def reset_parameters_v(self):
        stdv = 1. / math.sqrt(self.gcn_weights_v.size(1))     # gcn_weights
        self.gcn_weights_v.data.uniform_(-stdv, stdv)
        if self.gcn_bias_v is not None:
            #self.bias.data.uniform_(-stdv, stdv)
            nn.init.kaiming_normal_(self.gcn_weights_v, a=0)
            nn.init.constant_(self.gcn_bias_v, 0.0)

    def graph_normalization(self, A, add_self_loop=True, symmetric=True, clamp_value=None, eps=1e-5):
        if add_self_loop:
            eye = torch.eye(A.size(-1), device='cuda')  # [n, n]
            A = A + eye.unsqueeze(0).to(self.device)  # 批量添加自环 [b, n, n]

        if clamp_value is not None:
            A = torch.clamp(A, min=clamp_value[0], max=clamp_value[1])

        d = A.sum(dim=-1) + eps  # [b, n]
        
        if symmetric:
            # D^{-1/2} A D^{-1/2}
            D = torch.pow(d, -0.5)  # [b, n]
            D = torch.diag_embed(D)  # [b, n, n]
            norm_A = D @ A @ D
        else:
            # D^{-1} A
            D = torch.pow(d, -1)  # [b, n]
            D = torch.diag_embed(D)  # [b, n, n]
            norm_A = D @ A

        return norm_A

    def forward(self, feat, adj):
        # feat [batch, feature_dim, node_num]
        # adj [batch, node_num, node_num]
        # if self.mode == "text" :
        x = feat                        # [2, 3, 512]
        # node_size = adj.size()[1]  
        if self.drop_flag:
            x = self.dropout_layer(feat)
        adj = torch.clip(adj, min=0.0)  # [2, 3, 3]
        # I = torch.eye(node_size, device='cuda').unsqueeze(dim=0).to(self.device)   
        # adj = adj + I      # [1000, m+1, m+1]
        adj = self.graph_normalization(adj, add_self_loop=True, symmetric=True)  #[1000, m+1, m+1]  
        # x = x.transpose(1, 2)
        pre_sup = torch.matmul(x, self.gcn_weights)     # [m+1, 1000, 1024]         Ctt * Wtt
        output = torch.matmul(adj, pre_sup)             # [1000, m+1, 1024]         ε^tt * Ctt * Wtt

        if self.bias:
            output += self.gcn_bias.unsqueeze(1)
        if self.act is not None:
            return self.act(output[:, 0, :])
        else:
            return output[:, 0, :]

class MultiModalPromptLearner(nn.Module):
    def __init__(self, cfg, classnames, biomedclip_model, base_text_features, base_img_features):
        super().__init__()
        n_cls = len(classnames)         
        n_ctx = cfg.TRAINER.GRAPHDCPL.N_CTX  # 2
        ctx_init = cfg.TRAINER.GRAPHDCPL.CTX_INIT    # a photo of a
        self.mode = cfg.MODEL.mode
        print("MODE:", self.mode)

        self.dtype = biomedclip_model.text.transformer.dtype    # float32
        # ctx_dim = biomedclip_model.ln_final.weight.shape[0]
        # ctx_dim = cfg.TRAINER.GRAPHDCPL.hidden_dim
        self.ctx_dim = 768
        # clip_imsize = biomedclip_model.visual.input_resolution
        clip_imsize = 224
        
        self.tokenizer = get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"
        # Default is 1, which is compound shallow prompting
        assert cfg.TRAINER.GRAPHDCPL.PROMPT_DEPTH >= 1, "For MaPLe, PROMPT_DEPTH should be >= 1"
        self.compound_prompts_depth = cfg.TRAINER.GRAPHDCPL.PROMPT_DEPTH  # max=12, but will create 11 such shared prompts
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init and (n_ctx) <= 4:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            prompt = self.tokenizer(ctx_init)
            with torch.no_grad():
                embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(prompt).type(self.dtype)   # [1, 256, 768]
            ctx_vectors = embedding[0, 1: 1 + n_ctx, :]     # [2, 768]
            prompt_prefix = ctx_init
        else:
            # random initialization
            ctx_vectors = torch.empty(n_ctx, self.ctx_dim, dtype=self.dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)
        print('DCPL design: Domain Controlled')
        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of DCPL context words (tokens): {n_ctx}")
        # These below, related to the shallow prompts
        # Linear layer so that the tokens will project to 512 and will be initialized from 768
        # self.proj = nn.Linear(self.ctx_dim, 768) # Linear(in_features=512, out_features=768, bias=True)
        # self.proj.half()                    # Linear(in_features=512, out_features=768, bias=True)
        self.ctx = nn.Parameter(ctx_vectors)    # [2, 768]
        # These below parameters related to the shared prompts
        # Define the compound prompts for the deeper layers
        # self.meta_net = nn.Sequential(OrderedDict([
        #     ("linear1", nn.Linear(512, 512 // 16)),
        #     ("relu", nn.ReLU(inplace=True)),
        #     ("linear2", nn.Linear(512 // 16, self.ctx_dim))
        # ]))

        print("CLIP CTX_DIM=", self.ctx_dim)     # 512
        self.meta_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(256, 256 // 16)),     # W1=[256, 16], b1=16, features*W1+b1 -> [4, 16]
            ("relu", nn.ReLU(inplace=True)),            # 激活函数
            ("linear2", nn.Linear(256 // 16, self.ctx_dim))      # W2=[16, 512], b2=512, features*X2+b2 -> [4, 512]
        ]))

        if cfg.TRAINER.GRAPHDCPL.PREC == "fp16":
            self.meta_net.to(dtype = self.dtype)
        # Minimum can be 1, which defaults to shallow MaPLe
        # compound prompts
        # self.compound_prompts_text = nn.ParameterList([nn.Parameter(torch.empty(n_ctx, 512))
        #                                               for _ in range(self.compound_prompts_depth - 1)])
        # for single_para in self.compound_prompts_text:
        #     nn.init.normal_(single_para, std=0.02)

        # Also make corresponding projection layers, for each prompt
        # single_layer = nn.Linear(self.ctx_dim, 768)
        # self.compound_prompt_projections = _get_clones(single_layer, self.compound_prompts_depth - 1)

        classnames = [name.replace("_", " ") for name in classnames]        # ['glioma', 'meningioma'] for base
        name_lens = [len(self.tokenizer(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames] # ['a photo of a glioma.', 'a photo of a meningioma.']

        tokenized_prompts = torch.cat([self.tokenizer(p) for p in prompts])  # (n_cls, context length)    [2, 256]
        # Also create frozen CLIP
        # biomedclip_model_temp,_ = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
        # biomedclip_model_temp = biomedclip_model_temp.float().eval().cuda()
        with torch.no_grad():
            embedding = biomedclip_model.text.transformer.embeddings.word_embeddings(tokenized_prompts).type(self.dtype)       # [2, 256, 768]

        if self.mode == "train" :
            self.fc = nn.Linear(512, n_cls, bias=False)
            self.fc.weight.data[:len(classnames)] = base_img_features
            # base_img_features = self._get_base_image_features(cfg, classnames, biomedclip_model, image_encoder, train_loader_x)
            # self.register_buffer("base_img_features", base_img_features)
        else :
            self.new_fc = nn.Linear(512, n_cls, bias=False)
            self.new_fc.weight.data[:len(classnames)] = base_img_features
            # self.new_angles_mean = nn.Parameter(torch.zeros(n_cls), requires_grad=False)
            # self.new_angles_var = nn.Parameter(torch.zeros(n_cls), requires_grad=False)
            # new_img_features = self._get_new_image_features(cfg, classnames, biomedclip_model, image_encoder, train_loader_x)

        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS    [2, 1, 768]
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])  # CLS, EOS      [2, 253, 512]


        # GraphAdapter
        self.register_buffer("base_text_features", base_text_features) #[1000, 1024]
        self.register_buffer("base_img_features", base_img_features)
        # self.alpha_it = cfg.TRAINER.GRAPHADAPTER.ALPHA
        self.alpha_it = 0.7
        self.beta_it = 0.3
        self.node_num = 1
        self.hidden_dim = 512

        print("Building Subgraph....")
        self.GCN_tt = GraphConvolution(self.hidden_dim, name='metagraph', device=self.dtype, class_num=base_text_features.size()[0])
        self.GCN_it = GraphConvolution(self.hidden_dim, name='metagraph', device=self.dtype, class_num=base_text_features.size()[0])
        self.subgraph_num = 1
        print("Subgraph Number ", self.subgraph_num)

        # LRT
        self.relu = nn.LeakyReLU(0.2)
        self.scale_mm = nn.Parameter(torch.ones(1), requires_grad=True)
        # self.angles_mean = nn.Parameter(torch.zeros(n_cls), requires_grad=False)
        # self.angles_var = nn.Parameter(torch.zeros(n_cls), requires_grad=False)
        self.dropout = 0.2
        self.num_features = 512
        self.GCN_fuse = GraphConvolution(self.hidden_dim, name='metagraph', device=self.dtype, class_num=base_text_features.size()[0], act=None, dropout = self.dropout, mode="image")
        
        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.classnames = classnames

    def reset_parameters(self):
        for i in range(self.node_num):
            stdv = 1. / math.sqrt(self.graph_node[i].size(0))
            self.graph_node[i].data.uniform_(-stdv, stdv)

    def construct_prompts(self, ctx, prefix, suffix, label=None):
        # dim0 is either batch_size (4, during training) or n_cls (2, during testing)
        # ctx: context tokens, with shape of (dim0, n_ctx, ctx_dim)
        # prefix: the sos token, with shape of (n_cls, 1, ctx_dim)
        # suffix: remaining tokens, with shape of (n_cls, *, ctx_dim)

        if label is not None:
            prefix = prefix[label]
            suffix = suffix[label]

        prompts = torch.cat(
            [
                prefix,  # (dim0, 1, dim=512)   [2, 1, 768]
                ctx,  # (dim0, n_ctx, dim=512)  [2, 2, 768]
                suffix,  # (dim0, *, dim=512)   [2, 74, 768]
            ],
            dim=1,
        )

        return prompts  

    def build_subgraph(self, text_features) :
        # text_features = text_encoder(pts_i, self.tokenized_prompts) # [2, 512]
        # grouped_text_features = self.AttenMerge(self.learnable_token[i:i + 1,:], text_features)      #   [1, 512]
        graph_o_t_all = []
        for id in range(1):
            fuse_features = self.build_prompt_graph(self.base_text_features, text_features)         # [2, 512]
            graph_o_t = (self.base_img_features)*self.alpha_it + (1-self.alpha_it)*fuse_features   # b*ztt + (1-b)*zvt    
            graph_o_t_all.append(graph_o_t)         # [2, 1, 512]
        graph_o_t = torch.stack(graph_o_t_all, dim=0).mean(dim=0)   #  [2, 512]

        final_graph_feature = self.beta_it * self.base_text_features + (1 - self.beta_it) * graph_o_t.squeeze()    # a*zt + (1-a)*z't
        return final_graph_feature
        # final_graph_feature.append(self.beta_it * self.base_text_features + (1 - self.beta_it) * graph_o_t.squeeze())    # a*zt + (1-a)*z't
    # final_graph_features = torch.stack(final_graph_feature)

    def build_prompt_graph(self, text_features, prompts):
        # [4, 4, 512]
        if self.mode=="test" :
            # print("using test...")
            imgproto_features = F.normalize(self.new_fc.weight.data[:len(self.classnames)], p=2, dim=-1)
            # imgproto_features = F.normalize(self.base_img_features, p=2, dim=-1)
        else :
            imgproto_features = F.normalize(self.fc.weight.data[:len(self.classnames)], p=2, dim=-1)         # [2, 512]
        # imgproto_features = F.normalize(self.base_img_features, p=2, dim=-1)
        with torch.no_grad():
            inputs_text = prompts.unsqueeze(dim=1)    #[2, 1, 512]       zt
            # inputs_text = text_features.unsqueeze(dim=1)    #[2, 1, 512]       zt
            node_cluster_t = text_features.view(1, text_features.size()[0]//1, 1, text_features.size()[1])    # [1, 2, 1, 512]
            node_cluster_tt =  node_cluster_t[:, :, 0, :].repeat(inputs_text.size()[0], 1, 1)  # t->t [1,2,512]->[2,2,512]
            feat_tt = torch.cat([inputs_text, node_cluster_tt], dim=1)      # [2, 3, 512]
            feat_tt = feat_tt.transpose(1, 2)          # [2, 512, 3]
            node_num = feat_tt.size()[-1]
            edge_tt = cal_edge_emb(feat_tt)    # εtt zt与文本子图Ct的边    [2, 3, 3]
        # text_features = text_features + prompts
        imgproto_features = imgproto_features.unsqueeze(0).expand(node_num, -1, -1)      # [1, 2, 512] -> [2, 2, 512]

        # text_features = text_features.unsqueeze(0).expand(self.n_cls, -1, -1)              # [1, 2, 512] -> [2, 2, 512]
        # text_features = text_features.transpose(1, 2)    # [1, 512, 2]
        
        # edge_text = cal_edge_emb(text_features, enable_mask=True)       # [4, 4, 4]
        # adj = norm_adj_batch(edge_text)        # D^{-1/2} * A * D^{-1/2}     [4, 4, 4]
        img_gfeature = self.relu(self.GCN_fuse(imgproto_features.transpose(0, 1), edge_tt))  # [4, 4, 512]

        # fused_global_proto = img_gfeature + text_features  # [4, 2, 512]

        # x_normalized = F.normalize(x, p=2, dim=-1)  # [4, 2, 512]
        # x_fuse = self.scale_mm * text_features + img_gfeature # [4, 2, 512]

        return img_gfeature  # [1, 512]
        
    def forward(self, maefeatures, text_encoder):
        ctx = self.ctx # (n_ctx, ctx_dim)   [2, 768]
        
        bias = self.meta_net(maefeatures)  # (batch, ctx_dim)   [4, 768]
        bias = bias.unsqueeze(1)  # (batch, 1, ctx_dim)         [4, 1, 768]
        ctx = ctx.unsqueeze(0)  # (1, n_ctx, ctx_dim)           [1, 2, 768]
        ctx_shifted = ctx + bias  # (batch, n_ctx, ctx_dim)     [4, 2, 768]
        prefix = self.token_prefix  # [2, 1, 768]
        suffix = self.token_suffix  # [2, 253, 768]
        
        prompts = []
            
        for ctx_shifted_i in ctx_shifted:     
            # [2, 768]
            graph_o_t_all = []
            ctx_i = ctx_shifted_i.unsqueeze(0).expand(self.n_cls, -1, -1)   # [2, 2, 768]   
            pts_i = self.construct_prompts(ctx_i, prefix, suffix)           # (n_cls, n_tkn, ctx_dim)   [2, 256, 768]
            prompts.append(pts_i)                                           # [4, 2, 256, 768]
        prompts = torch.stack(prompts)
        
        return prompts    # [4, 2, 512]

class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, biomedclip_model, train_loader_x):
        super().__init__()
        # self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = biomedclip_model.visual
        self.text_encoder = TextEncoder(biomedclip_model)
        self.logit_scale = biomedclip_model.logit_scale
        self.dtype = biomedclip_model.text.transformer.dtype       # torch.float32
        self.classnames = classnames
        

        # GraphAdapter
        base_text_features = _get_base_text_features(cfg, classnames, biomedclip_model, self.text_encoder)
        base_img_features = _get_base_image_features(cfg, classnames, biomedclip_model, self.image_encoder, train_loader_x)
        
        self.prompt_learner = MultiModalPromptLearner(cfg, classnames, biomedclip_model, base_text_features, base_img_features)

        print('loading medical model')
        MedSAM_CKPT_PATH = "/path/to/your/medsam.pth"
        self.medical_model = sam_model_registry['vit_b'](checkpoint=MedSAM_CKPT_PATH)

        # self.visual_net = nn.Sequential(OrderedDict([
        #     ("linear1", nn.Linear(512, 512 // 16)),
        #     ("relu", nn.ReLU(inplace=True)),
        #     ("linear2", nn.Linear(512 // 16, 512))
        # ]))
        self.ctx_dim = self.prompt_learner.ctx_dim
        self.visual_net = nn.Sequential(OrderedDict([
            ("linear1", nn.Linear(256, 256 // 16)),     # W1=[256, 16], b1=16, features*W1+b1 -> [4, 16]
            ("relu", nn.ReLU(inplace=True)),            
            ("linear2", nn.Linear(256 // 16, 512))      # W2=[16, 512], b2=512, features*X2+b2 -> [4, 512]
        ]))

        if cfg.TRAINER.GRAPHDCPL.PREC == "fp16":
            self.visual_net.half()
            # self.p1.half()
            # self.p2.half()

    def forward(self, image, label=None):
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        logit_scale = self.logit_scale.exp()        # tensor(100., device='cuda:7')
        
        with torch.no_grad():           
            medical_feature = self.medical_model.image_encoder(nn.functional.interpolate(image, (1024, 1024)))
            mb,mc,mh,mw = medical_feature.shape
            medical_feature = medical_feature.view(mb,mc,mh*mw)
            remote_mae_feature = medical_feature.mean(dim=-1).type(self.dtype)

        # print(image.shape)                  # [4, 3, 224, 224]
        # print(medical_feature.shape)        # [4, 256, 4096]
        # print(remote_mae_feature.shape)     # [4, 256]
        # print(remote_mae_feature.type(self.dtype))  # 4*256 tensor device=cuda dtype=float16

        text_features = self.prompt_learner(remote_mae_feature.type(self.dtype), self.text_encoder)    # [4, 2, 512]

        image_features = self.image_encoder(image.type(self.dtype))

        image_features_bias = self.visual_net(remote_mae_feature.type(self.dtype))  # [4, 512]    
        image_features = image_features + image_features_bias              # [4, 512]


        image_features = image_features / image_features.norm(dim=-1, keepdim=True) # [4, 512]
        # text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logits = []
        # logits = torch.zeros((mb,len(self.classnames)),dtype=self.dtype, requires_grad=True).cuda()
        # img = torch.randn(4, 512).cuda()
        # tokenized_prompts [2, 77]
        # pts_i [2, 77, 512]
        # deep_compound_prompts_text = List 8 * Parameter[2, 512]
        for pts_i, imf_i in zip(text_features.type(self.dtype), image_features.type(self.dtype)):
            # v = [1, 512]
            # text_features = self.text_encoder(pts_i, tokenized_prompts, deep_compound_prompts_text)
            txt_features = self.text_encoder(pts_i, self.tokenized_prompts) # [2, 512]
            x_fuse = self.prompt_learner.build_subgraph(txt_features)
            normalized_text_features = x_fuse / x_fuse.norm(dim=-1, keepdim=True)     # [1, 512]
            l_i = logit_scale * imf_i @ normalized_text_features.t() 
            logits.append(l_i)  # [2] 
        logits = torch.stack(logits)    # [4, 2]

        if self.prompt_learner.training:
            return F.cross_entropy(logits, label)

        return logits

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

@TRAINER_REGISTRY.register()
class medkcoop(TrainerX):
    def check_cfg(self, cfg):
        assert cfg.TRAINER.GRAPHDCPL.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        print("Using Model GRAPHDCPL...")
        
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading BiomedCLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        biomedclip_model, preprocess = create_model_from_pretrained('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

        if cfg.TRAINER.GRAPHDCPL.PREC == "fp32" or cfg.TRAINER.GRAPHDCPL.PREC == "amp":
            # CLIP's default precision is fp16
            biomedclip_model.float()

        # torch.cuda.set_device(0)
        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, biomedclip_model.eval(), self.train_loader_x)

        print("Turning off gradients in both the image and the text encoder")
        name_to_update = "prompt_learner"

        for name, param in self.model.named_parameters():
            if name_to_update not in name:
                # Make sure that VPT prompts are updated
                if "VPT" in name:
                # if "VPT" in name or 'visual_net ' in name:
                    param.requires_grad_(True)
                else:
                    param.requires_grad_(False)
            if "visual_net" in name:
                param.requires_grad_(True)

        for param in self.model.prompt_learner.parameters():
            param.requires_grad_(True)

        # Double check
        enabled = set()
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                enabled.add(name)
        print(f"Parameters to be updated: {enabled}")

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # self.model = self.model.float()

        # NOTE: only give prompt_learner to the optimizer
        # opt_para = list(self.model.prompt_learner.parameters())+list(self.model.prompt_learner.AttenMerge.parameters())
        # opt_para.append(self.model.learnable_token)
        # opt_para = list(self.model.prompt_learner.parameters())
        # opt_para.append(self.model.prompt_learner.learnable_token)
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)

        # module_state_dict = nn.ModuleDict()
        # param_state_list = nn.ParameterList()
        # param_state_list.append(self.model.prompt_learner.learnable_token)
        # module_state_dict['prompt_learner'] = self.model.prompt_learner
        # module_state_dict['visual_net'] = self.model.visual_net
        # # module_state_dict['AttenMerge'] = self.model.prompt_learner.AttenMerge
        # module_state_dict['token'] = param_state_list
        self.register_model("MultiModalPromptLearner", self.model, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.GRAPHDCPL.PREC == "amp" else None
        # Cosine scheduler
        self.total_epochs = cfg.OPTIM.MAX_EPOCH
        self.step_counter = 1

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)

        model = self.model
        optim = self.optim
        scaler = self.scaler

        prec = self.cfg.TRAINER.GRAPHDCPL.PREC
        if prec == "amp":
            with autocast():
                loss = model(image, label)
            optim.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
        else:
            loss = model(image, label)
            # self.model_backward_and_update(loss)
            optim.zero_grad()
            loss.backward()
            optim.step()

        loss_summary = {"loss": loss.item()}

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "prompt_learner.token_prefix" in state_dict:
                del state_dict["prompt_learner.token_prefix"]

            if "prompt_learner.token_suffix" in state_dict:
                del state_dict["prompt_learner.token_suffix"]

            # Ignore base features
            if "prompt_learner.base_text_features" in state_dict:
                del state_dict["prompt_learner.base_text_features"]

            if "prompt_learner.base_img_features" in state_dict:
                del state_dict["prompt_learner.base_img_features"]

            # if "prompt_learner.fc" in state_dict:
            #     del state_dict["prompt_learner.fc"]

            # Ignore base features
            if "prompt_learner.GCN_tt.gcn_bias" in state_dict:
                del state_dict["prompt_learner.GCN_tt.gcn_bias"]

            if "prompt_learner.GCN_it.gcn_bias" in state_dict:
                del state_dict["prompt_learner.GCN_it.gcn_bias"]

            if "prompt_learner.GCN_fuse.gcn_bias" in state_dict:
                del state_dict["prompt_learner.GCN_fuse.gcn_bias"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
