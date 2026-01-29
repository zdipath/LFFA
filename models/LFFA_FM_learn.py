import torch 
from torch import nn
import torch.nn.functional as F
from torch import Tensor, nn
import warnings
from environs import Env
from torch.nn.attention import SDPBackend, sdpa_kernel
import numpy as np
from timm.layers.helpers import to_2tuple
env = Env()
PERCEIVER_MEM_EFF_ATTN: bool = env.bool('PERCEIVER_MEM_EFF_ATTN', default=False)
if PERCEIVER_MEM_EFF_ATTN:
    warnings.warn('Perceiver: using memory-efficient attention')

try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    if PERCEIVER_MEM_EFF_ATTN:
        raise Exception(
            'Memory efficient attention flag is set (PERCEIVER_MEM_EFF_ATTN) '
            'but xformers lib is not available.'
        )
    pass







from torch.utils.data import Dataset, DataLoader


















class WSIDataset(Dataset):
    def __init__(self, features, labels,num_aug =None):
        """
        features: list of tensors, each of shape [1, N_i, 512]
        labels:  numpy array or list, length = num_samples
        """
        self.features = features
        self.labels = torch.as_tensor(labels)
        if num_aug is not None:
            self.num_aug = num_aug
        else:
            self.num_aug = None
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat = self.features[idx]
        label = self.labels[idx]         
        if self.num_aug is not None:
            num_augm = self.num_aug[idx]
        else:
            num_augm = -2
        return feat, label,num_augm 

def collate_wsi_batch(batch):

    feats, labels, num_augm = zip(*batch)
    B = len(feats)


    feats = [f.squeeze(0) for f in feats]


    targets = [f[0] for f in feats]    
    seqs    = [f[1:] for f in feats]   

    lengths = [s.shape[0] for s in seqs]
    max_len = max(lengths)
    dim = seqs[0].shape[1]


    if isinstance(seqs[0], torch.Tensor):
        data = seqs[0].new_zeros((B, max_len, dim))
    elif isinstance(seqs[0], np.ndarray):

        data = np.zeros((B, max_len, dim), dtype=seqs[0].dtype)
    attn_mask = torch.zeros((B, max_len), dtype=torch.bool)

    for i, (s, L) in enumerate(zip(seqs, lengths)):
        if L > 0:
            data[i, :L, :] = s
            attn_mask[i, :L] = True


    target = torch.stack(targets, dim=0) if isinstance(targets[0], torch.Tensor) else torch.from_numpy(np.stack(targets, axis=0))
    labels = torch.stack(labels, dim=0) if isinstance(labels[0], torch.Tensor) \
             else torch.as_tensor(labels)
    num_augm = torch.stack(num_augm, dim=0) if isinstance(num_augm[0], torch.Tensor) \
             else torch.as_tensor(num_augm)
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    return data, target, labels, attn_mask,num_augm

class MixupConsistencyLoss(nn.Module):
    def __init__(self, contrast_weight=0.1, temperature=0.07):
        super(MixupConsistencyLoss, self).__init__()
        self.contrast_weight = contrast_weight
        self.temperature = temperature

    def forward(self, aug_fea, ori_fea, lam, labels, aug_label_b):


        fea_a = ori_fea[aug_label_b[0]]
        fea_b = ori_fea[aug_label_b[1]]
        B_aug = aug_fea.size(0)
        B_ori = ori_fea.size(0)
        

        labels_ori = labels[:B_ori]       
        labels_aug_a = labels[B_ori:]     
        aug_label_bb = labels[aug_label_b[1]]

        aug_norm = F.normalize(aug_fea, dim=1)
        ori_norm = F.normalize(ori_fea, dim=1)
        



        mask_a = (labels_aug_a.unsqueeze(1) == labels_ori.unsqueeze(0)).float()

        sum_a = mask_a.sum(dim=1, keepdim=True)

        mask_b = (aug_label_bb.unsqueeze(1) == labels_ori.unsqueeze(0)).float()
        parta_global = False
        if parta_global:

            mask_a_norm = mask_a / (sum_a + 1e-8)
 
            feat_a_recon = torch.mm(mask_a_norm, ori_norm)
            

            
            sum_b = mask_b.sum(dim=1, keepdim=True)
            mask_b_norm = mask_b / (sum_b + 1e-8)

            feat_b_recon = torch.mm(mask_b_norm, ori_norm)
            

            lam_view = lam.view(B_aug, 1)
            virtual_target = lam_view * feat_a_recon + (1 - lam_view) * feat_b_recon
            
            virtual_target = F.normalize(virtual_target, dim=1)

            valid_mask = (sum_a > 0) & (sum_b > 0)
            valid_mask = valid_mask.float()
            


            loss_manifold_elements = 1 - (aug_norm * virtual_target).sum(dim=1, keepdim=True)
            loss_manifold = (loss_manifold_elements * valid_mask).sum() / (valid_mask.sum() + 1e-8)
        else:

            fea_a_norm = F.normalize(fea_a, dim=1)
            fea_b_norm = F.normalize(fea_b, dim=1)
            lam_view = lam.view(-1, 1)
            virtual_target = lam_view * fea_a_norm + (1.0 - lam_view) * fea_b_norm
            virtual_target = F.normalize(virtual_target, dim=1)

            loss_manifold = 1.0 - (aug_norm * virtual_target).sum(dim=1)
            loss_manifold = loss_manifold.mean()

        loss_contrast = torch.tensor(0.0, device=aug_fea.device)
        
        if self.contrast_weight > 0:

            logits = torch.mm(aug_norm, ori_norm.t()) / self.temperature
            

            is_parent_a = mask_a.bool()

            is_parent_b = mask_b.bool()
            

            is_negative = ~(is_parent_a | is_parent_b)
            

            if is_negative.sum() > 0:
                neg_logits = logits * is_negative.float()

                loss_contrast = (neg_logits ** 2).sum() / (is_negative.sum() + 1e-8)
        



        total_loss = (1-self.contrast_weight) * loss_manifold + self.contrast_weight * loss_contrast
        return total_loss











class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self._init_weight_()

    def forward(self, q,x,mask,delta, return_atten = False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = qkv.unbind(0)  
        v = delta.reshape(B, N,  self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q = self.q(q).reshape(B, 1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            attn = attn.masked_fill(~mask, -torch.finfo(attn.dtype).max)
        if return_atten:
            return attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, 1, C)

        return x

    def _init_weight_(self):
        nn.init.trunc_normal_(self.qkv.weight, std=.02)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        
        nn.init.trunc_normal_(self.proj.weight, std=.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x, **kwargs):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    def _init_weight_(self):
        nn.init.trunc_normal_(self.fc1.weight, std=.02)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        
        nn.init.trunc_normal_(self.fc2.weight, std=.02)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)

class MultiAttentionBlock(nn.Module):
    
    def __init__(
            self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., 
            act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn_layer=Attention):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, q,cls_fea, x,mask, delta, return_atten = False):
        if return_atten:
            return self.attn(self.norm1(q),self.norm1(x),mask,delta,return_atten = True)
        x = cls_fea + self.attn(self.norm1(q),self.norm1(x),mask,delta)

        x = x + self.mlp(self.norm2(x))
        return x






class GEGLU(nn.Module):
    def forward(self, x: Tensor):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)
class FeedForward(nn.Module):
    def __init__(
        self,
        *,
        dim: int,
        mult: int = 1,
        dropout: float = 0.0,
        activation: str = 'geglu',
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        extra_dim = 1

        if activation == 'geglu':
            actfn = GEGLU
            extra_dim = 2
        elif activation == 'gelu':
            actfn = nn.GELU
        else:
            raise Exception(f'{activation=} not supported.')

        self.fc1 = nn.Linear(dim, dim * mult * extra_dim)
        self.act = actfn()
        self.fc2 = nn.Linear(dim * mult, dim)
        self.dropout_p = dropout
    def forward(self, x: Tensor):
        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        if not self.training or self.dropout_p > 0:
            x = F.dropout(x, p=self.dropout_p, training=True)
        x = self.fc2(x)
        return x
class MHSA(nn.Module):
    def __init__(self, *, dim: int, num_heads: int):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=0.0,
            bias=False,
            add_bias_kv=False,
            add_zero_attn=False,
            kdim=dim,
            vdim=dim,
            batch_first=True,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x)

        kernels = [SDPBackend.MATH]
        if PERCEIVER_MEM_EFF_ATTN:
            kernels.append(SDPBackend.EFFICIENT_ATTENTION)
        with sdpa_kernel(kernels):
            x, _ = self.mha(
                x,
                x,
                x,
                key_padding_mask=None,
                need_weights=False,
                attn_mask=None,
                average_attn_weights=False,
                is_causal=False,
            )

        return x
class CrossAttention(nn.Module):
    def __init__(
        self,
        *,
        query_dim: int,
        context_dim: int,
        head_dim: int,
        heads: int,
        return_attn: bool = False,
        c_norm: bool = True,
        dropout: float=0.1
    ) -> None:
        super().__init__()
        self.attn_drop_p = dropout
        self.query_dim = query_dim
        self.context_dim = context_dim

        self.head_dim = head_dim
        self.heads = heads

        self.scale = self.head_dim**-0.5

        self.inner_dim = self.head_dim * self.heads

        self.return_attn = return_attn

        self.x_norm = nn.LayerNorm(self.query_dim)
        self.c_norm = nn.LayerNorm(self.context_dim) if c_norm is True else None

        self.to_q = nn.Linear(self.query_dim, self.inner_dim, bias=False)

        self.to_kv = nn.Linear(self.context_dim, self.inner_dim * 2, bias=False)

        self.to_out = nn.Linear(self.inner_dim, self.query_dim, bias=False)

    def forward(
        self,
        x: Tensor,
        c = None,
        kvt = None,
        attn_mask = None,
        return_attn = False
    ) -> tuple[Tensor, tuple[Tensor, Tensor], Tensor]:
        """
        Args:
            x: queries
            c: context
            kvt: key-value cache (instead of context)
            attn_mask: mask out part of context since contexts can vary in length

        Returns:
            processed output queries, KV-cache, attention weights
        """
        Bx, Nx, Dimx = x.shape

        x = self.x_norm(x)
        c = self.c_norm(c) if self.c_norm is not None else c

        q: Tensor = self.to_q(x)
        q = q.reshape(Bx, Nx, self.heads, self.head_dim)

        if c is not None and kvt is None:
            Bc, Nc, _ = c.shape
            kv: Tensor = self.to_kv(c)
            kv = kv.reshape(Bc, Nc, 2, self.heads, self.head_dim)
            k, v = kv.unbind(2)
            kvt = (k, v)
        elif kvt is not None and c is None:
            k, v = kvt
            Bc, Nc, _, _ = k.shape
            assert (Bc, Nc) == (v.shape[0], v.shape[1])
        else:
            raise Exception(f'XOR(c, kvt) but got: {type(c)} and {type(kvt)}.')

        if attn_mask is not None:
            attn_mask = attn_mask.reshape(Bc, 1, Nx, Nc).expand(-1, self.heads, -1, -1)

        if self.return_attn:
            warnings.warn('XATTN RETURNS ATTN SCORES, ONLY FOR EVAL!')
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            q = q * self.scale
            sim = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)
            attn = sim.softmax(dim=-1)
            a = attn @ v
            a = a.transpose(1, 2)

        elif PERCEIVER_MEM_EFF_ATTN:
            assert q.shape == (Bx, Nx, self.heads, Dimx // self.heads)
            assert k.shape == (Bx, Nc, self.heads, Dimx // self.heads)
            assert v.shape == (Bx, Nc, self.heads, Dimx // self.heads)
            if attn_mask is not None:
                attn_bias = torch.zeros_like(attn_mask, dtype=q.dtype, device=q.device)
                attn_bias = attn_bias.masked_fill(~attn_mask, -torch.finfo(q.dtype).max)
            else:
                attn_bias = None
            a = memory_efficient_attention(
                q,
                k,
                v,
                attn_bias=attn_bias,
                p=0.0,
                scale=None,
                op=None,
                output_dtype=None,
            )
            attn = torch.empty(0)

        else:
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)
            drop_p = self.attn_drop_p if self.training else 0.0
            if return_attn:
                qq = q * self.scale
                sim = qq @ k.transpose(-2, -1)
                if attn_mask is not None:
                    sim = sim.masked_fill(~attn_mask, -torch.finfo(sim.dtype).max)
                attn = sim.softmax(dim=-1).mean(dim=1).squeeze(1)
            else:
                attn = torch.empty(0)
            with sdpa_kernel(SDPBackend.MATH):
                a: Tensor = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask,dropout_p=drop_p,is_causal=False)

            a = a.transpose(1, 2)

        c = a.reshape(Bx, Nx, self.inner_dim)

        o = self.to_out(c)

        return o, kvt, attn
    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])
        self.norm = nn.LayerNorm(in_features)
    def forward(self, x, **kwargs):
        x = self.fc1(self.norm(x))
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

    def _init_weight_(self):
        nn.init.trunc_normal_(self.fc1.weight, std=.02)
        if self.fc1.bias is not None:
            nn.init.zeros_(self.fc1.bias)
        
        nn.init.trunc_normal_(self.fc2.weight, std=.02)
        if self.fc2.bias is not None:
            nn.init.zeros_(self.fc2.bias)
class Projector(nn.Module):
    def __init__(self, input_dim = 768, hidden_dim = 2048, output_dim = 128, drop=0.):
        super().__init__()
        drop_probs = to_2tuple(drop)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.drop1 = nn.Dropout(drop_probs[0])
        self.drop2 = nn.Dropout(drop_probs[1])
    def forward(self, x):
        h = self.fc1(x)
        h = self.relu(h)
        h = self.drop1(h)
        output = self.fc2(h)

        return output

class AdaLN(nn.Module):
    """
    Adaptive LayerNorm parameter generator.
    Input:  inj  shape (B, D) or (B, 1, D)
    Output: (delta_gamma, delta_beta) concatenated, shape (B, 2D) or (B, 1, 2D)
    """
    def __init__(self, dim: int, hidden_dim: int = None, act: str = "silu", zero_init: bool = True):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = dim * 2

        if act.lower() == "gelu":
            activation = nn.GELU()
        elif act.lower() in ("silu", "swish"):
            activation = nn.SiLU()
        else:
            raise ValueError(f"Unsupported act: {act}. Use 'gelu' or 'silu'.")
       
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 2 * dim),
        )
        '''
        self.net = nn.Parameter(torch.randn(2, int(hidden_dim//2)) / int(hidden_dim//2)**0.5) '''

        if zero_init:
            last = self.net[-1]
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)



    def forward(self, inj: torch.Tensor) -> torch.Tensor:

        y = self.net(inj)


        return y

class LFFA(nn.Module):
    def __init__(self, input_dim=1024, mhsa_heads = 8,mlp_mult = 1, num_classifier = 2,dropout = 0.2,
                depth = 1, mlp_activation = "geglu", aug_para = 1, aug_method = 'brightness', class_way = 'linear',dim1 = 512,dim2 = 128):
        super().__init__()

        self.norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        self.aug_para = aug_para
        self.aug_method = aug_method
        self.num_class = num_classifier
        self.cls_mlp = Mlp(in_features=input_dim, hidden_features=int(input_dim ), act_layer=nn.GELU, drop=dropout)
        self.projector = Projector(input_dim, dim1, dim2, drop=dropout)


        self.inj_gain = nn.Parameter(torch.ones(1, 1, input_dim)) 

        self.class_way = class_way
        if self.class_way == 'fewshot':
            
            self.temperature = 1

            self.weight = nn.Parameter(
                torch.randn(num_classifier*2, dim2)
            )
            '''
            self.temperature = 1
            self.weight = nn.ModuleList()

            self.weights = nn.ParameterList()
            for k in range(2):
                w = nn.Parameter(torch.empty(num_classifier+1, dim2))
                nn.init.orthogonal_(w)
                eps = 1e-3
                w.data += eps * torch.randn_like(w)
                self.weights.append(w)
            '''
        else:
            self.head = nn.Linear(dim2, num_classifier*2)

            


        self.mab = MultiAttentionBlock(dim=input_dim, num_heads=8)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Dropout(p=0.2),
            nn.Linear(input_dim, input_dim),

        )
        self.gate = nn.Sequential(nn.Linear(input_dim, 1), nn.Sigmoid())


    def get_attention(self,x,cls_fea,weight_target):
        context = x
        cls_fea = cls_fea.unsqueeze(0)
        label = torch.ones(1).to(torch.long).to(x.device)
        aug_exp = (torch.ones(1)).to(torch.long).to(x.device)
        attn_mask = None
        infer_time = 0
        fea, labels, owner_idx, attn_aug, ori_patch_fea = self.frozen_aug(x,cls_fea,label,aug_exp,attn_mask,infer_time)
        fea = fea[1:]
        ori_patch_fea = ori_patch_fea[1:]
        c = F.normalize(cls_fea, dim=-1)
        s = F.normalize(fea, dim=-1)

        sim = (s * c).sum(dim=-1)
        w = F.softmax(sim *10, dim=-1).unsqueeze(-1)
        if weight_target == 'LFFA_sim':
            return w.squeeze(-1).squeeze()
        delta = fea - ori_patch_fea
        inj = (delta * w).sum(dim=1, keepdim=True)
        x = cls_fea+ inj * self.inj_gain
        B = x.shape[0]
        
        if attn_mask is None:
            if B > 1:
                raise Exception('tile pad mask must be provided with batch size>1.')

            attn_mask = torch.ones(
                    context.shape[:2], device=context.device, dtype=torch.bool
                )
            
        atten = self.mab(x,cls_fea,fea,attn_mask.unsqueeze(1),delta,return_atten = True)
        atten = torch.mean(atten,dim= 1).squeeze().squeeze()
        return atten

    def learn_aug(self, x,cls_fea,label,infer_time,attn_mask,aug_exp):

        B = x.size(0)
        if infer_time >0:
            aug_exp = (torch.ones(B)*int(infer_time)).to(torch.long).to(x.device)
        rep = aug_exp.clamp(min=0).to(torch.long)
        device = x.device
        total_aug = int(rep.sum().item())
        if total_aug == 0:

            aug_features = None
            aug_label = None
            owner_idx = None
        else:


            owner_idx = torch.repeat_interleave(
                torch.arange(B, device=device),
                repeats=rep
            )

            if attn_mask is None:
                if B > 1:
                    raise Exception('tile pad mask must be provided with batch size>1.')

                attn_mask = torch.ones(
                        x.shape[:2], device=x.device, dtype=torch.bool
                    )
            mask = attn_mask.unsqueeze(-1)
            fea_masked = x * mask
            sum_fea = fea_masked.sum(dim=1)
            count = mask.sum(dim=1)
            x_avg = (sum_fea / count.clamp(min=1)).unsqueeze(1)
            diff = x - cls_fea

            diff_sel = diff[owner_idx]
            x_sel = x[owner_idx]
            M,N,d = x_sel.shape


            delta = self.mlp(x_sel)
            g = self.gate(x_sel)
            if self.training:
                mask = (torch.rand(M, N, 1, device=x_sel.device) < 0.8).float()
            else:
                mask = (torch.rand(M, N, 1, device=x_sel.device) < 0.8).float()
            g = g * mask
            aug_features = x_sel + g * delta
            aug_label = label[owner_idx] 
            ori_fea = torch.cat([x, x_sel], dim=0)
        return aug_features, aug_label, owner_idx, owner_idx, ori_fea


    def frozen_aug(self, context,cls_fea,label,aug_exp,attn_mask,infer_time):
        if aug_exp is None and infer_time == 0:
            return context,label,None, None,context
        if self.aug_method not in ['tokenmix','mixup']:
            aug_label_b = None
            lam_eff = None
            replace = None

        ori_fea = context
        aug_fea, aug_label, owner_idx,replace,ori_patch_fea = self.learn_aug(ori_fea,cls_fea.unsqueeze(1),label,infer_time,attn_mask,aug_exp = aug_exp)
        if aug_fea is None:
            fea = context 
            labels = label
        else:
            fea = torch.cat([ori_fea, aug_fea], dim=0)
            labels = torch.cat([label, aug_label], dim=0)


        if self.aug_method not in ['onemask']:
            attn_aug = None

        return fea,labels,owner_idx,attn_aug,ori_patch_fea

    
    def forward(self, context,cls_fea,label,aug_exp = None, attn_mask = None,return_aug = False, infer_time = 0):


        fea, labels, owner_idx, attn_aug, ori_patch_fea = self.frozen_aug(context,cls_fea,label,aug_exp,attn_mask,infer_time)

        if owner_idx is not None:
            cls_fea = torch.cat([cls_fea,cls_fea[owner_idx]],dim = 0)
        else:
            ...
        B = context.shape[0]
        B_aug = fea.shape[0]
        if attn_mask is None:
            if B > 1:
                raise Exception('tile pad mask must be provided with batch size>1.')
            if owner_idx is None:
                attn_mask = torch.ones(
                    context.shape[:2], device=context.device, dtype=torch.bool
                )
            else:
                attn_mask = torch.ones(
                    context.shape[:2], device=context.device, dtype=torch.bool
                )
                if attn_aug is None:
                    aug_attn_mask = attn_mask[owner_idx]
                else:
                    aug_attn_mask = attn_aug
                attn_mask = torch.cat([attn_mask, aug_attn_mask], dim=0)
        else:
            if owner_idx is not None:
                if attn_aug is None:
                    aug_attn_mask = attn_mask[owner_idx]
                else:
                    aug_attn_mask = attn_aug
                attn_mask = torch.cat([attn_mask, aug_attn_mask], dim=0)
        attn_mask2 = attn_mask






        '''
        cls_mask = torch.ones(B_aug, 1, device=cls_fea.device, dtype=torch.bool)
        attn_mask2 = attn_mask
        mask3 = attn_mask2.unsqueeze(1)
        all_fea =  fea - ori_patch_fea
        x = self.mab(cls_fea.unsqueeze(1),all_fea,mask3)[:, 0, :]'''

        ''''''
        mask3 = attn_mask2
        cls_fea = cls_fea.unsqueeze(1)
        delta = fea - ori_patch_fea
        c = F.normalize(cls_fea, dim=-1)
        s = F.normalize(fea, dim=-1)

        sim = (s * c).sum(dim=-1)

        sim_masked = sim.masked_fill(~mask3, -1e4)
        w = F.softmax(sim_masked *10, dim=-1).unsqueeze(-1)
        inj = (delta * w).sum(dim=1, keepdim=True)


















        x = cls_fea+ inj * self.inj_gain




        x = self.mab(x,cls_fea,fea,attn_mask.unsqueeze(1),delta)[:, 0, :]


        x = self.norm(x)
        x = self.projector(x)
        if self.class_way == 'fewshot':
            
            x = F.normalize(x.squeeze(1), p=2, dim=-1)
            w = F.normalize(self.weight, p=2, dim=-1)
            logits = (F.linear(x, w) / self.temperature).unsqueeze(1)
            '''
            x = F.normalize(x.squeeze(1), p=2, dim=-1)
            ws = [F.normalize(head, p=2, dim=-1) for head in self.weights]
            logits = [(F.linear(x, w) / self.temperature).unsqueeze(1) for w in ws]
            logits = torch.stack(logits, dim=0)
            logits = logits.mean(dim=0)
            '''
        else:
            logits = self.head(x)

        return logits,labels
    
