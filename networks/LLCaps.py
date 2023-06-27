import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from networks.ReverseDiffusion import Unet, GaussianDiffusion
from utils.antialias import Downsample as downsamp
from networks.wavelet import DWT, IWT

##########################################################################

def conv(in_channels, out_channels, kernel_size, bias=False, padding = 1, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

##########################################################################
##---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3,reduction=8,bias=False):
        super(SKFF, self).__init__()
        
        self.height = height
        d = max(int(in_channels/reduction),4)
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))
        
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]
        

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])
        
        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        
        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)
        
        return feats_V        


##########################################################################
# Spatial Attention Layer
class SALayer(nn.Module):
    def __init__(self, kernel_size=5, bias=False):
        super(SALayer, self).__init__()
        self.conv_du = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        # torch.max will output 2 things, and we want the 1st one
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        avg_pool = torch.mean(x, 1, keepdim=True)
        channel_pool = torch.cat([max_pool, avg_pool], dim=1)  # [N,2,H,W]  could add 1x1 conv -> [N,3,H,W]
        y = self.conv_du(channel_pool)

        return x * y

##########################################################################
# Channel Attention Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16, bias=False):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

##########################################################################
## Curved Attention Layer
class CurveCALayer(nn.Module):
    def __init__(self, channel):
        super(CurveCALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.n_curve = 3
        self.relu = nn.ReLU(inplace=False)
        self.predict_a = nn.Sequential(
            nn.Conv2d(channel, channel, 5, stride=1, padding=2),nn.ReLU(inplace=True),
            nn.Conv2d(channel, channel, 3, stride=1, padding=1),nn.ReLU(inplace=True),
            nn.Conv2d(channel, 3, 1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        a = self.predict_a(x) 
        x = self.relu(x) - self.relu(x-1)
        for i in range(self.n_curve):
            x = x + a[:,i:i+1]*x*(1-x)
        return x

##########################################################################
##---------- Curved Wavelet Attention (CWA) Blocks ----------    
class CWA(nn.Module):
        def __init__(self, n_feat=64, kernel_size=3, reduction=16, bias=False, act=nn.PReLU()):
            super(CWA, self).__init__()
            self.dwt = DWT()
            self.iwt = IWT()
            modules_body = \
                [
                    conv(n_feat*2, n_feat, kernel_size, bias=bias),
                    act,
                    conv(n_feat, n_feat*2, kernel_size, bias=bias)
                ]
            self.body = nn.Sequential(*modules_body)

            self.WSA = SALayer()
            self.CurCA = CurveCALayer(n_feat*2)

            self.conv1x1 = nn.Conv2d(n_feat*4, n_feat*2, kernel_size=1, bias=bias)  #256 to 128
            self.conv3x3 = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=bias)
            self.activate = act
            self.conv1x1_final = nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)

        def forward(self, x):
            residual = x
            wavelet_path_in, identity_path = torch.chunk(x, 2, dim=1)

            # Wavelet domain (Dual attention)
            x_dwt = self.dwt(wavelet_path_in) 

            res = self.body(x_dwt)      
            branch_sa = self.WSA(res)
            branch_curveca_2 = self.CurCA(res)
            res = torch.cat([branch_sa, branch_curveca_2], dim=1)
            res = self.conv1x1(res) + x_dwt                 
            
            wavelet_path = self.iwt(res)
            out = torch.cat([wavelet_path, identity_path], dim=1)
            out = self.activate(self.conv3x3(out))
            out += self.conv1x1_final(residual)
            
            return out

##########################################################################
##---------- Resizing Modules ----------    
class ResidualDownSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualDownSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels,   1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=bias),
                                nn.PReLU(),
                                downsamp(channels=in_channels,filt_size=3,stride=2),
                                nn.Conv2d(in_channels, in_channels*2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(downsamp(channels=in_channels,filt_size=3,stride=2),
                                nn.Conv2d(in_channels, in_channels*2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top+bot
        return out

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualDownSample(in_channels))
            in_channels = int(in_channels * stride)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class ResidualUpSample(nn.Module):
    def __init__(self, in_channels, bias=False):
        super(ResidualUpSample, self).__init__()

        self.top = nn.Sequential(nn.Conv2d(in_channels, in_channels,   1, stride=1, padding=0, bias=bias),
                                nn.PReLU(),
                                nn.ConvTranspose2d(in_channels, in_channels, 3, stride=2, padding=1, output_padding=1,bias=bias),
                                nn.PReLU(),
                                nn.Conv2d(in_channels, in_channels//2, 1, stride=1, padding=0, bias=bias))

        self.bot = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias),
                                nn.Conv2d(in_channels, in_channels//2, 1, stride=1, padding=0, bias=bias))

    def forward(self, x):
        top = self.top(x)
        bot = self.bot(x)
        out = top+bot
        return out

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, stride=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(ResidualUpSample(in_channels))
            in_channels = int(in_channels // stride)
        
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x


##########################################################################
##---------- Multi-Scale Resiudal Block (MSRB) ----------
class MSRB(nn.Module):
    def __init__(self, n_feat, height, width, stride, bias):
        super(MSRB, self).__init__()

        self.n_feat, self.height, self.width = n_feat, height, width
        self.blocks = nn.ModuleList([nn.ModuleList([CWA(int(n_feat*stride**i))]*width) for i in range(height)])
        INDEX = np.arange(0,width, 2)
        FEATS = [int((stride**i)*n_feat) for i in range(height)]
        SCALE = [2**i for i in range(1,height)]

        self.last_up   = nn.ModuleDict()
        for i in range(1,height):
            self.last_up.update({f'{i}': UpSample(int(n_feat*stride**i),2**i,stride)})

        self.down = nn.ModuleDict()
        self.up   = nn.ModuleDict()

        i=0
        SCALE.reverse()
        for feat in FEATS:
            for scale in SCALE[i:]:
                self.down.update({f'{feat}_{scale}': DownSample(feat,scale,stride)})
            i+=1

        i=0
        FEATS.reverse()
        for feat in FEATS:
            for scale in SCALE[i:]:                
                self.up.update({f'{feat}_{scale}': UpSample(feat,scale,stride)})
            i+=1

        self.conv_out = nn.Conv2d(n_feat, n_feat, kernel_size=3, padding=1, bias=bias)

        self.selective_kernel = nn.ModuleList([SKFF(n_feat*stride**i, height) for i in range(height)])
        
    def forward(self, x):
        inp = x.clone()
        #col 1 only
        blocks_out = []
        for j in range(self.height):
            if j==0:
                inp = self.blocks[j][0](inp)
            else:
                inp = self.blocks[j][0](self.down[f'{inp.size(1)}_{2}'](inp))
            blocks_out.append(inp)
        
        #rest of grid
        for i in range(1,self.width):
            if True:
                tmp=[]
                for j in range(self.height):
                    TENSOR = []
                    nfeats = (2**j)*self.n_feat
                    for k in range(self.height):
                        TENSOR.append(self.select_up_down(blocks_out[k], j, k)) 

                    selective_kernel_fusion = self.selective_kernel[j](TENSOR)
                    tmp.append(selective_kernel_fusion)
            #Plain
            else:
                tmp = blocks_out
            #Forward through either mesh or plain
            for j in range(self.height):
                blocks_out[j] = self.blocks[j][i](tmp[j])

        #Sum after grid
        out=[]
        for k in range(self.height):
            out.append(self.select_last_up(blocks_out[k], k))  

        out = self.selective_kernel[0](out)

        out = self.conv_out(out)
        out = out + x

        return out

    def select_up_down(self, tensor, j, k):
        if j==k:
            return tensor
        else:
            diff = 2 ** np.abs(j-k)
            if j<k:
                return self.up[f'{tensor.size(1)}_{diff}'](tensor)
            else:
                return self.down[f'{tensor.size(1)}_{diff}'](tensor)


    def select_last_up(self, tensor, k):
        if k==0:
            return tensor
        else:
            return self.last_up[f'{k}'](tensor)


##########################################################################
##---------- Recursive Residual Group (RRG) ----------
class RRG(nn.Module):
    def __init__(self, n_feat, n_MSRB, height, width, stride, bias=False):
        super(RRG, self).__init__()
        modules_body = [MSRB(n_feat, height, width, stride, bias) for _ in range(n_MSRB)]
        modules_body.append(conv(n_feat, n_feat, kernel_size=3))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

class LLCaps(nn.Module):
    def __init__(self,device, in_channels=3, out_channels=3, n_feat=64, kernel_size=3, stride=2, n_RRG=3, n_MSRB=2, height=3, width=2, bias=False):
        super(LLCaps, self).__init__()
        self.device = device
        self.conv_in = nn.Conv2d(in_channels, n_feat, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=bias)
        modules_body = [RRG(n_feat, n_MSRB, height, width, stride, bias) for _ in range(n_RRG)]
        self.body = nn.Sequential(*modules_body)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=bias)
        self.rd_model = Unet(dim=6, init_dim=None, out_dim=None, dim_mults=(1,2,4,8), channels=3, self_condition=False, resnet_block_groups=6, learned_variance=False, learned_sinusoidal_cond=False, random_fourier_features=False, learned_sinusoidal_dim=16)
        self.rd_procedure = GaussianDiffusion(self.rd_model, image_size=256, timesteps=1000, sampling_timesteps=None, loss_type='l1',objective='pred_noise', beta_schedule='sigmoid', schedule_fn_kwargs=dict(), ddim_sampling_eta=0., auto_normalize = True)

    def forward(self, x):
        h = self.conv_in(x)
        h = self.body(h)
        h = self.conv_out(h)
        h += x
        h = self.rd_procedure(h)
        return h
