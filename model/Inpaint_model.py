import torch
from torch import nn
import torch.nn.init as init
from torch.nn import functional as F
from torch.nn import Parameter
from utilz.inp_util import *
import torchvision
import pytorch_lightning as pl

# -----------------------------------------------
#                Normal ConvBlock
# -----------------------------------------------
#https://github.com/csqiangwen/DeepFillv2_Pytorch
class Conv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
                 activation='elu', norm='none', sn=False):
        super(Conv2dLayer, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)

    def forward(self, x):
        x = self.pad(x)
        x = self.conv2d(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class TransposeConv2dLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
                 activation='lrelu', norm='none', sn=False, scale_factor=2):
        super(TransposeConv2dLayer, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.conv2d = Conv2dLayer(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type,
                                  activation, norm, sn)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.conv2d(x)
        return x


# -----------------------------------------------
#                Gated ConvBlock
# -----------------------------------------------
class GatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='reflect',
                 activation='elu', norm='none', sn=False):
        super(GatedConv2d, self).__init__()
        # Initialize the padding scheme
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # Initialize the normalization type
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'ln':
            self.norm = LayerNorm(out_channels)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # Initialize the activation funtion
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # Initialize the convolution layers
        if sn:
            self.conv2d = SpectralNorm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
            self.mask_conv2d = SpectralNorm(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation))
        else:
            self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
            self.mask_conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=0, dilation=dilation)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.pad(x)
        conv = self.conv2d(x)
        mask = self.mask_conv2d(x)
        gated_mask = self.sigmoid(mask)
        if self.activation:
            conv = self.activation(conv)
        x = conv * gated_mask
        return x


class TransposeGatedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, pad_type='zero',
                 activation='lrelu', norm='none', sn=True, scale_factor=2):
        super(TransposeGatedConv2d, self).__init__()
        # Initialize the conv scheme
        self.scale_factor = scale_factor
        self.gated_conv2d = GatedConv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, pad_type,
                                        activation, norm, sn)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        x = self.gated_conv2d(x)
        return x


# ----------------------------------------
#               Layer Norm
# ----------------------------------------
class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-8, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = Parameter(torch.Tensor(num_features).uniform_())
            self.beta = Parameter(torch.zeros(num_features))

    def forward(self, x):
        # layer norm
        shape = [-1] + [1] * (x.dim() - 1)  # for 4d input: [-1, 1, 1, 1]
        if x.size(0) == 1:
            # These two lines run much faster in pytorch 0.4 than the two lines listed below.
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)
        x = (x - mean) / (std + self.eps)
        # if it is learnable
        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)  # for 4d input: [1, -1, 1, 1]
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x


# -----------------------------------------------
#                  SpectralNorm
# -----------------------------------------------
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class ContextualAttention(nn.Module):
    def __init__(self, ksize=3, stride=1, rate=1, fuse_k=3, softmax_scale=10,
                 fuse=True, use_cuda=True, device_ids=None):
        super(ContextualAttention, self).__init__()
        self.ksize = ksize
        self.stride = stride
        self.rate = rate
        self.fuse_k = fuse_k
        self.softmax_scale = softmax_scale
        self.fuse = fuse
        self.use_cuda = use_cuda
        self.device_ids = device_ids

    def forward(self, f, b, mask=None):
        """ Contextual attention layer implementation.
            Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            f: Input feature to match (foreground).
            b: Input feature for match (background).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
        Returns:
            torch.tensor: output
        """
        # get shapes
        raw_int_fs = list(f.size())  # b*c*h*w
        raw_int_bs = list(b.size())  # b*c*h*w

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # raw_w is extracted for reconstruction
        raw_w = extract_image_patches(b, ksizes=[kernel, kernel],
                                      strides=[self.rate * self.stride,
                                               self.rate * self.stride],
                                      rates=[1, 1],
                                      padding='same')  # [N, C*k*k, L]
        # raw_shape: [N, C, k, k, L] [4, 192, 4, 4, 1024]
        raw_w = raw_w.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        raw_w = raw_w.permute(0, 4, 1, 2, 3)  # raw_shape: [N, L, C, k, k]
        raw_w_groups = torch.split(raw_w, 1, dim=0)

        # downscaling foreground option: downscaling both foreground and
        # background for matching and use original background for reconstruction.
        f = F.interpolate(f, scale_factor=1. / self.rate, mode='nearest')
        b = F.interpolate(b, scale_factor=1. / self.rate, mode='nearest')
        int_fs = list(f.size())  # b*c*h*w
        int_bs = list(b.size())
        f_groups = torch.split(f, 1, dim=0)  # split tensors along the batch dimension
        # w shape: [N, C*k*k, L]
        w = extract_image_patches(b, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')
        # w shape: [N, C, k, k, L]
        w = w.view(int_bs[0], int_bs[1], self.ksize, self.ksize, -1)
        w = w.permute(0, 4, 1, 2, 3)  # w shape: [N, L, C, k, k]
        w_groups = torch.split(w, 1, dim=0)

        # process mask
        mask = F.interpolate(mask, scale_factor=1. / self.rate, mode='nearest')
        int_ms = list(mask.size())
        # m shape: [N, C*k*k, L]
        m = extract_image_patches(mask, ksizes=[self.ksize, self.ksize],
                                  strides=[self.stride, self.stride],
                                  rates=[1, 1],
                                  padding='same')

        # m shape: [N, C, k, k, L]
        m = m.view(int_ms[0], int_ms[1], self.ksize, self.ksize, -1)
        m = m.permute(0, 4, 1, 2, 3)  # m shape: [N, L, C, k, k]
        m = m[0]  # m shape: [L, C, k, k]
        # mm shape: [L, 1, 1, 1]
        mm = (reduce_mean(m, axis=[1, 2, 3], keepdim=True) == 0.).to(torch.float32)
        mm = mm.permute(1, 0, 2, 3)  # mm shape: [1, L, 1, 1]

        y = []
        offsets = []
        k = self.fuse_k
        scale = self.softmax_scale  # to fit the PyTorch tensor image value range
        fuse_weight = torch.eye(k).view(1, 1, k, k)  # 1*1*k*k
        if self.use_cuda:
            fuse_weight = fuse_weight.cuda()

        for xi, wi, raw_wi in zip(f_groups, w_groups, raw_w_groups):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            raw_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            escape_NaN = torch.FloatTensor([1e-4])
            if self.use_cuda:
                escape_NaN = escape_NaN.cuda()
            wi = wi[0]  # [L, C, k, k]
            max_wi = torch.sqrt(reduce_sum(torch.pow(wi, 2) + escape_NaN, axis=[1, 2, 3], keepdim=True))
            wi_normed = wi / max_wi
            # xi shape: [1, C, H, W], yi shape: [1, L, H, W]
            xi = same_padding(xi, [self.ksize, self.ksize], [1, 1], [1, 1])  # xi: 1*c*H*W
            yi = F.conv2d(xi, wi_normed, stride=1)  # [1, L, H, W]
            # conv implementation for fuse scores to encourage large patches
            if self.fuse:
                # make all of depth to spatial resolution
                yi = yi.view(1, 1, int_bs[2] * int_bs[3], int_fs[2] * int_fs[3])  # (B=1, I=1, H=32*32, W=32*32)
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)  # (B=1, C=1, H=32*32, W=32*32)
                yi = yi.contiguous().view(1, int_bs[2], int_bs[3], int_fs[2], int_fs[3])  # (B=1, 32, 32, 32, 32)
                yi = yi.permute(0, 2, 1, 4, 3)
                yi = yi.contiguous().view(1, 1, int_bs[2] * int_bs[3], int_fs[2] * int_fs[3])
                yi = same_padding(yi, [k, k], [1, 1], [1, 1])
                yi = F.conv2d(yi, fuse_weight, stride=1)
                yi = yi.contiguous().view(1, int_bs[3], int_bs[2], int_fs[3], int_fs[2])
                yi = yi.permute(0, 2, 1, 4, 3).contiguous()
            yi = yi.view(1, int_bs[2] * int_bs[3], int_fs[2], int_fs[3])  # (B=1, C=32*32, H=32, W=32)
            # softmax to match
            yi = yi * mm
            yi = F.softmax(yi * scale, dim=1)
            yi = yi * mm  # [1, L, H, W]

            offset = torch.argmax(yi, dim=1, keepdim=True)  # 1*1*H*W

            if int_bs != int_fs:
                # Normalize the offset value to match foreground dimension
                times = float(int_fs[2] * int_fs[3]) / float(int_bs[2] * int_bs[3])
                offset = ((offset + 1).float() * times - 1).to(torch.int64)
            offset = torch.cat([offset // int_fs[3], offset % int_fs[3]], dim=1)  # 1*2*H*W

            # deconv for patch pasting
            wi_center = raw_wi[0]
            # yi = F.pad(yi, [0, 1, 0, 1])    # here may need conv_transpose same padding
            yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_int_fs)

        return y


def weights_init(net, init_type='kaiming', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)       -- network to be initialized
        init_type (str)     -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_var (float)    -- scaling factor for normal, xavier and orthogonal.
    """

    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, 0.02)
            init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            init.normal_(m.weight, 0, 0.01)
            init.constant_(m.bias, 0)

    # Apply the initialization function <init_func>
    net.apply(init_func)


# -----------------------------------------------
#                   Generator
# -----------------------------------------------
# Input: masked image + mask
# Output: filled image
class GatedGenerator(pl.LightningModule):

    def __init__(self, latent_channels = 64):
        super(GatedGenerator, self).__init__()
        in_channels = 4
        out_channels = 3
        #latent_channels = 48
        activation='elu'
        norm='none'
        pad_type='zero'
        self.latent_channels = latent_channels
        self.norm = norm
        self.pad_type = pad_type
        self.activation = activation
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.init_conv_layer = GatedConv2d(in_channels, latent_channels, 5, 1, 2, pad_type=pad_type, activation=activation,
                        norm=norm)
        self.mid_conv_layer = GatedConv2d(latent_channels, latent_channels * 2, 3, 2, 1, pad_type=pad_type,
                        activation=activation, norm=norm)

        self.coarse = nn.Sequential(
            # encoder
            GatedConv2d(latent_channels * 2, latent_channels * 2, 3, 1, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels * 2, latent_channels * 4, 3, 2, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            # Bottleneck
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation=2, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 4, dilation=4, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 8, dilation=8, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 16, dilation=16, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            # decoder
            TransposeGatedConv2d(latent_channels * 4, latent_channels * 2, 3, 1, 1, pad_type=pad_type,
                                 activation=activation, norm=norm),
            GatedConv2d(latent_channels * 2, latent_channels * 2, 3, 1, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            TransposeGatedConv2d(latent_channels * 2, latent_channels, 3, 1, 1, pad_type=pad_type,
                                 activation=activation, norm=norm),


        )
        self.mid_deconv_layer = GatedConv2d(latent_channels, latent_channels // 2, 3, 1, 1, pad_type=pad_type,
                        activation=activation, norm=norm)
        self.final_deconv_layer = nn.Sequential(GatedConv2d(latent_channels // 2, out_channels, 3, 1, 1, pad_type=pad_type, activation='none',
                        norm=norm),
            nn.Tanh())

        self.refine_conv = nn.Sequential(
            GatedConv2d(in_channels, latent_channels, 5, 1, 2, pad_type=pad_type, activation=activation,
                        norm=norm),
            GatedConv2d(latent_channels, latent_channels, 3, 2, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels, latent_channels * 2, 3, 1, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels * 2, latent_channels * 2, 3, 2, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels * 2, latent_channels * 4, 3, 1, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 2, dilation=2, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 4, dilation=4, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 8, dilation=8, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 16, dilation=16, pad_type=pad_type,
                        activation=activation, norm=norm)
        )
        self.refine_atten_1 = nn.Sequential(
            GatedConv2d(in_channels, latent_channels, 5, 1, 2, pad_type=pad_type, activation=activation,
                        norm=norm),
            GatedConv2d(latent_channels, latent_channels, 3, 2, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels, latent_channels * 2, 3, 1, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels * 2, latent_channels * 4, 3, 2, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type=pad_type,
                        activation='relu', norm=norm)
        )
        self.refine_atten_2 = nn.Sequential(
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type=pad_type,
                        activation=activation, norm=norm)
        )
        self.refine_combine = nn.Sequential(
            GatedConv2d(latent_channels * 8, latent_channels * 4, 3, 1, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels * 4, latent_channels * 4, 3, 1, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            TransposeGatedConv2d(latent_channels * 4, latent_channels * 2, 3, 1, 1, pad_type=pad_type,
                                 activation=activation, norm=norm),
            GatedConv2d(latent_channels * 2, latent_channels * 2, 3, 1, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            TransposeGatedConv2d(latent_channels * 2, latent_channels, 3, 1, 1, pad_type=pad_type,
                                 activation=activation, norm=norm),
            GatedConv2d(latent_channels, latent_channels // 2, 3, 1, 1, pad_type=pad_type,
                        activation=activation, norm=norm),
            GatedConv2d(latent_channels // 2, out_channels, 3, 1, 1, pad_type=pad_type, activation='none',
                        norm=norm),
            nn.Tanh()
        )
        self.context_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
                                                     fuse=True)

    def forward(self, img, mask):
        # img: entire img
        # mask: 1 for mask region; 0 for unmask region
        # Coarse
        first_masked_img = img * (1 - mask) + mask
        first_in = torch.cat((first_masked_img, mask), dim=1)  # in: [B, 4, H, W]
        first_in = self.init_conv_layer(first_in)
        first_in = self.mid_conv_layer(first_in)
        first_out = self.coarse(first_in)  # out: [B, 3, H, W]
        first_out = self.mid_deconv_layer(first_out)
        first_out = self.final_deconv_layer(first_out)
        first_out = nn.functional.interpolate(first_out, (img.shape[2], img.shape[3]))
        '''# Refinement
        second_masked_img = img * (1 - mask) + first_out * mask
        second_in = torch.cat([second_masked_img, mask], dim=1)
        refine_conv = self.refine_conv(second_in)
        refine_atten = self.refine_atten_1(second_in)
        mask_s = nn.functional.interpolate(mask, (refine_atten.shape[2], refine_atten.shape[3]))
        refine_atten = self.context_attention(refine_atten, refine_atten, mask_s)
        refine_atten = self.refine_atten_2(refine_atten)
        second_out = torch.cat([refine_conv, refine_atten], dim=1)
        second_out = self.refine_combine(second_out)
        second_out = nn.functional.interpolate(second_out, (img.shape[2], img.shape[3]))'''
        return first_out, first_out

    def re_init_conv(self):
        self.init_conv_layer = GatedConv2d(self.in_channels, self.latent_channels, 5, 1, 2, pad_type=self.pad_type,
                                           activation=self.activation,
                                           norm=self.norm)

    def re_init_mid_conv(self):
        self.mid_conv_layer = GatedConv2d(self.latent_channels, self.latent_channels * 2, 3, 2, 1,
                                          pad_type=self.pad_type,
                                          activation=self.activation, norm=self.norm)

    def re_init_mid_deconv(self):
        self.mid_deconv_layer = GatedConv2d(self.latent_channels, self.latent_channels // 2, 3, 1, 1,
                                            pad_type=self.pad_type,
                                            activation=self.activation, norm=self.norm)

    def re_init_final_deconv(self):
        self.final_deconv_layer = nn.Sequential(
            GatedConv2d(self.latent_channels // 2, self.out_channels, 3, 1, 1, pad_type=self.pad_type,
                        activation='none',
                        norm=self.norm),
            nn.Tanh())


# ----------------------------------------
#            Perceptual Network
# ----------------------------------------
# VGG-16 conv4_3 features
class PerceptualNet(nn.Module):
    def __init__(self):
        super(PerceptualNet, self).__init__()
        block = [torchvision.models.vgg16(pretrained=True).features[:15].eval()]
        for p in block[0]:
            p.requires_grad = False
        self.block = torch.nn.ModuleList(block)
        self.transform = torch.nn.functional.interpolate
        self.register_buffer('mean', torch.FloatTensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.FloatTensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        x = self.transform(x, mode='bilinear', size=(224, 224), align_corners=False)
        for block in self.block:
            x = block(x)
        return x