# Standard library imports
import math

# Third-party imports
import torch
import torch.nn.functional as F
import numpy as np



class NCC:
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        self.win = win

    def loss(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)


class MSE:
    """
    Mean squared error loss.
    """

    def loss(self, y_true, y_pred):
        return torch.mean((y_true - y_pred) ** 2)


class Dice:
    """
    N-D dice for segmentation
    """

    def loss(self, y_true, y_pred):
        ndims = len(list(y_pred.size())) - 2
        vol_axes = list(range(2, ndims + 2))
        top = 2 * (y_true * y_pred).sum(dim=vol_axes)
        bottom = torch.clamp((y_true + y_pred).sum(dim=vol_axes), min=1e-5)
        dice = torch.mean(top / bottom)
        return -dice


class Grad:
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        self.penalty = penalty
        self.loss_mult = loss_mult

    def _diffs(self, y):
        vol_shape = [n for n in y.shape][2:]
        ndims = len(vol_shape)

        df = [None] * ndims
        for i in range(ndims):
            d = i + 2
            # permute dimensions
            r = [d, *range(0, d), *range(d + 1, ndims + 2)]
            y = y.permute(r)
            dfi = y[1:, ...] - y[:-1, ...]

            # permute back
            # note: this might not be necessary for this loss specifically,
            # since the results are just summed over anyway.
            r = [*range(d - 1, d + 1), *reversed(range(1, d - 1)), 0, *range(d + 1, ndims + 2)]
            df[i] = dfi.permute(r)

        return df

    def loss(self, _, y_pred):
        if self.penalty == 'l1':
            dif = [torch.abs(f) for f in self._diffs(y_pred)]
        else:
            assert self.penalty == 'l2', 'penalty can only be l1 or l2. Got: %s' % self.penalty
            dif = [f * f for f in self._diffs(y_pred)]

        df = [torch.mean(torch.flatten(f, start_dim=1), dim=-1) for f in dif]
        grad = sum(df) / len(df)

        if self.loss_mult is not None:
            grad *= self.loss_mult

        return grad.mean()


# class SSIM:
#     """
#     Structural Similarity Index Measure (SSIM) Loss for 3D volumes.
#     """
#     def __init__(self, win_size=3, C1=0.01*2, C2=0.03*2):
#         self.win_size = win_size
#         self.C1 = C1
#         self.C2 = C2
    
#     @staticmethod
#     def gaussian_kernel_1d(size, sigma=1.0):
#         coords = torch.arange(size).float() - size // 2
#         g = torch.exp(-(coords*2) / (2 * sigma*2))
#         return g / g.sum()
#     gauss_1d = self.gaussian_kernel_1d(self.win_size).to(y_true.device)
    
#     def loss(self, y_true, y_pred):
#         # Gaussian kernel
#         print("[SSIM] Starting loss comutation")
#         ndims = len(list(y_true.size())) - 2
#         pad = self.win_size // 2
#         channels = y_true.size(1)
#         print(f"[SSIM] ndims={ndims}, pad={pad}, channels{channels}")
        
#         # sigma = 1.5
    
#         # def gaussian_jernel_1d(size, sigma=1.0):
#         #     coords = torch.aragne(size).float() - size//2
#         #     g = torch.exp(-(coords**2) / (2 * sigma**2))
#         #     return g / g.sum()
        
#         # gauss_1d = gaussian_kernel_1d(self.win_size).to(y_true.device)
    
        
#         if ndims == 3:
#             kernel = gauss_1d[:, None, None] * gauss_1d[None, :, None] * gauss_1d[None, None, :]
#             kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
#         else:
#             raise ValueError("SSIM implemented here only for 3D volumes")

#         kernel = kernel.repeat(channels, 1, 1, 1, 1)  # שכפול לערוצים
#         # print("[SSIM] kernel repeated for channels")

#         conv = torch.nn.functional.conv3d

#         mu_x = conv(y_true, kernel, padding=pad, groups=channels)
#         # print("[SSIM mu_x computed]")
#         mu_y = conv(y_pred, kernel, padding=pad, groups=channels)
#         # print("[SSIM nu_y computed]")


#         mu_x2 = mu_x**2
#         mu_y2 = mu_y**2
#         mu_xy = mu_x * mu_y
#         # print("[SSIM] nu_x^2, nu_y^2, mu_x*nu_y computed")


#         sigma_x2 = conv(y_true * y_true, kernel, padding=pad, groups=channels) - mu_x2
#         # print("[SSIM] sigma_x2 computed")
#         sigma_y2 = conv(y_pred * y_pred, kernel, padding=pad, groups=channels) - mu_y2
#         # print("[SSIM] sigma_y2 computed")
#         sigma_xy = conv(y_true * y_pred, kernel, padding=pad, groups=channels) - mu_xy
#         # print("[SSIM] sigma_xy computed")
#         ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / \
#                    ((mu_x2 + mu_y2 + self.C1) * (sigma_x2 + sigma_y2 + self.C2))
#         # print("[SSIM] ssim_map compute]")
#         return 1 -ssim_map.mean()

import torch
import torch.nn.functional as F

class SSIM:
    """
    Structural Similarity Index Measure (SSIM) Loss for 3D volumes with small kernel (3x3x3).
    """
    def __init__(self, win_size=5, C1=0.01*2, C2=0.03*2):
        self.win_size = win_size
        self.C1 = C1
        self.C2 = C2

    def loss(self, y_true, y_pred):
        ndims = len(list(y_true.size())) - 2
        if ndims != 3:
            raise ValueError("SSIM implemented only for 3D volumes")

        pad = self.win_size // 2
        channels = y_true.size(1)

        # Gaussian kernel 1D
        def gaussian_kernel_1d(size, sigma=1.0):
            coords = torch.arange(size).float() - size // 2
            g = torch.exp(-(coords*2) / (2 * sigma*2))
            return g / g.sum()

        gauss_1d = gaussian_kernel_1d(self.win_size).to(y_true.device)

        # 3D kernel
        kernel = gauss_1d[:, None, None] * gauss_1d[None, :, None] * gauss_1d[None, None, :]
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # [1,1,D,H,W]
        kernel = kernel.repeat(channels, 1, 1, 1, 1)  # repeat for all channels

        conv = F.conv3d

        mu_x = conv(y_true, kernel, padding=pad, groups=channels)
        mu_y = conv(y_pred, kernel, padding=pad, groups=channels)

        mu_x2 = mu_x ** 2
        mu_y2 = mu_y ** 2
        mu_xy = mu_x * mu_y

        sigma_x2 = conv(y_true * y_true, kernel, padding=pad, groups=channels) - mu_x2
        sigma_y2 = conv(y_pred * y_pred, kernel, padding=pad, groups=channels) - mu_y2
        sigma_xy = conv(y_true * y_pred, kernel, padding=pad, groups=channels) - mu_xy

        ssim_map = ((2 * mu_xy + self.C1) * (2 * sigma_xy + self.C2)) / \
                   ((mu_x2 + mu_y2 + self.C1) * (sigma_x2 + sigma_y2 + self.C2))

        return 1 -ssim_map.mean()
    

class SSIM_MSE():
    def __init__(self):
        self.mse = MSE()
        self.ssim = SSIM()

    def loss(self,  y_true, y_pred):
        return 0.5 * self.mse.loss(y_true, y_pred) + 0.5 * self.ssim.loss(y_true, y_pred) 