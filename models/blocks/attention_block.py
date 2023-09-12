import cv2
import librosa
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision import utils


class AttentionBlock(nn.Module):
    def __init__(self, in_features_l, in_features_g, attn_features, up_factor, normalize_attn=True):
        super(AttentionBlock, self).__init__()
        self.up_factor = up_factor
        self.normalize_attn = normalize_attn
        self.W_l = nn.Conv2d(in_channels=in_features_l, out_channels=attn_features, kernel_size=1, padding=0,
                             bias=False)
        self.W_g = nn.Conv2d(in_channels=in_features_g, out_channels=attn_features, kernel_size=1, padding=0,
                             bias=False)
        self.phi = nn.Conv2d(in_channels=attn_features, out_channels=1, kernel_size=1, padding=0, bias=True)

    def forward(self, l, g):
        N, C, W, H = l.size()
        l_ = self.W_l(l)
        g_ = self.W_g(g)
        if self.up_factor > 1:
            g_ = F.interpolate(g_, scale_factor=self.up_factor, mode='bilinear', align_corners=False)
        c = self.phi(F.relu(l_ + g_))  # batch_sizex1xWxH

        # compute attn map
        if self.normalize_attn:
            a = F.softmax(c.view(N, 1, -1), dim=2).view(N, 1, W, H)
        else:
            a = torch.sigmoid(c)
        # re-weight the local feature
        f = torch.mul(a.expand_as(l), l)  # batch_sizexCxWxH
        if self.normalize_attn:
            output = f.view(N, C, -1).sum(dim=2)  # weighted sum
        else:
            output = F.adaptive_avg_pool2d(f, (1, 1)).view(N, C)  # global average pooling
        return a, output


def print_original_spec(original_spec, fig, spec, n_rows, epoch):
    for idx, el in enumerate(original_spec):
        el = el.cpu().numpy()
        ax0 = fig.add_subplot(spec[0, idx])
      
        img = librosa.display.specshow(el, sr=16000, x_axis='time', y_axis='mel', ax=ax0)
        if idx == len(original_spec) - 1:
            fig.colorbar(img, format='%+2.0f dB', ax=ax0)  # Aggiungi una barra dei colori con etichette in dB


def visualize_attention(I_train: Tensor, a: Tensor, up_factor, no_attention=False):
    img = I_train.permute((1, 2, 0)).cpu().numpy()
    # compute the heatmap
    if up_factor > 1:
        a = F.interpolate(a, scale_factor=up_factor, mode='bilinear', align_corners=False)
    attn = utils.make_grid(a, nrow=8, normalize=True, scale_each=True, padding=2)
    attn = attn.permute((1, 2, 0)).mul(255).byte().cpu().numpy()
    attn = cv2.applyColorMap(attn, cv2.COLORMAP_JET)
    attn = cv2.cvtColor(attn, cv2.COLOR_BGR2RGB)
    attn = np.float32(attn) / 255
    # add the heatmap to the image
    img = cv2.resize(img, (466, 60))
    if no_attention:
        return torch.from_numpy(img)
    else:
        vis = 0.6 * img + 0.4 * attn
        vis = cv2.resize(vis, (vis.shape[1] * 3, vis.shape[0] * 2))
        num_images = 8
        padding_width = 30  # Adjust as needed

        vis = add_padding_to_grid(vis, num_images, padding_width)

        # Ensure the values are within the expected range
        vis = np.clip(vis, 0, 255).astype(np.uint8)
        vis = np.flipud(vis)
        return torch.from_numpy(vis.copy())


def add_padding_to_grid(grid, num_images, padding_width):
    img_w = grid.shape[1] // num_images
    # add padding
    padded_row_w = num_images * (img_w + padding_width - 1)  # -1 because first image does not have margin before
    padded_array = np.ones((grid.shape[0], padded_row_w, 3), dtype=np.uint8) * 255
    for i in range(num_images):
        img = grid[:, i * img_w: (i + 1) * img_w, :]
        start_col = i * (img_w + padding_width)
        padded_array[:, start_col: start_col + img_w, :] = img * 255
    return padded_array
