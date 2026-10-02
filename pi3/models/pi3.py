import torch
import torch.nn as nn
from functools import partial
from copy import deepcopy

from .dinov2.layers import Mlp
from ..utils.geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import TransformerDecoder, LinearPts3d
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from huggingface_hub import PyTorchModelHubMixin

from einops import rearrange
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.colors import LogNorm
import matplotlib.cm as cm
import matplotlib.ticker as mticker
import numpy as np 

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

def denorm_imagenet(x):  # x: (3,H,W)
    return (x * IMAGENET_STD.to(x.device)) + IMAGENET_MEAN.to(x.device)

def top_attention_pairs(A, S, T, topk=25, b=0, head=None, exclude_self=False):
    ST = S * T
    assert A.shape[-2:] == (ST, ST), f"Expected {(ST,ST)} but got {tuple(A.shape[-2:])}"

    # mask out diagonal if desired
    if exclude_self:
        diag = torch.eye(ST, device=A.device, dtype=torch.bool)
        A = A.masked_fill(diag, float("-inf"))

    # topk over all directed pairs (q -> k)
    flat = A.reshape(-1)                       # (ST*ST,)
    vals, idx = torch.topk(flat, k=topk)       # idx in [0, ST*ST)

    q_idx = idx // ST                          # query token global index
    k_idx = idx %  ST                          # key token global index

    q_frame = (q_idx // T).tolist()
    q_tok   = (q_idx %  T).tolist()
    k_frame = (k_idx // T).tolist()
    k_tok   = (k_idx %  T).tolist()

    out = []
    for r in range(topk):
        out.append({
            "rank": r,
            "score": float(vals[r].item()),
            "q_idx": int(q_idx[r].item()),
            "k_idx": int(k_idx[r].item()),
            "q_frame": int(q_frame[r]),
            "q_tok": int(q_tok[r]),
            "k_frame": int(k_frame[r]),
            "k_tok": int(k_tok[r]),
        })
    return out

def draw_top_pairs_on_frames(
    imgs,                   # (B,S,3,H,W) torch, values [0,1] or [-1,1]
    pairs,                  # list of dicts from top_attention_pairs(...)
    S, T,
    b=0,
    patch=14,
    max_pairs=50,
    figsize_per_frame=3.0,
    line_alpha=0.5,
    box_lw=1.0,
    marker_size=1,
    title=None,
    out_path=None,
    dpi=300
):
    assert imgs.ndim == 4
    S_img, C, H, W = imgs.shape
    assert S_img == S and C == 3

    # ---- denormalize to [0,1] ----
    x = imgs.detach().float()
    x = denorm_imagenet(x)
    x = x.clamp(0, 1).cpu()
    x = x.permute(0, 2, 3, 1).numpy()   # (S,H,W,3)


    # map token -> (row,col) -> pixel center
    Hp = H // patch
    Wp = W // patch
    assert Hp * Wp == T, f"H//p*W//p={Hp*Wp} but T={T}. H,W={H},{W}, p={patch}"
    def tok_to_xy(tok):
        r = tok // Wp
        c = tok % Wp
        # patch box
        x0 = c * patch
        y0 = r * patch
        cx = x0 + patch * 0.5
        cy = y0 + patch * 0.5
        return cx, cy, x0, y0

    fig = plt.figure(figsize=(S * 3, 3), frameon=False)  # frameon=False removes border
    axes = []

    for s in range(S):
        # manually place each image to fill the figure (no gaps, no margins)
        ax = fig.add_axes([s / S, 0.0, 1.0 / S, 1.0])  # [left, bottom, width, height]
        ax.imshow(x[s])#, interpolation="nearest")
        ax.set_axis_off()
        ax.set_xlim([0, W])
        ax.set_ylim([H, 0])
        axes.append(ax)

    # color by rank (1..max_pairs)
    cmap = cm.get_cmap("hsv", max_pairs)

    for r, p in enumerate(pairs[:max_pairs]):
        qf, qt = p["q_frame"], p["q_tok"]
        kf, kt = p["k_frame"], p["k_tok"]

        # skip if out of range (safety)
        if not (0 <= qf < S and 0 <= kf < S):
            continue
        if not (0 <= qt < T and 0 <= kt < T):
            continue
        qcx, qcy, qx0, qy0 = tok_to_xy(qt)
        kcx, kcy, kx0, ky0 = tok_to_xy(kt)

        col = cmap(r)

        # draw boxes around patches
        axes[qf].add_patch(Rectangle((qx0, qy0), patch, patch, fill=False, lw=box_lw, edgecolor=col))
        axes[qf].scatter([qcx], [qcy], s=marker_size, c=[col])

        # if self-pair: just draw once (no line)
        if (qf == kf) and (qt == kt):
            continue

        # otherwise draw key patch box + dot + connection line
        axes[kf].add_patch(Rectangle((kx0, ky0), patch, patch, fill=False, lw=box_lw, edgecolor=col))
        axes[kf].scatter([kcx], [kcy], s=marker_size, c=[col])

        con = ConnectionPatch(
            xyA=(qcx, qcy), coordsA=axes[qf].transData,
            xyB=(kcx, kcy), coordsB=axes[kf].transData,
            arrowstyle="-",
            lw=1.2,
            color=col,
            alpha=line_alpha,
        )
        fig.add_artist(con)

    if title is not None:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()
    if out_path is not None:
        plt.savefig(out_path, dpi=dpi,  bbox_inches=None, pad_inches=0)
        plt.close(fig)
    else:
        plt.show()

def save_attn_full_and_zoom_two_figs(
    A, zoom_i0, zoom_j0, zoom_h, zoom_w,
    out_path_full, out_path_zoom,
    cmap="magma", use_log=True, dpi=300
):
    # to numpy
    if hasattr(A, "detach"):
        A = A.detach().float().cpu().numpy()
    A = np.asarray(A)
    N = A.shape[0]
    assert A.shape[0] == A.shape[1]

    # crop
    i1 = min(N, zoom_i0 + zoom_h)
    j1 = min(N, zoom_j0 + zoom_w)
    crop = A[zoom_i0:i1, zoom_j0:j1]

    # norm
    if use_log:
        vmin = np.percentile(crop, 0.001 * 100)
        vmin = max(vmin, 1e-12)
        vmax = np.percentile(crop, 99.9)
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    # =========================
    # FIG 1: FULL (NO WHITE AREA)
    # =========================
    fig1 = plt.figure(figsize=(3.2, 3.2), frameon=False)
    ax1 = fig1.add_axes([0, 0, 1, 1])  # fill entire canvas

    ax1.imshow(A, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
    ax1.set_axis_off()

    rect = Rectangle(
        (zoom_j0 - 0.5, zoom_i0 - 0.5),
        (j1 - zoom_j0),
        (i1 - zoom_i0),
        fill=False,
        edgecolor="red",
        linewidth=0.8,
    )
    ax1.add_patch(rect)

    fig1.savefig(out_path_full, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig1)

    # =========================
    # FIG 2: ZOOM + COLORBAR
    # =========================

    fig2, ax_zoom = plt.subplots(figsize=(4.0, 3.2), frameon=False)

    im = ax_zoom.imshow(crop, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
    ax_zoom.set_axis_off()

    # Colorbar
    cb = fig2.colorbar(im, ax=ax_zoom, fraction=0.046, pad=0.02)

    # Set tick locations (8 ticks across the actual displayed numeric range)
    vmin, vmax = im.norm.vmin, im.norm.vmax
    ticks = np.linspace(vmin, vmax, 8)
    cb.set_ticks(ticks)

    # Use formatter (cleaner than set_ticklabels)
    cb.ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.4f"))

    # Disable minor ticks completely
    cb.ax.minorticks_off()
    cb.ax.yaxis.set_minor_locator(mticker.NullLocator())
    cb.ax.tick_params(which="minor", length=0)

    # Control major tick style
    cb.ax.tick_params(which="major", labelsize=6, length=2)

    plt.tight_layout(pad=0.0)

    fig2.savefig(out_path_zoom, dpi=dpi, bbox_inches=None, pad_inches=0)
    plt.close(fig2)




class Pi3(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            pos_type='rope100',
            decoder_size='large',
        ):
        super().__init__()

        # ----------------------
        #        Encoder
        # ----------------------
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token

        # ----------------------
        #  Positonal Encoding
        # ----------------------
        self.pos_type = pos_type if pos_type is not None else 'none'
        self.rope=None
        if self.pos_type.startswith('rope'): # eg rope100 
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(self.pos_type[len('rope'):])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        

        # ----------------------
        #        Decoder
        # ----------------------
        enc_embed_dim = self.encoder.blocks[0].attn.qkv.in_features        # 1024
        if decoder_size == 'small':
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'base':
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == 'large':
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        # self.encoder_to_decoder = (
        #     nn.Identity()
        #     if enc_embed_dim == dec_embed_dim
        #     else nn.Linear(enc_embed_dim, dec_embed_dim)
        # )
        self.decoder = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=0.01,
                qk_norm=True,
                attn_class=FlashAttentionRope,
                rope=self.rope
            ) for _ in range(dec_depth)])
        self.dec_embed_dim = dec_embed_dim
        
        # ----------------------
        #     Register_token
        # ----------------------
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(torch.randn(1, 1, num_register_tokens, self.dec_embed_dim))
        nn.init.normal_(self.register_token, std=1e-6)

        # ----------------------
        #  Local Points Decoder
        # ----------------------
        self.point_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
        )
        self.point_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=3)

        # ----------------------
        #     Conf Decoder
        # ----------------------
        self.conf_decoder = deepcopy(self.point_decoder)
        self.conf_head = LinearPts3d(patch_size=14, dec_embed_dim=1024, output_dim=1)

        # ----------------------
        #  Camera Pose Decoder
        # ----------------------
        self.camera_decoder = TransformerDecoder(
            in_dim=2*self.dec_embed_dim, 
            dec_embed_dim=1024,
            dec_num_heads=16,                # 8
            out_dim=512,
            rope=self.rope,
            use_checkpoint=False
        )
        self.camera_head = CameraHead(dim=512)

        # For ImageNet Normalize
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)


    def decode(self, hidden, N, H, W):
        BN, hw, _ = hidden.shape
        B = BN // N

        final_output = []
        # hidden = self.encoder_to_decoder(hidden)
        hidden = hidden.reshape(B*N, hw, -1)
        

        register_token = self.register_token.repeat(B, N, 1, 1).reshape(B*N, *self.register_token.shape[-2:])

        # Concatenate special tokens with patch tokens
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]

        if self.pos_type.startswith('rope'):
            pos = self.position_getter(B * N, H//self.patch_size, W//self.patch_size, hidden.device)

        if self.patch_start_idx > 0:
            # do not use position embedding for special tokens (camera and register tokens)
            # so set pos to 0 for the special tokens
            pos = pos + 1
            pos_special = torch.zeros(B * N, self.patch_start_idx, 2).to(hidden.device).to(pos.dtype)
            pos = torch.cat([pos_special, pos], dim=1)
       
        for i in range(len(self.decoder)):
            blk = self.decoder[i]

            if i % 2 == 0:
                pos = pos.reshape(B*N, hw, -1)
                hidden = hidden.reshape(B*N, hw, -1)
            else:
                pos = pos.reshape(B, N*hw, -1)
                hidden = hidden.reshape(B, N*hw, -1)

            hidden = blk(hidden, xpos=pos)

            if i+1 in [len(self.decoder)-1, len(self.decoder)]:
                final_output.append(hidden.reshape(B*N, hw, -1))

        return torch.cat([final_output[0], final_output[1]], dim=-1), pos.reshape(B*N, hw, -1)
    
    def forward(self, imgs):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        hidden, pos = self.decode(hidden, N, H, W)
        BN, hw, _ = hidden.shape
        
        import os
        # os.makedirs('/cis/home/zshao14/Documents/dpp/yan2017/kitchen/pi3/token_pairs', exist_ok=True)
        # os.makedirs('/cis/home/zshao14/Documents/dpp/yan2017/kitchen/pi3/zoom_attn', exist_ok=True)
        # os.makedirs('/cis/home/zshao14/Documents/dpp/heinly2014/alex_subset/SparsePi3_fix-swa-4-1/frame_attn', exist_ok=True)
        # for i in range(len(self.decoder)):
        #     if i % 2 != 0:
        #         attn = self.decoder[i].attn.last_attn
        #         mean_attn = attn.mean(dim=1)[0].view(N, hw, N, hw)[:,self.patch_start_idx:,:,self.patch_start_idx:]
        #         frame_to_frame = mean_attn.sum(dim=3).mean(dim=1)
        #         frame_to_frame = frame_to_frame / (frame_to_frame.sum(dim=-1, keepdim=True) + 1e-8)
        #         # frame_to_frame = F.normalize(frame_to_frame, p=1, dim=-1)
        #         # Percentage of attention dedicated to the top 5 most attended frames
        #         topk_values, _ = torch.topk(frame_to_frame, k=3, dim=-1)
        #         energy_fraction = topk_values.sum(dim=-1) # Goal: Close to 1.0 (e.g., > 0.90)
        #         print(f"Layer{i} Top-3 frames capture {energy_fraction.mean().item():.2%} of total attention.")
        #         plot_frame_attention_plt(frame_to_frame, f'/cis/home/zshao14/Documents/dpp/heinly2014/alex_subset/SparsePi3_fix-swa-4-1/frame_attn/layer{i}.png')
            
            # attn_vis = attn.mean(dim=1)[0].view(N, hw, N, hw)[:,self.patch_start_idx:,:,self.patch_start_idx:].reshape(N*(hw-self.patch_start_idx),N*(hw-self.patch_start_idx))
            
            # pairs = top_attention_pairs(attn_vis, N, hw-self.patch_start_idx, topk=50)

            # draw_top_pairs_on_frames(
            #     imgs,                   
            #     pairs,    
            #     S, P-self.patch_start_idx,
            #     out_path=f'/cis/home/zshao14/Documents/dpp/yan2017/test/vggt/token_pairs/layer{i}.png',
            # )
            # save_attn_full_and_zoom_two_figs(
            #     attn_vis, zoom_i0=P-self.patch_start_idx, zoom_j0=P-self.patch_start_idx, zoom_h=P-self.patch_start_idx, zoom_w=P-self.patch_start_idx, 
            #     out_path_full=f'/cis/home/zshao14/Documents/dpp/yan2017/test/vggt/zoom_attn/layer{i}_full.png', out_path_zoom=f'/cis/home/zshao14/Documents/dpp/yan2017/test/vggt/zoom_attn/layer{i}_zoom.png'
            # )

        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            conf_hidden = conf_hidden.float()
            conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)

            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
        )
    def inference(self, imgs):
        imgs = (imgs - self.image_mean) / self.image_std

        B, N, _, H, W = imgs.shape
        patch_h, patch_w = H // 14, W // 14
        
        # encode by dinov2
        imgs = imgs.reshape(B*N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)

        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]

        hidden, pos = self.decode(hidden, N, H, W)
        
        point_hidden = self.point_decoder(hidden, xpos=pos)
        conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)

        with torch.amp.autocast(device_type='cuda', enabled=False):
            # local points
            point_hidden = point_hidden.float()
            ret = self.point_head([point_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)

            # confidence
            conf_hidden = conf_hidden.float()
            conf = self.conf_head([conf_hidden[:, self.patch_start_idx:]], (H, W)).reshape(B, N, H, W, -1)

            # camera
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(camera_hidden[:, self.patch_start_idx:], patch_h, patch_w).reshape(B, N, 4, 4)

            # unproject local points using camera poses
            points = torch.einsum('bnij, bnhwj -> bnhwi', camera_poses, homogenize_points(local_points))[..., :3]

        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
        )
def plot_frame_attention_plt(frame_matrix, output_path):
    """
    Visualizes the frame-to-frame attention matrix using matplotlib only.
    
    Args:
        frame_matrix (Tensor or ndarray): Shape (F, F) normalized attention matrix
    """
    # Convert PyTorch tensor to numpy if necessary
    if isinstance(frame_matrix, torch.Tensor):
        matrix_np = frame_matrix.detach().cpu().numpy()
    else:
        matrix_np = np.array(frame_matrix)
        
    F = matrix_np.shape[0]
    
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # Render the heatmap matrix
    # cmap='viridis' gives great contrast for sparsity (dark blue/purple is close to 0)
    im = ax.imshow(matrix_np, cmap='viridis', aspect='equal')
    
    # Add colorbar on the side
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.03)
    cbar.ax.set_ylabel("Attention Weight Fraction", rotation=-90, va="bottom", fontsize=11)
    
    # Configure grid ticks and labels
    ax.set_xticks(np.arange(F))
    ax.set_yticks(np.arange(F))
    ax.set_xticklabels([f"F{i}" for i in range(F)], fontsize=10)
    ax.set_yticklabels([f"F{i}" for i in range(F)], fontsize=10)
    
    # Rotate the x-axis tick labels for readability
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center")
    
    # Loop over data dimensions and create text annotations inside the blocks
    # We choose black or white text dynamically depending on background brightness
    threshold = matrix_np.max() / 2.0
    for i in range(F):
        for j in range(F):
            val = matrix_np[i, j]
            color = "black" if val > threshold else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)
            
    # Set titles and structural labels
    # ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Key Frames", fontsize=12, labelpad=10)
    ax.set_ylabel("Query Frames", fontsize=12, labelpad=10)
    
    # Turn off the white bounding frame spines for a cleaner look
    for edge in ['top', 'right', 'bottom', 'left']:
        ax.spines[edge].set_visible(False)
        
    plt.tight_layout()
    plt.savefig(output_path)
