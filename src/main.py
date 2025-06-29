import PIL
import PIL.Image
import torch
import torch.nn as nn
from torch.nn.functional import conv2d, interpolate
from torch.optim import Adam
from torchvision.transforms.functional import normalize, pil_to_tensor

from model.model import DepthAnythingV2

device = "cuda"


class AffineInvariantLoss(nn.Module):
    def __init__(self, trim_factor: float):
        super().__init__()
        self._trim_factor = trim_factor

    def forward(self, disparity_map_gt: torch.Tensor, disparity_map_pred: torch.Tensor):
        assert disparity_map_gt.dim() == 4
        assert disparity_map_gt.shape == disparity_map_pred.shape

        loss_map = disparity_map_pred - disparity_map_gt

        t_gt = torch.nanmedian(disparity_map_gt)
        t_pred = torch.nanmedian(disparity_map_pred)

        s_gt = torch.nanmean(torch.abs(disparity_map_gt - t_gt))
        s_pred = torch.nanmean(torch.abs(disparity_map_pred - t_pred))

        scaled_and_shifted_gt = (disparity_map_gt - t_gt) / s_gt
        scaled_and_shifted_pred = (disparity_map_pred - t_pred) / s_pred

        valid_mask = torch.isfinite(scaled_and_shifted_gt)
        loss_map = torch.abs(scaled_and_shifted_pred - scaled_and_shifted_gt)
        loss_map[~valid_mask] = 0.0

        trimmed_count = int((1.0 - self._trim_factor) * torch.sum(valid_mask))
        loss_topk = torch.topk(loss_map.flatten(), trimmed_count, largest=False).values
        loss = torch.mean(loss_map.flatten())

        return loss


class GradientMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, disparity_map_gt: torch.Tensor, disparity_map_pred: torch.Tensor):
        assert disparity_map_gt.dim() == 4
        assert disparity_map_gt.shape == disparity_map_pred.shape

        dx_kernel = torch.tensor([[-1, 1]], device=disparity_map_gt.device).float().unsqueeze(0).unsqueeze(0)
        dy_kernel = torch.tensor([[-1], [1]], device=disparity_map_gt.device).float().unsqueeze(0).unsqueeze(0)

        disparity_map_gt_dx = conv2d(disparity_map_gt, dx_kernel, padding=0)
        disparity_map_gt_dy = conv2d(disparity_map_gt, dy_kernel, padding=0)

        disparity_map_pred_dx = conv2d(disparity_map_pred, dx_kernel, padding=0)
        disparity_map_pred_dy = conv2d(disparity_map_pred, dy_kernel, padding=0)

        valid_mask_dx = torch.isfinite(disparity_map_gt_dx)
        valid_mask_dy = torch.isfinite(disparity_map_gt_dy)

        loss_dx = torch.abs(disparity_map_gt_dx - disparity_map_pred_dx)
        loss_dx[~valid_mask_dx] = 0.0
        loss_dy = torch.abs(disparity_map_gt_dy - disparity_map_pred_dy)
        loss_dy[~valid_mask_dy] = 0.0

        loss = (torch.sum(loss_dx) / torch.sum(valid_mask_dx) + torch.sum(loss_dy) / torch.sum(valid_mask_dy)) / 2.0
        return loss


class MultiScaleGradientMatchingLoss(nn.Module):
    def __init__(self, levels: int):
        super().__init__()
        self.levels = levels
        self.gm_loss_fn = GradientMatchingLoss()

    def forward(self, disparity_map_gt: torch.Tensor, disparity_map_pred: torch.Tensor):
        assert disparity_map_gt.dim() == 4
        assert disparity_map_gt.shape == disparity_map_pred.shape

        loss = 0.0
        for k in range(self.levels):
            scale_factor = 1 / pow(2.0, k)
            disparity_map_gt_scaled = interpolate(disparity_map_gt, scale_factor=scale_factor)
            disparity_map_pred_scaled = interpolate(disparity_map_pred, scale_factor=scale_factor)
            level_loss = self.gm_loss_fn(disparity_map_gt_scaled, disparity_map_pred_scaled)
            loss += level_loss

        loss /= self.levels
        return loss


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self._ai_fn = AffineInvariantLoss(trim_factor=0.1)
        self._gm_loss_fn = MultiScaleGradientMatchingLoss(levels=4)

    def forward(self, disparity_map_gt: torch.Tensor, disparity_map_pred: torch.Tensor):
        ai_loss = self._ai_fn(disparity_map_gt, disparity_map_pred)
        gm_loss = self._gm_loss_fn(disparity_map_gt, disparity_map_pred)
        loss = ai_loss + 0.5 * gm_loss
        return loss


if __name__ == "__main__":
    # Example from ml-hypersim (ai_001_001/images/scene_cam_00_final_preview/frame.0088.color.jpg)
    color_map = pil_to_tensor(PIL.Image.open("color_map.tiff")).to(device).float().unsqueeze(0)
    color_map = normalize(color_map, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    disparity_map_gt = pil_to_tensor(PIL.Image.open("disparity_map.tiff")).to(device).unsqueeze(0)

    model = DepthAnythingV2(encoder="vits", freeze_backend=True)
    model.to(device)
    trainable_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = Adam(trainable_parameters, lr=1e-6, fused=True)
    loss_fn = CombinedLoss()

    while True:
        optimizer.zero_grad()
        disparity_map_pred = model(color_map).unsqueeze(0)
        loss = loss_fn(disparity_map_gt, disparity_map_pred)
        loss.backward()
        optimizer.step()
        print(f"current loss is {loss}")
