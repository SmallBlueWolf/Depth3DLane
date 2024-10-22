import sys
sys.path.append('/media/bluewolf/Data/bluewolf/projs/depth_label/Depth3DLane')
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.nn as nn
from models.util.load_model import load_checkpoint, resume_training
from models.util.save_model import save_model_dp
from models.loss import IoULoss, NDPushPullLoss
from utils.config_util import load_config_module
from sklearn.metrics import f1_score
import numpy as np
import pytorch_ssim
import os

os.chdir("/media/bluewolf/Data/bluewolf/projs/depth_label/Depth3DLane/tools")

class GradientLoss(nn.Module):
    """
    Computes the gradient loss between two depth maps.
    """
    def __init__(self):
        super(GradientLoss, self).__init__()
        # Define Sobel kernels for gradient computation
        self.grad_x = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
        self.grad_y = nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)

        # # Initialize Sobel kernels
        # sobel_x = torch.tensor([[[[-1, 0, 1],
        #                           [-2, 0, 2],
        #                           [-1, 0, 1]]]], dtype=torch.float32)
        # sobel_y = torch.tensor([[[[-1, -2, -1],
        #                           [0, 0, 0],
        #                           [1, 2, 1]]]], dtype=torch.float32)
        #
        # # Assign Sobel kernels to convolution layers
        # self.grad_x.weight = nn.Parameter(sobel_x, requires_grad=False)
        # self.grad_y.weight = nn.Parameter(sobel_y, requires_grad=False)

        # # Define L1 loss for gradients
        # self.l1 = nn.L1Loss()

        # Initialize Scharr kernels
        scharr_x = torch.tensor([[[[-3, 0, 3],
                                   [-10, 0, 10],
                                   [-3, 0, 3]]]], dtype=torch.float32)
        scharr_y = torch.tensor([[[[-3, -10, -3],
                                   [0, 0, 0],
                                   [3, 10, 3]]]], dtype=torch.float32)
        # Assign Scharr kernels to convolution layers
        self.grad_x.weight = nn.Parameter(scharr_x, requires_grad=False)
        self.grad_y.weight = nn.Parameter(scharr_y, requires_grad=False)

        self.huber = nn.SmoothL1Loss()


    def forward(self, pred, target):
        # Compute gradients for predictions and targets
        pred_grad_x = self.grad_x(pred)
        pred_grad_y = self.grad_y(pred)
        target_grad_x = self.grad_x(target)
        target_grad_y = self.grad_y(target)

        # # Compute L1 loss between gradients
        # loss_x = self.l1(pred_grad_x, target_grad_x)
        # loss_y = self.l1(pred_grad_y, target_grad_y)

        # Compute Huber loss between gradients
        loss_x = self.huber(pred_grad_x, target_grad_x)
        loss_y = self.huber(pred_grad_y, target_grad_y)

        return loss_x + loss_y


class CombinedDepthLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.3, gamma=0.2): # 这里的可以根据效果修改，三种损失函数的比例
        """
        Initializes the CombinedDepthLoss.

        Parameters:
        - alpha (float): Weight for L1 loss.
        - beta (float): Weight for SSIM loss.
        - gamma (float): Weight for Gradient loss.
        """
        super(CombinedDepthLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = pytorch_ssim.SSIM(window_size=11, size_average=True)
        self.gradient_loss = GradientLoss()

    def forward(self, pred, target):
        """
        Computes the combined depth loss.

        Parameters:
        - pred (torch.Tensor): Predicted depth map (B, 1, H, W).
        - target (torch.Tensor): Ground truth depth map (B, 1, H, W).

        Returns:
        - torch.Tensor: Combined depth loss.
        """
        # Ensure the depth maps are in the same range
        pred = torch.sigmoid(pred)  # Normalize to [0, 1] if not already

        # Compute individual loss components
        loss_l1 = self.l1_loss(pred, target)
        loss_ssim = 1 - self.ssim_loss(pred, target)
        loss_grad = self.gradient_loss(pred, target)

        # Combine losses with respective weights
        combined_loss = self.alpha * loss_l1 + self.beta * loss_ssim + self.gamma * loss_grad
        return combined_loss


class Combine_Model_and_Loss(torch.nn.Module):
    def __init__(self, model):
        super(Combine_Model_and_Loss, self).__init__()
        self.model = model
        self.bce = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]))
        self.iou_loss = IoULoss()
        self.poopoo = NDPushPullLoss(1.0, 1., 1.0, 5.0, 200)
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        # Initialize the combined depth loss
        self.combined_depth_loss = CombinedDepthLoss(alpha=0.5, beta=0.3, gamma=0.2)

    def forward(self, inputs, gt_seg=None, gt_instance=None, gt_offset_y=None, gt_z=None, image_gt_segment=None,
                image_gt_instance=None, depth_gt_map=None, train=True):
        """
        Forward pass that computes predictions and losses.

        Parameters:
        - inputs (torch.Tensor): Input images (B, 3, H, W).
        - gt_seg (torch.Tensor): Ground truth segmentation maps.
        - gt_instance (torch.Tensor): Ground truth instance maps.
        - gt_offset_y (torch.Tensor): Ground truth offset maps.
        - gt_z (torch.Tensor): Ground truth height maps.
        - image_gt_segment (torch.Tensor): Ground truth 2D segmentation maps.
        - image_gt_instance (torch.Tensor): Ground truth 2D instance maps.
        - depth_gt_map (torch.Tensor): Ground truth depth maps.
        - train (bool): Flag indicating training mode.

        Returns:
        - If train=True:
            - pred (torch.Tensor): Predicted segmentation maps.
            - loss_total (torch.Tensor): Combined BEV and 2D loss.
            - loss_total_2d (torch.Tensor): Combined 2D loss.
            - loss_offset (torch.Tensor): Offset loss.
            - loss_z (torch.Tensor): Height loss.
            - loss_depth (torch.Tensor): Depth loss.
        - Else:
            - pred (torch.Tensor): Predicted segmentation maps.
        """
        # Forward pass through the model
        res = self.model(inputs)
        lane_outputs, lane_2d_outputs, depth_map = res  # Unpack model outputs
        pred, emb, offset_y, z = lane_outputs
        pred_2d, emb_2d = lane_2d_outputs

        if train:
            ## 3D Losses
            # Segmentation Loss
            loss_seg = self.bce(pred, gt_seg) + self.iou_loss(torch.sigmoid(pred), gt_seg)

            # Embedding Loss
            loss_emb = self.poopoo(emb, gt_instance)

            # Offset Loss
            loss_offset = self.bce_loss(gt_seg * torch.sigmoid(offset_y), gt_offset_y)

            # Height Loss
            loss_z = self.mse_loss(gt_seg * z, gt_z)

            # Combined BEV Loss
            loss_total = 3 * loss_seg + 0.5 * loss_emb
            loss_total = loss_total.unsqueeze(0)
            loss_offset = 60 * loss_offset.unsqueeze(0)
            loss_z = 30 * loss_z.unsqueeze(0)

            ## 2D Losses
            # 2D Segmentation Loss
            loss_seg_2d = self.bce(pred_2d, image_gt_segment) + self.iou_loss(torch.sigmoid(pred_2d), image_gt_segment)

            # 2D Embedding Loss
            loss_emb_2d = self.poopoo(emb_2d, image_gt_instance)

            # Combined 2D Loss
            loss_total_2d = 3 * loss_seg_2d + 0.5 * loss_emb_2d
            loss_total_2d = loss_total_2d.unsqueeze(0)

            ## Depth Loss
            if depth_gt_map is not None and depth_map is not None:
                loss_depth = self.combined_depth_loss(depth_map, depth_gt_map)
                loss_depth = loss_depth.unsqueeze(0)
                loss_total_combined = loss_total + 0.5 * loss_total_2d + loss_offset + loss_z + loss_depth # loss_depth后面可能需要增加系数
                return pred, loss_total_combined, loss_total_2d, loss_offset, loss_z, loss_depth
            else:
                loss_total_combined = loss_total + 0.5 * loss_total_2d + loss_offset + loss_z
                return pred, loss_total_combined, loss_total_2d, loss_offset, loss_z
            ## Total Combined Loss
            # Adjust weights as necessary

        else:
            return pred


def train_epoch(model, dataloader, optimizer, configs, epoch):
    """
    Trains the model for one epoch.

    Parameters:
    - model (torch.nn.Module): Combined model and loss.
    - dataloader (DataLoader): DataLoader for training data.
    - optimizer (torch.optim.Optimizer): Optimizer for training.
    - configs: Configuration module containing training parameters.
    - epoch (int): Current epoch number.
    """
    model.train()
    losses_avg = {}
    for idx, output in enumerate(dataloader):
        # Move data to GPU
        if len(output) == 8:
            input_data, gt_seg_data, gt_emb_data, offset_y_data, z_data, image_gt_segment, image_gt_instance, depth_gt_map = output
            depth_gt_map = depth_gt_map.cuda()
        else:
            input_data, gt_seg_data, gt_emb_data, offset_y_data, z_data, image_gt_segment, image_gt_instance = output
            depth_gt_map = None
        input_data = input_data.cuda()
        gt_seg_data = gt_seg_data.cuda()
        gt_emb_data = gt_emb_data.cuda()
        offset_y_data = offset_y_data.cuda()
        z_data = z_data.cuda()
        image_gt_segment = image_gt_segment.cuda()
        image_gt_instance = image_gt_instance.cuda()

        # Forward pass
        outputs = model(
            input_data,
            gt_seg=gt_seg_data,
            gt_instance=gt_emb_data,
            gt_offset_y=offset_y_data,
            gt_z=z_data,
            image_gt_segment=image_gt_segment,
            image_gt_instance=image_gt_instance,
            depth_gt_map=depth_gt_map,
            train=True
        )

        if len(outputs) == 6:
            prediction, loss_total_combined, loss_total_2d, loss_offset, loss_z, loss_depth = outputs
            loss_depth = loss_depth.mean()
        elif len(outputs) == 5:
            prediction, loss_total_combined, loss_total_2d, loss_offset, loss_z = outputs
            loss_depth = 0
        else:
            raise ValueError("Expected 5 or 6 outputs from the model during training.")

        # Compute mean losses
        loss_back_combined = loss_total_combined.mean()
        loss_back_2d = loss_total_2d.mean()
        loss_offset = loss_offset.mean()
        loss_z = loss_z.mean()

        # Total loss
        # Adjust weights as necessary; currently already weighted in Combine_Model_and_Loss
        loss_back_total = loss_back_combined + 0.5 * loss_back_2d + loss_offset + loss_z + loss_depth

        # Backward pass and optimization
        optimizer.zero_grad()
        loss_back_total.backward()
        optimizer.step()

        # Logging
        if idx % 50 == 0:
            if loss_depth != 0:
                print(
                    f"Epoch [{epoch + 1}], Step [{idx}/{len(dataloader)}], BEV Loss: {loss_back_combined.item():.4f}, Depth Loss: {loss_depth.item():.4f} {'*' * 10}")
            else:
                print(f"Epoch [{epoch + 1}], Step [{idx}/{len(dataloader)}], Loss: {loss_back_combined.item():.4f}")

        if idx % 300 == 0:
            target = gt_seg_data.detach().cpu().numpy().ravel()
            pred = torch.sigmoid(prediction).detach().cpu().numpy().ravel()
            f1_bev_seg = f1_score(
                (target > 0.5).astype(np.int64),
                (pred > 0.5).astype(np.int64),
                zero_division=1
            )
            if loss_depth != 0:
                loss_iter = {
                    "BEV Loss": loss_back_combined.item(),
                    "Offset Loss": loss_offset.item(),
                    "Z Loss": loss_z.item(),
                    "Depth Loss": loss_depth.item(),
                    "F1_BEV_seg": f1_bev_seg
                }
            else:
                loss_iter = {
                    "BEV Loss": loss_back_combined.item(),
                    "Offset Loss": loss_offset.item(),
                    "Z Loss": loss_z.item(),
                    "F1_BEV_seg": f1_bev_seg
                }
            print(f"Epoch [{epoch + 1}], Step [{idx}], Losses: {loss_iter}")


def worker_function(config_file, gpu_id, checkpoint_path=None):
    print('Using GPU IDs:', ','.join([str(i) for i in gpu_id]))
    configs = load_config_module(config_file)

    ''' Models and Optimizer '''
    model = configs.model()
    model = Combine_Model_and_Loss(model)
    if torch.cuda.is_available():
        model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=gpu_id)
    optimizer = configs.optimizer(
        filter(lambda p: p.requires_grad, model.parameters()),
        **configs.optimizer_params
    )
    scheduler = getattr(configs, "scheduler", CosineAnnealingLR)(optimizer, configs.epochs)

    if checkpoint_path:
        if getattr(configs, "load_optimizer", True):
            resume_training(checkpoint_path, model.module, optimizer, scheduler)
        else:
            load_checkpoint(checkpoint_path, model.module, None)

    ''' Dataset '''
    Dataset = getattr(configs, "train_dataset", None)
    if Dataset is None:
        Dataset = configs.training_dataset
    train_loader = DataLoader(Dataset(), **configs.loader_args, pin_memory=True)

    ''' Training Loop '''
    for epoch in range(configs.epochs):
        print('*' * 100, f"Epoch {epoch + 1}/{configs.epochs}")
        train_epoch(model, train_loader, optimizer, configs, epoch)
        scheduler.step()

        # Save model checkpoints
        save_model_dp(model, optimizer, configs.model_save_path, f'ep{epoch + 1:03d}.pth')
        save_model_dp(model, None, configs.model_save_path, 'latest.pth')


# TODO template config file.
if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")
    worker_function('./openlane_config.py', gpu_id=[0])
