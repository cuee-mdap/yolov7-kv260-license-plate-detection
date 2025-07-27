import os
import math
import warnings
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.ops import nms
from PIL import Image
from tqdm import tqdm
import logging
import sys
import random
from datetime import datetime

# เพิ่มการนำเข้า NumPy
import numpy as np

# ป้องกันการค้างจาก Matplotlib ในสภาพแวดล้อมไม่มี GUI
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# กำหนด seed ให้ผลเทรน reproducible
def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# ปิด FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# Setup Logger with Timestamp
# =============================================================================
def setup_logger(name='train_yolov7'):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = os.path.join('logs', f'training_{timestamp}.log')
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    if not logger.handlers:
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(console_handler)

    return logger

# =============================================================================
# CSP Bottleneck
# =============================================================================
class CSPBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, shortcut=True):
        super(CSPBottleneck, self).__init__()
        self.shortcut = shortcut and (in_channels == out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(out_channels, out_channels, 1, 1, 0)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

        if not self.shortcut:
            self.shortcut_conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, 1, 0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.shortcut:
            out += identity
        else:
            out += self.shortcut_conv(identity)
        out = self.relu3(out)
        return out

class YOLOv7TinySmallPlus(nn.Module):
    """
    YOLOv7TinySmallPlus model with BatchNorm2d layers
    """
    def __init__(self, nc=1, num_predictions=3):
        super(YOLOv7TinySmallPlus, self).__init__()
        self.num_predictions = num_predictions

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU(inplace=True)

        self.csp1 = CSPBottleneck(16, 32)
        self.down1 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn_down1 = nn.BatchNorm2d(64)

        self.csp2 = CSPBottleneck(64, 64)
        self.down2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn_down2 = nn.BatchNorm2d(128)

        self.csp3 = CSPBottleneck(128, 128)
        self.down3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn_down3 = nn.BatchNorm2d(256)

        self.csp4 = CSPBottleneck(256, 256)  # เพิ่ม csp4

        self.head = nn.Conv2d(256, nc * 5 * num_predictions, kernel_size=1, stride=1, padding=0)
        self.bn_head = nn.BatchNorm2d(nc * 5 * num_predictions)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.csp1(x)
        x = self.down1(x)
        x = self.bn_down1(x)

        x = self.csp2(x)
        x = self.down2(x)
        x = self.bn_down2(x)

        x = self.csp3(x)
        x = self.down3(x)
        x = self.bn_down3(x)

        x = self.csp4(x)  # ใช้ csp4

        out = self.head(x)
        out = self.bn_head(out)
        out = out.view(x.size(0), self.num_predictions, 5, x.size(2), x.size(3))
        out = out.permute(0, 1, 3, 4, 2).contiguous()
        out = out.view(x.size(0), -1, 5)
        return out

# =============================================================================
# Dataset
# =============================================================================
class LicensePlateDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None, img_size=320):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.img_size = img_size
        self.image_files = sorted([f for f in os.listdir(images_dir) 
                                   if f.endswith(('.jpg', '.jpeg', '.png'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_file)
        image = Image.open(image_path).convert('RGB')

        label_file = os.path.splitext(image_file)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_file)
        targets = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, width, height = map(float, line.strip().split())
                    x_min = (x_center - width / 2) * self.img_size
                    y_min = (y_center - height / 2) * self.img_size
                    x_max = (x_center + width / 2) * self.img_size
                    y_max = (y_center + height / 2) * self.img_size
                    x_min = max(x_min, 0)
                    y_min = max(y_min, 0)
                    x_max = min(x_max, self.img_size)
                    y_max = min(y_max, self.img_size)
                    targets.append([x_min, y_min, x_max, y_max])
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(targets, dtype=torch.float32)

def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)

# =============================================================================
# Loss Function
# =============================================================================
from torchmetrics.detection.mean_ap import MeanAveragePrecision  # เพิ่มการนำเข้าที่นี่

class ComputeLoss:
    def __init__(self, model, anchors, num_classes=1, logger=None):
        self.model = model
        self.anchors = anchors
        self.num_classes = num_classes
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_loss = nn.MSELoss(reduction='none')
        self.logger = logger

    def __call__(self, predictions, targets, img_size, batch_idx, log_batches):
        total_box_loss = 0.0
        total_obj_loss = 0.0

        for img_idx, (pred, target) in enumerate(zip(predictions, targets)):
            num_preds = self.model.num_predictions
            grid_size = int(math.sqrt(pred.size(0) // num_preds))
            H, W = grid_size, grid_size

            pred = pred.view(num_preds, H, W, 5)

            obj_target = torch.zeros_like(pred[..., 4])
            box_target = torch.zeros_like(pred[..., :4])

            for t in target:
                x_min, y_min, x_max, y_max = t
                x_center = (x_min + x_max) / 2 / img_size * W
                y_center = (y_min + y_max) / 2 / img_size * H
                width = (x_max - x_min) / img_size * W
                height = (y_max - y_min) / img_size * H

                grid_x = min(int(x_center), W - 1)
                grid_y = min(int(y_center), H - 1)

                anchor_ws = self.anchors[:, 0]
                anchor_hs = self.anchors[:, 1]
                iou_scores = torch.stack([
                    self.iou((width, height), (aw, ah))
                    for aw, ah in zip(anchor_ws, anchor_hs)
                ])
                anchor_idx = torch.argmax(iou_scores).item()

                if anchor_idx >= num_preds:
                    continue

                obj_target[anchor_idx, grid_y, grid_x] = 1.0
                box_target[anchor_idx, grid_y, grid_x, 0] = (x_center - grid_x)
                box_target[anchor_idx, grid_y, grid_x, 1] = (y_center - grid_y)
                box_target[anchor_idx, grid_y, grid_x, 2] = torch.log(
                    torch.clamp(width / self.anchors[anchor_idx][0], min=1e-6)
                )
                box_target[anchor_idx, grid_y, grid_x, 3] = torch.log(
                    torch.clamp(height / self.anchors[anchor_idx][1], min=1e-6)
                )

                # Debug logging
                if self.logger and batch_idx in log_batches and img_idx < 1:
                    box_loss_tensor = self.mse_loss(
                        pred[anchor_idx, grid_y, grid_x, :4],
                        box_target[anchor_idx, grid_y, grid_x]
                    )
                    obj_loss_tensor = self.bce_loss(
                        pred[anchor_idx, grid_y, grid_x, 4],
                        obj_target[anchor_idx, grid_y, grid_x]
                    )
                    self.logger.debug(
                        f"Batch {batch_idx} - Image {img_idx} - Anchor {anchor_idx} - Grid ({grid_x}, {grid_y})"
                    )
                    self.logger.debug(
                        f"Pred Box: {pred[anchor_idx, grid_y, grid_x, :4].cpu().detach().tolist()}, "
                        f"Pred Conf: {pred[anchor_idx, grid_y, grid_x, 4].cpu().detach().item()}"
                    )
                    self.logger.debug(
                        f"Target Box: {box_target[anchor_idx, grid_y, grid_x].cpu().detach().tolist()}"
                    )
                    self.logger.debug(
                        f"Box Loss: {box_loss_tensor.sum().cpu().detach().item()}, "
                        f"Objectness Loss: {obj_loss_tensor.cpu().detach().item()}"
                    )

            box_loss = self.mse_loss(pred[..., :4], box_target).sum()
            obj_loss = self.bce_loss(pred[..., 4], obj_target).sum()

            total_box_loss += box_loss
            total_obj_loss += obj_loss

        total_loss = total_box_loss + total_obj_loss
        return total_loss / len(predictions), total_loss.item()

    @staticmethod
    def iou(box1, box2):
        w1, h1 = box1
        w2, h2 = box2
        intersection = min(w1, w2) * min(h1, h2)
        union = w1 * h1 + w2 * h2 - intersection
        return intersection / union if union != 0 else 0

# =============================================================================
# Visualization Function (for debugging/validation)
# =============================================================================
def visualize_predictions(image, predictions, targets=None, save_path=None):
    fig, ax = plt.subplots(1, figsize=(10, 10))
    image = image.cpu().permute(1, 2, 0).numpy()
    image = (image * 0.5 + 0.5) * 255.0  # Reverse normalization if applied
    image = image.astype(np.uint8)
    ax.imshow(image)

    # วาด bounding boxes ทำนาย (สีแดง)
    for box in predictions['boxes']:
        x_min, y_min, x_max, y_max = box.cpu().detach().tolist()
        rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                 linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # วาด bounding boxes จริง (สีเขียว)
    if targets is not None:
        for box in targets['boxes']:
            x_min, y_min, x_max, y_max = box.cpu().detach().tolist()
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(rect)

    plt.axis('off')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    else:
        plt.show()

# =============================================================================
# Training Function
# =============================================================================
def train_yolov7_plus(dataset_path, output_dir, device,
                      num_epochs=300,
                      train_batch_size=2,
                      val_batch_size=12,   # เพิ่ม batch size Validate เป็น 12
                      num_workers=4,       # เพิ่ม worker เป็น 4
                      learning_rate=1e-3,
                      early_stop_threshold=95.0,
                      early_stop_patience=5,
                      save_images_interval=10):
    """
    - เทรน YOLOv7TinySmallPlus
    - Early Stop ที่ mAP > early_stop_threshold% ติดกัน early_stop_patience epoch
    - บันทึกโมเดล best, last
    - เซฟภาพ Validate เมื่อ epoch==1 หรือทุกๆ save_images_interval
    - Vectorized decode ใน validate เพื่อความเร็ว
    """

    os.makedirs(output_dir, exist_ok=True)

    logger = setup_logger(name='train_yolov7_plus')
    logger.info(f"Log file created: training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    train_images_dir = os.path.join(dataset_path, 'images', 'train')
    train_labels_dir = os.path.join(dataset_path, 'labels', 'train')
    val_images_dir = os.path.join(dataset_path, 'images', 'val')
    val_labels_dir = os.path.join(dataset_path, 'labels', 'val')

    img_size = 320

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)  # Normalize to [-1, 1]
    ])
    train_dataset = LicensePlateDataset(train_images_dir, train_labels_dir, transform, img_size=img_size)
    val_dataset = LicensePlateDataset(val_images_dir, val_labels_dir, transform, img_size=img_size)

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    # สร้าง DataLoader
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers)

    # Anchors
    anchors = torch.tensor([
        [10.0, 13.0],
        [16.0, 30.0],
        [33.0, 23.0]
    ], device=device)

    # สร้างโมเดล
    model = YOLOv7TinySmallPlus(nc=1, num_predictions=3).to(device)
    # ตรวจสอบว่าโมเดลอยู่ใน float32
    # model.half()  # ถ้ามี ให้ลบออก

    compute_loss = ComputeLoss(model, anchors, num_classes=1, logger=logger)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    metric = MeanAveragePrecision(max_detection_thresholds=[100, 300, 500])
    metric.warn_on_many_detections = False

    scaler = torch.cuda.amp.GradScaler()

    best_mAP = 0.0
    best_epoch = 0  # เพิ่มตัวแปรนี้เพื่อเก็บ epoch ที่ได้ best mAP
    epochs_above_threshold = 0

    # สร้าง tqdm progress bar สำหรับ epochs
    epoch_bar = tqdm(range(1, num_epochs + 1), desc="Epochs", position=0)
    for epoch in epoch_bar:
        model.train()
        total_loss = 0
        logger.info(f"Epoch {epoch}/{num_epochs}")

        num_batches = len(train_loader)
        log_batches = set(random.sample(range(num_batches), min(5, num_batches)))

        # สร้าง tqdm progress bar สำหรับ batches
        batch_bar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=False, position=1)
        for batch_idx, (images, targets) in enumerate(batch_bar):
            images = torch.stack(images).to(device).float()  # แก้ไขบรรทัดนี้
            targets = [t.to(device) for t in targets]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                predictions = model(images)
                loss, _ = compute_loss(predictions, targets, img_size=img_size,
                                       batch_idx=batch_idx, log_batches=log_batches)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            torch.cuda.empty_cache()

            # อัปเดตบรรทัดของ batch progress bar
            if batch_idx in log_batches:
                batch_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch}/{num_epochs} Loss: {avg_loss:.4f}")

        # Validation
        mAP = validate_yolov7(
            model, 
            val_loader, 
            metric, 
            device, 
            epoch,
            output_dir,
            visualize=True,  # ถ้าอยากดูรูปเปลี่ยนเป็น True
            anchors=anchors,
            img_size=img_size,
            logger=logger,
            save_images_interval=save_images_interval
        )

        # Save last model
        last_ckpt_path = os.path.join(output_dir, 'yolov7_tiny_small_last.pth')
        torch.save(model.state_dict(), last_ckpt_path)

        # Save best model
        if mAP > best_mAP:
            best_mAP = mAP
            best_epoch = epoch  # อัปเดต epoch ที่ได้ best mAP
            best_ckpt_path = os.path.join(output_dir, 'yolov7_tiny_small_best.pth')
            torch.save(model.state_dict(), best_ckpt_path)
            logger.info(f"New best mAP: {mAP:.2f}% saved to {best_ckpt_path}")

        # เพิ่มการบันทึก Best mAP และ Epoch ที่เกี่ยวข้อง
        logger.info(f"Validation mAP: {mAP:.2f}% | Epoch: {epoch} | Best mAP: {best_mAP:.2f}% at epoch {best_epoch}")

        # อัปเดตคำอธิบายของ epoch progress bar
        epoch_bar.set_description(f"Epoch {epoch}/{num_epochs}: mAP: {mAP:.2f}%, Best mAP: {best_mAP:.2f}% at epoch {best_epoch}")

        # Early stopping
        if mAP > early_stop_threshold:
            epochs_above_threshold += 1
            logger.info(f"mAP {mAP:.2f}% > {early_stop_threshold}% for {epochs_above_threshold} consecutive epochs.")
            if epochs_above_threshold >= early_stop_patience:
                logger.info(f"Early stopping triggered after {epochs_above_threshold} consecutive epochs with mAP > {early_stop_threshold}%.")
                break
        else:
            epochs_above_threshold = 0

    logger.info(f"Training completed. Best mAP: {best_mAP:.2f}% at epoch {best_epoch}.")

# =============================================================================
# Validation Function (Vectorized Decode)
# =============================================================================
def validate_yolov7(model, val_loader, metric, device, epoch, output_dir,
                    iou_threshold=0.5, visualize=False, anchors=None,
                    img_size=320, logger=None, save_images_interval=10):
    import math
    import torch
    import random
    from tqdm import tqdm
    import numpy as np

    model.eval()
    metric.reset()
    images_to_save = 5
    saved = 0
    num_val_batches = len(val_loader)
    log_val_batches = set(random.sample(range(num_val_batches), min(5, num_val_batches)))
    save_debug_images = (epoch == 1) or (epoch % save_images_interval == 0)
    anchor_w = anchors[:, 0].view(1, -1, 1, 1).to(device)
    anchor_h = anchors[:, 1].view(1, -1, 1, 1).to(device)
    saved_images_indices = set(random.sample(range(len(val_loader.dataset)),
                                             min(images_to_save, len(val_loader.dataset))))
    with torch.no_grad():
        for idx, (images, targets) in enumerate(tqdm(val_loader, desc=f"Validating epoch {epoch}", leave=False)):
            images = torch.stack(images).to(device)
            formatted_targets = []
            for t in targets:
                formatted_targets.append({"boxes": t[:, :4].to(device), "labels": torch.zeros(t.size(0), dtype=torch.long, device=device)})
            predictions = model(images)
            batch_size = predictions.size(0)
            num_pred = model.num_predictions
            total_preds = predictions.size(1)
            grid_size = int(math.sqrt(total_preds // num_pred))
            stride = img_size / grid_size
            predictions_5d = predictions.view(batch_size, num_pred, grid_size, grid_size, 5)
            tx = torch.sigmoid(predictions_5d[...,0])
            ty = torch.sigmoid(predictions_5d[...,1])
            tw = torch.exp(predictions_5d[...,2])
            th = torch.exp(predictions_5d[...,3])
            confs = torch.sigmoid(predictions_5d[...,4])
            yv, xv = torch.meshgrid([torch.arange(grid_size, device=device), torch.arange(grid_size, device=device)], indexing='ij')
            xv = xv.float().unsqueeze(0).unsqueeze(0)
            yv = yv.float().unsqueeze(0).unsqueeze(0)
            x_center = (tx + xv) * stride
            y_center = (ty + yv) * stride
            w = tw * anchor_w * stride
            h = th * anchor_h * stride
            x_min = (x_center - w/2).clamp(min=0, max=img_size)
            y_min = (y_center - h/2).clamp(min=0, max=img_size)
            x_max = (x_center + w/2).clamp(min=0, max=img_size)
            y_max = (y_center + h/2).clamp(min=0, max=img_size)
            boxes_concat = torch.stack([x_min, y_min, x_max, y_max, confs], dim=-1).view(batch_size, -1, 5)
            formatted_preds = []
            for b in range(batch_size):
                bboxes = boxes_concat[b,:,:4]
                bscores = boxes_concat[b,:,4]
                conf_mask = bscores>=0.5
                bboxes = bboxes[conf_mask]
                bscores = bscores[conf_mask]
                if bboxes.numel()==0:
                    formatted_preds.append({"boxes":torch.empty((0,4),device=device),
                                            "scores":torch.empty((0,),device=device),
                                            "labels":torch.empty((0,),dtype=torch.long,device=device)})
                    continue
                keep = nms(bboxes,bscores,iou_threshold)
                b_kept = bboxes[keep]
                s_kept = bscores[keep]
                l_kept = torch.zeros(len(b_kept),dtype=torch.long,device=device)
                formatted_preds.append({"boxes":b_kept,"scores":s_kept,"labels":l_kept})
            if logger and idx in log_val_batches:
                logger.debug(f"Val batch {idx}, #pred {sum(len(p['boxes']) for p in formatted_preds)}")
            metric.update(formatted_preds, formatted_targets)
            if save_debug_images and visualize:
                for i in range(batch_size):
                    if saved>=images_to_save: break
                    ds_idx= idx * val_loader.batch_size + i
                    if ds_idx in saved_images_indices:
                        im=images[i]
                        tg= formatted_targets[i]["boxes"]
                        pr= formatted_preds[i]["boxes"]
                        sp= os.path.join(output_dir,"validation_samples",f"epoch_{epoch}_sample_{saved+1}.png")
                        os.makedirs(os.path.dirname(sp),exist_ok=True)
                        visualize_predictions(im,{"boxes":pr},{"boxes":tg}, sp)
                        saved+=1
    results= metric.compute()
    mAP= results["map"].item()*100
    if logger:
        logger.info(f"Validation mAP: {mAP:.2f}% | Epoch: {epoch}")
    else:
        print(f"Validation mAP: {mAP:.2f}% | Epoch: {epoch}")
    return mAP

# =============================================================================
# Main Function
# =============================================================================
def main():
    dataset_path = 'thai_license_plate_dataset_for_yolov7'
    output_dir = 'trained_yolov7_tiny_small_plus'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # เรียก train โดยกำหนด val_batch_size=12, num_workers=4
    # และเราจะทำ Vectorized Decode ใน validate ให้เร็วขึ้น
    train_yolov7_plus(
        dataset_path,
        output_dir,
        device,
        num_epochs=1000,               # ลดหรือเพิ่มได้
        train_batch_size=2,
        val_batch_size=12,           # เพิ่ม batch validate
        num_workers=4,               # worker ช่วยโหลด data
        learning_rate=1e-3,
        early_stop_threshold=95.0,
        early_stop_patience=5,
        save_images_interval=10       # เซฟรูปทุก 10 epoch + epoch 1
    )

if __name__ == "__main__":
    main()
