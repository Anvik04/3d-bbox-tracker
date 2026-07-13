import os
import sys

import torch
import torch.optim as optim

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.data.kitti_dataset import KITTIDataset
from src.models.detector import CameraLiDARDetector
from src.training.losses import CameraLiDARLoss


def main():
    # Setup paths
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_dir = os.path.join(repo_root, "data", "fixtures")
    checkpoint_dir = os.path.join(repo_root, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Initializing dataset...")
    dataset = KITTIDataset(data_dir=data_dir)

    print("Initializing model...")
    # Run on CPU by default or GPU if available (we will use CPU for training verification)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CameraLiDARDetector().to(device)
    criterion = CameraLiDARLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 15
    print("Starting training loop on synthetic fixtures...")
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_reg_loss = 0.0

        for sample in dataset:
            points = sample["points"].to(device)
            image = sample["image"].to(device)
            calib = sample["calib"]
            gt_boxes_3d = sample["gt_boxes_3d"].to(device)

            optimizer.zero_grad()

            # Forward pass
            cls_logits, reg_preds = model(points, image, calib)

            # Compute loss
            loss, cls_loss, reg_loss = criterion(cls_logits, reg_preds, gt_boxes_3d)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cls_loss += cls_loss.item()
            epoch_reg_loss += reg_loss.item()

        avg_loss = epoch_loss / len(dataset)
        avg_cls = epoch_cls_loss / len(dataset)
        avg_reg = epoch_reg_loss / len(dataset)
        print(
            f"Epoch [{epoch+1:02d}/{num_epochs:02d}] - Loss: {avg_loss:.6f} (Cls: {avg_cls:.6f}, Reg: {avg_reg:.6f})"
        )

    # Save checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "detector.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )
    print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
