import os
import sys

import numpy as np
from PIL import Image

# Ensure project root is in Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.kitti_dataset import KITTIDataset
from src.viz.visualize import draw_projected_boxes_2d


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(repo_root, "data", "fixtures")
    output_dir = os.path.join(repo_root, "outputs", "viz", "ground_truth")
    os.makedirs(output_dir, exist_ok=True)

    print("Loading dataset...")
    dataset = KITTIDataset(data_dir=data_dir)

    print("Generating ground truth 3D visualizations...")
    for idx, sample in enumerate(dataset):
        file_id = sample["file_id"]
        image = sample["image"]
        calib = sample["calib"]
        gt_boxes = sample["gt_boxes_3d"].numpy()

        # Convert image tensor back to PIL Image
        img_pil = Image.fromarray(
            (image.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
        )

        # Draw the perfect ground truth boxes in red
        # Let's specify red color (255, 0, 0)
        red_colors = [(255, 0, 0)] * len(gt_boxes)
        img_viz = draw_projected_boxes_2d(
            img_pil,
            calib,
            gt_boxes,
            track_ids=list(range(1, len(gt_boxes) + 1)),
            colors=red_colors,
        )

        # Save visualization
        viz_path = os.path.join(output_dir, f"{file_id}.png")
        img_viz.save(viz_path)
        print(f"Saved ground truth visualization: {viz_path}")

    print("\nVisualization generation completed!")
    print(f"Outputs are saved in: {output_dir}")


if __name__ == "__main__":
    main()
