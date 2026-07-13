import torch
from src.models.pillar_encoder import PillarEncoder


def test_pillar_encoder_shape():
    encoder = PillarEncoder(
        x_range=(0.0, 48.0),
        y_range=(-16.0, 16.0),
        z_range=(-3.0, 1.0),
        voxel_size=(0.25, 0.25),
        max_points_per_pillar=20,
        out_channels=64,
    )

    # Grid size should be H=128, W=192
    assert encoder.nx == 192
    assert encoder.ny == 128

    # Create dummy points: 1000 points of (x, y, z, intensity)
    # Some inside, some outside bounds
    points = torch.zeros((1000, 4))
    points[:, 0] = torch.linspace(-5.0, 50.0, 1000)  # x: -5 to 50
    points[:, 1] = torch.linspace(-20.0, 20.0, 1000)  # y: -20 to 20
    points[:, 2] = torch.linspace(-4.0, 2.0, 1000)  # z: -4 to 2
    points[:, 3] = torch.rand(1000)  # intensity

    # Forward pass
    pseudo_image = encoder(points)

    assert isinstance(pseudo_image, torch.Tensor)
    # Batch size 1, channels 64, H=128, W=192
    assert pseudo_image.shape == (1, 64, 128, 192)


def test_pillar_encoder_empty_input():
    encoder = PillarEncoder()
    points = torch.zeros((0, 4))
    pseudo_image = encoder(points)
    assert pseudo_image.shape == (1, 64, 128, 192)
    assert (pseudo_image == 0).all()
