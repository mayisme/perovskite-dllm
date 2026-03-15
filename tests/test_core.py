import torch
import numpy as np
from models.egnn import EGNNModel, EGNNLayer
from models.physics_loss import rdf_loss

def test_egnn_equivariance():
    """Verify that EGNNModel is equivariant to rotations."""
    model = EGNNModel(hidden_dim=32, n_layers=2)
    model.eval()
    
    B, N = 2, 5
    x = torch.randn(B, N, 3)
    t = torch.randint(0, 100, (B,))
    atom_types = torch.randint(0, 5, (B, N))
    bg = torch.randn(B)
    lat = torch.eye(3).unsqueeze(0).repeat(B, 1, 1)
    
    # Rotation matrix
    theta = torch.tensor(np.pi / 4)
    rot = torch.tensor([
        [torch.cos(theta), -torch.sin(theta), 0],
        [torch.sin(theta), torch.cos(theta), 0],
        [0, 0, 1]
    ])
    
    # Forward original
    out_x1, out_l1 = model(x, t, atom_types, bg, lat)
    
    # Forward rotated
    x_rot = torch.matmul(x, rot)
    out_x2, out_l2 = model(x_rot, t, atom_types, bg, lat)
    
    # Check out_x2 ~= out_x1 * rot
    out_x1_rot = torch.matmul(out_x1, rot)
    
    diff = torch.abs(out_x2 - out_x1_rot).max().item()
    print(f"Equivariance Max Diff: {diff}")
    assert diff < 1e-4

def test_pbc_dist():
    """Verify PBC distance calculation."""
    coords = torch.tensor([[[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]]]) # Dist is 0.2 across boundary
    lattice = torch.eye(3).unsqueeze(0) * 10.0 # 10A box
    
    # Expected distance: 0.2 frac * 10A = 2.0A
    loss = rdf_loss(coords, lattice, target_peaks=[2.0], sigma=0.1)
    # If MIC works, loss should be high (near -1)
    print(f"PBC RDF Loss: {loss.item()}")
    assert loss.item() < -0.5

if __name__ == "__main__":
    test_egnn_equivariance()
    test_pbc_dist()
    print("All tests passed!")
