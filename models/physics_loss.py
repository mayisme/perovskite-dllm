"""Physics-informed loss functions."""
import torch


def rdf_loss(coords, target_peaks=[1.0, 2.0], sigma=0.1):
    """
    Radial Distribution Function loss.
    Penalizes unrealistic atom distances.
    
    coords: (B, N, 3) fractional coordinates
    """
    B, N = coords.shape[:2]
    
    # Pairwise distances
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # (B, N, N, 3)
    dist = torch.norm(diff, dim=-1)  # (B, N, N)
    
    # Remove self-distances
    mask = ~torch.eye(N, dtype=torch.bool, device=coords.device)
    dist = dist[:, mask].reshape(B, -1)
    
    # Compute RDF histogram
    loss = 0
    for peak in target_peaks:
        # Gaussian penalty around expected peaks
        deviation = torch.abs(dist - peak)
        loss += torch.exp(-deviation**2 / (2 * sigma**2)).mean()
    
    return -loss  # Negative because we want to maximize overlap


def pauli_repulsion_loss(coords, min_dist=0.5):
    """
    Penalize atoms that are too close (Pauli exclusion).
    
    coords: (B, N, 3)
    """
    B, N = coords.shape[:2]
    
    diff = coords.unsqueeze(2) - coords.unsqueeze(1)
    dist = torch.norm(diff, dim=-1)
    
    # Mask diagonal
    mask = ~torch.eye(N, dtype=torch.bool, device=coords.device)
    dist = dist[:, mask].reshape(B, -1)
    
    # Penalty for distances below threshold
    violation = torch.relu(min_dist - dist)
    
    return violation.mean()


def energy_penalty(coords, bandgap, target_bandgap):
    """
    Simple energy-based penalty.
    In real implementation, would call DFT or use learned energy model.
    """
    # Placeholder: penalize deviation from target bandgap
    return torch.abs(bandgap - target_bandgap).mean()


def combined_physics_loss(coords, bandgap, target_bandgap, weights=None):
    """Combine all physics constraints."""
    if weights is None:
        weights = {"rdf": 0.1, "pauli": 1.0, "energy": 0.01}
    
    loss = 0
    loss += weights["rdf"] * rdf_loss(coords)
    loss += weights["pauli"] * pauli_repulsion_loss(coords)
    loss += weights["energy"] * energy_penalty(bandgap, bandgap, target_bandgap)
    
    return loss
