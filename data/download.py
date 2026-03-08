"""Download perovskite data from Materials Project."""
import os
from mp_api.client import MPRester
from dotenv import load_dotenv
import json

load_dotenv()

API_KEY = os.getenv("MP_API_KEY")

def download_perovskites(max_structures=5000):
    """Download ABO3 perovskite structures."""
    with MPRester(API_KEY) as mpr:
        # Query for perovskites with specific criteria
        docs = mpr.materials.summary.search(
            num_elements=(3, 3),
            fields=["material_id", "formula_pretty", "structure", 
                   "band_gap", "formation_energy_per_atom", "symmetry",
                   "energy_above_hull"]
        )
        
        perovskites = []
        for doc in docs:
            # Filter: stable structures only
            if doc.energy_above_hull and doc.energy_above_hull > 0.1:
                continue
            
            struct = doc.structure
            if len(struct) < 5:
                continue
                
            perovskites.append({
                "material_id": doc.material_id,
                "formula": doc.formula_pretty,
                "structure": struct.as_dict(),
                "band_gap": doc.band_gap,
                "formation_energy": doc.formation_energy_per_atom,
                "space_group": doc.symmetry.symbol if doc.symmetry else None,
                "n_atoms": len(struct)
            })
            
            if len(perovskites) >= max_structures:
                break
        
        return perovskites

if __name__ == "__main__":
    print("Downloading ALL stable perovskite structures...")
    print("This may take 5-10 minutes...")
    data = download_perovskites(max_structures=20000)  # Set high limit
    
    os.makedirs("data/raw", exist_ok=True)
    with open("data/raw/perovskites_full.json", "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Downloaded {len(data)} stable structures to data/raw/perovskites_full.json")
