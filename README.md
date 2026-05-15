# py_popratio

A Python implementation of atomic population ratio calculations for neutral species (e.g., C I), inspired by classical PopRatio codes.

---

## 🔬 Features

- Multi-level atomic population solver
- Includes:
  - Collisions (H, e⁻, H₂, etc.)
  - CMB radiation (redshift dependent)
  - UV radiation fields:
    - Draine (1978)
    - PopRatio / Gondhalekar
    - Khaire & Srianand (2019) UV background
  - Fluorescence / UV pumping

---

## 🚀 Quick Start

```python
from atomic_data import parse_atom_file
from solver import compute_population_ratios

atom = parse_atom_file("CI.dat")

result = compute_population_ratios(
    atom,
    nH_grid=[1, 10, 100],
    T_gas=100,
    z=0.0,
    uv_field="draine",
    uv_scale=1.0,
)

print(result["ratios_to_ground"])
