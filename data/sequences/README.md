# Temporal Sequence Dataset

Place `.npz` files in this directory.

Each `.npz` should include:

- `sequence`: shape `[20, 3]` (Open, Closed, Half probabilities per timestep)
- `label`: integer class id
  - `0`: Awake
  - `1`: Drowsy
  - `2`: Distracted
  - `3`: Microsleep

Alternative keys `x` and `y` are also supported by the training loader.
