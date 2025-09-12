# Air Vortex Cannon for Drone Defense – Simulation Toolkit

This repository contains a Python-based simulation toolkit supporting the paper *"Air Vortex Cannon for Drone Defense: A Clean Alternative to Conventional Air Defense Systems"*. The code models vortex ring formation, propagation, and interaction with drone swarms for theoretical analysis and engagement scenario visualization.

## Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd vortex-cannon-sim
    ```

2.  Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### Run a Simulation
To calculate an optimal shot against a drone swarm from a CSV file:
```bash
python scripts/engage.py --drones examples/drones_rect_5x5.csv --config config/default.txt
```

## Generate a 3D Visualization
To create a 3D plot of the cannon, vortex ring trajectory, and drone positions:
```bash
python scripts/visualise.py --drones examples/drones_v_10.csv --output figs/engagement.png
```

## Repository Structure
Air-Vortex-Cannon-for-Drone-Defense-Simulation-Toolkit
```
├── README.md
├── requirements.txt
├── src
│   ├── cannon.py
│   ├── ballistics.py
│   └── target.py
├── config
│   └── default.txt
├── scripts
│   ├── engage.py
│   └── visualise.py
├── examples
│   ├── drones_rect_5x5.csv
│   └── drones_v_10.csv
└── figs
```

