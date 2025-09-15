# Air Vortex Cannon for Small Drone Defense – Performance Calculator

This repository contains a Python-based performance calculator supporting the paper *"Air Vortex Cannon for Small Drone Defense: Exploiting Toroidal Expansion for Consumer UAV Suppression"*. The tool calculates optimal engagement parameters and visualizes vortex ring trajectories for small drone suppression scenarios.

## Quick Start

### 1. Fix Import Issues (Required First Step)
```bash
# Run this first to fix Python import compatibility
python import_fix.py
```

### 2. Calculate Engagement
```bash
python scripts/engage.py --target-x 25 --target-y 10 --target-z 15 --drone-size small
```

### 3. Visualize Shot
```bash
python scripts/visualize.py --target-x 25 --target-y 10 --target-z 15 --output figs/shot.png
```

## Setup Instructions

### Initial Setup
1. **Clone the repository**
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Fix imports**: `python import_fix.py` (This fixes Python module import issues)
4. **Test installation**: `python scripts/engage.py --target-x 30 --target-y 0 --target-z 15 --drone-size small`

**Note**: The `import_fix.py` script resolves Python relative import issues that may occur when running the scripts directly. Run this once after cloning the repository.

## Repository Structure

```
Air-Vortex-Cannon-Performance-Calculator/
├── README.md
├── requirements.txt
├── import_fix.py           # Import compatibility fix (run first)
├── .gitignore             # Git ignore rules
├── src/
│   ├── cannon.py           # Cannon physics and configuration
│   ├── vortex_ring.py      # Vortex ring formation and propagation
│   └── engagement.py       # Target engagement calculations
├── config/
│   └── cannon_specs.yaml  # Cannon specifications
├── scripts/
│   ├── engage.py          # Calculate optimal shot parameters
│   └── visualize.py       # Generate 3D visualization
├── examples/
│   ├── single_drone.py    # Example: single target engagement
│   ├── multiple_targets.py # Example: multiple drone engagement
│   └── parametric_study.py # Example: range vs elevation study
├── results/               # Auto-generated analysis results (.txt files)
└── figs/
    └── (generated plots)
```

## Cannon Configuration (config/cannon_specs.yaml)

```yaml
cannon:
  barrel_length: 2.0
  barrel_diameter: 0.5
  max_chamber_pressure: 300000  # 3 bar - more realistic for effective vortex generation
  
vortex_ring:
  formation_number: 4.0
  initial_velocity: 80           # Higher velocity for realistic energy transfer
  effective_range: 45            # Reduced to match physics limitations
  
environment:
  air_density: 1.225
  wind_sensitivity: 3.0          # m/s - performance degrades above this
  accuracy_degradation: 0.02     # per meter beyond 20m range
  
drone_models:
  small: 
    mass: 0.5
    size: 0.3
    vulnerability: 0.65          # More realistic - accounts for partial hits, recovery
    
  medium: 
    mass: 2.0
    size: 0.6
    vulnerability: 0.45          # Harder to disrupt larger platforms
    
  large: 
    mass: 8.0
    size: 1.2
    vulnerability: 0.1           # Very limited effectiveness against large drones

# Performance limitations for realistic modeling
limitations:
  max_effective_range: 45        # meters
  min_kill_probability: 0.3     # threshold for "effective"
  wind_degradation_factor: 0.1  # per m/s above wind_sensitivity
  range_energy_decay: 0.95      # energy retention per meter traveled
```

## Usage Examples

### 1. Single Target Engagement
```bash
# Engage small drone at 30m range, 15m altitude
python scripts/engage.py --target-x 30 --target-y 0 --target-z 15 --drone-size small

# Expected Output:
# [SUCCESS] ENGAGEMENT FEASIBLE
# Elevation: 23.43 degrees
# Kill probability: 0.904
# Results auto-saved to: results/engagement_single_YYYYMMDD_HHMMSS.txt
```

### 2. Visualization
```bash
# Generate 3D plot showing cannon, trajectory, and target
python scripts/visualize.py --target-x 25 --target-y 15 --target-z 18 --output figs/engagement.png
```

### 3. Multiple Targets
```bash
# Engage multiple drones with optimal sequencing
python examples/multiple_targets.py
# Results auto-saved to: results/multiple_targets_analysis.txt
```

## Usage Examples for Paper Validation

The following examples reproduce key results from the research paper:

### System Effective Range (Section 5.2)
```bash
# Close range effectiveness - optimal performance zone
python scripts/engage.py --target-x 15 --target-y 0 --target-z 10 --drone-size small
# Expected: P_kill > 0.90, demonstrates high close-range effectiveness
```

### Maximum Range Analysis (Section 5.3)
```bash
# Maximum effective range testing
python scripts/engage.py --target-x 45 --target-y 0 --target-z 25 --drone-size small
# Expected: P_kill ≈ 0.88, shows effective range limits
```

### Target Size Limitations (Section 5.3)
```bash
# Small drone (effective)
python scripts/engage.py --target-x 30 --target-y 0 --target-z 15 --drone-size small
# Expected: P_kill > 0.88

# Medium drone (limited effectiveness)
python scripts/engage.py --target-x 30 --target-y 0 --target-z 15 --drone-size medium  
# Expected: P_kill = 0.000 (demonstrates size limitations)

# Large drone (ineffective)
python scripts/engage.py --target-x 30 --target-y 0 --target-z 15 --drone-size large
# Expected: P_kill = 0.000 (confirms scope boundaries)
```

### Moving Target Interception (Section 4.3)
```bash
# Fast-moving target scenario
python scripts/engage.py --target-x 35 --target-y 0 --target-z 18 --drone-size medium \
                        --velocity-x -8 --velocity-y 0 --velocity-z 0
# Demonstrates intercept calculation and movement limitations
```

### Engagement Envelope Analysis (Figure 7)
```bash
# Generate complete engagement envelope data
python scripts/engage.py --envelope-analysis --drone-type small
# Expected: Max effective range: 80m, Optimal range: 45m, Max kill prob: 0.924
# Results saved to: results/engagement_envelope_small_YYYYMMDD_HHMMSS.txt
```

### Comprehensive Analysis Generation
```bash
# Generate all paper analysis data (auto-saves to results/ directory)
python examples/single_drone.py      # Single target scenarios
python examples/multiple_targets.py  # Multi-target engagement sequences  
python examples/parametric_study.py  # Design optimization analysis
```

## Visual Results

### Engagement Envelope Analysis
![Engagement Envelope](figs/engagement_envelope_small.png)
*Small drone engagement envelope showing kill probability (left) and hit probability (right) across range and elevation. Maximum effective range: 80m with optimal performance at 20m @ 70°.*

### 3D Engagement Visualization
![3D Engagement](figs/engagement_3d_example.png)
*3D visualization of successful engagement showing cannon position, vortex ring trajectory, and target location with color-coded range zones.*

### Trajectory Physics Analysis
![Trajectory Analysis](figs/trajectory_analysis.png)
*Vortex ring physics showing velocity decay, diameter expansion, kinetic energy reduction, and velocity-diameter relationship validating theoretical models.*

## Key Features

### Realistic Performance Modeling
- **Effective target range**: Small consumer drones (≤0.5kg) within 45m
- **Kill probability**: 88-92% within optimal envelope
- **Clear limitations**: Explicitly models ineffectiveness against larger targets
- **Auto-save results**: All analyses automatically saved to timestamped .txt files

### Deployment-Focused Design
- **Portable system**: 0.5m bore, standard 1-bar pressure
- **Rapid setup**: <10 minute deployment capability
- **Standard equipment**: Compatible with industrial air compressors

### Comprehensive Analysis
- **Single target optimization**: Elevation/azimuth calculation
- **Multi-target sequencing**: Optimal engagement order
- **Parametric studies**: Design parameter optimization
- **Envelope mapping**: Complete effectiveness boundaries

## System Capabilities and Limitations

### Effective Against:
- **Small consumer drones**: 0.3-0.5kg mass, ≤0.3m size
- **Stationary and slow-moving targets**: <8 m/s
- **Close to medium range**: 15-45m optimal effectiveness
- **Single or small sequential groups**: 1-3 targets

### Limited Effectiveness:
- **Medium drones**: 1-2kg mass shows poor engagement success
- **Fast targets**: >8 m/s velocity creates engagement challenges
- **Extended range**: >50m shows significant performance degradation
- **Large simultaneous swarms**: Sequential engagement limits rapid multi-target scenarios

### Not Effective Against:
- **Large drones**: >5kg mass (Shahed-136 style platforms)
- **Robust military platforms**: Designed for combat environments
- **High-speed targets**: >15 m/s approach speeds

## Dependencies
```txt
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
pyyaml>=6.0
```

## Troubleshooting

### Import Errors
If you encounter `ImportError: attempted relative import with no known parent package`:
```bash
python import_fix.py
```

### Results Not Saving
Results are automatically saved to the `results/` directory. If you don't want auto-save:
```bash
python scripts/engage.py --target-x 30 --target-y 0 --target-z 15 --drone-size small --quiet
```

## Paper Support

This tool validates the theoretical models presented in "Air Vortex Cannon for Small Drone Defense" through computational analysis. The results demonstrate:

- **Specialized effectiveness** against small consumer drone threats
- **Clear operational boundaries** defining suitable target categories  
- **Realistic deployment requirements** for rapid-response scenarios
- **Quantified performance metrics** supporting paper claims

All results are reproducible using the commands listed above, with automatic result logging for research validation and documentation.