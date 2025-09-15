# Air Vortex Cannon for Drone Defense – Performance Calculator

This repository contains a Python-based performance calculator supporting the paper *"Air Vortex Cannon for Drone Defense: A Clean Alternative to Conventional Air Defense Systems"*. The tool calculates optimal engagement parameters and visualizes vortex ring trajectories for drone suppression scenarios.

## Quick Start

### Calculate Engagement
```bash
python scripts/engage.py --target-x 25 --target-y 10 --target-z 15 --drone-size small
```

### Visualize Shot
```bash
python scripts/visualize.py --target-x 25 --target-y 10 --target-z 15 --output figs/shot.png
```

## Repository Structure

```
Air-Vortex-Cannon-Performance-Calculator/
├── README.md
├── requirements.txt
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
└── figs/
    └── (generated plots)
```

## Cannon Configuration (config/cannon_specs.yaml)

```yaml
# Vortex Cannon Specifications
cannon:
  barrel_length: 2.0          # meters
  barrel_diameter: 0.5        # meters
  max_chamber_pressure: 100000 # Pa
  max_elevation: 85           # degrees
  max_traverse: 360           # degrees
  
vortex_ring:
  formation_number: 4.0       # optimal stroke-to-diameter ratio
  initial_velocity: 50        # m/s
  ring_diameter: 0.3          # meters
  effective_range: 50         # meters
  decay_coefficient: 0.05     # per meter

drone_models:
  small:      # < 1kg quadcopter
    mass: 0.5
    size: 0.3
    vulnerability: 0.9
  medium:     # 1-5kg quadcopter  
    mass: 2.0
    size: 0.6
    vulnerability: 0.7
  large:      # > 5kg fixed-wing
    mass: 8.0
    size: 1.2
    vulnerability: 0.5
```

## Usage Examples

### 1. Single Target Engagement
```bash
# Engage small drone at 30m range, 20m altitude
python scripts/engage.py --target-x 30 --target-y 0 --target-z 20 --drone-size small

# Output:
# Optimal elevation: 33.7°
# Optimal azimuth: 0.0°
# Time to target: 0.72s
# Vortex ring diameter at impact: 0.45m
# Kill probability: 0.84
```

### 2. Visualization
```bash
# Generate 3D plot showing cannon, trajectory, and target
python scripts/visualize.py --target-x 25 --target-y 15 --target-z 18 --output figs/engagement.png
```

``` bash
# Stationary target
python scripts/visualize.py --target-x 30 --target-y 10 --target-z 15  --drone-size small --output figs/engagement.png
```
``` bash
# Moving target with velocity vector
python scripts/visualize.py --target-x 35 --target-y 0 --target-z 18 --drone-size medium --velocity-x -5 --velocity-y 2 --output figs/moving_target.png
```

### 3. Multiple Targets
```bash
# Engage multiple drones with optimal sequencing
python examples/multiple_targets.py
```

## Usage Examples for Paper

The following examples reproduce key results and figures from the research paper:

### Close Range Effectiveness (Section 5.2)
```bash
# Test optimal range performance
python scripts/engage.py --target-x 15 --target-y 0 --target-z 10 --drone-size small
# Expected: P_kill > 0.90, demonstrates high effectiveness at close range
```

### Maximum Range Analysis (Section 5.3)
```bash
# Test maximum effective range
python scripts/engage.py --target-x 50 --target-y 0 --target-z 25 --drone-size small
# Expected: P_kill ≈ 0.30, shows range limitations
```

### Moving Target Interception (Section 4.3)
```bash
# Fast-moving target scenario
python scripts/engage.py --target-x 35 --target-y 0 --target-z 18 --drone-size medium \
                        --velocity-x -8 --velocity-y 0 --velocity-z 0
# Demonstrates intercept calculation and lead angles
```

### Drone Size Sensitivity (Section 4.4)
```bash
# Compare effectiveness against different drone sizes
python scripts/engage.py --target-x 30 --target-y 0 --target-z 15 --drone-size small
python scripts/engage.py --target-x 30 --target-y 0 --target-z 15 --drone-size medium  
python scripts/engage.py --target-x 30 --target-y 0 --target-z 15 --drone-size large
# Shows decreasing effectiveness with larger, more robust targets
```

### Engagement Envelope Mapping (Figure 7)
```bash
# Generate engagement envelope data
python scripts/engage.py --envelope-analysis --drone-type small
python scripts/engage.py --envelope-analysis --drone-type medium
python scripts/engage.py --envelope-analysis --drone-type large
# Produces data for engagement envelope plots in paper
```

### Multi-Target Swarm Scenario (Section 6.1)
```bash
# Create swarm scenario file: examples/paper_swarm.yaml
# Then run:
python scripts/engage.py --targets-file examples/paper_swarm.yaml
# Demonstrates sequential engagement capability
```

### Elevation Angle Optimization (Section 4.2)
```bash
# Test various elevations at fixed range
python scripts/engage.py --target-x 25 --target-y 0 --target-z 5  --drone-size small  # Low angle
python scripts/engage.py --target-x 25 --target-y 0 --target-z 15 --drone-size small  # Medium angle
python scripts/engage.py --target-x 25 --target-y 0 --target-z 25 --drone-size small  # High angle
# Shows optimal elevation angles for different scenarios
```

### System Performance Validation
```bash
# Reproduce key performance metrics from paper
python scripts/engage.py --target-x 20 --target-y 5 --target-z 12 --drone-size small --verbose
# Detailed output for validation of theoretical calculations
```

## Key Features

### Automatic Optimization
- **Trajectory calculation**: Optimal elevation and azimuth angles
- **Timing analysis**: Time-to-target and engagement sequencing  
- **Kill probability**: Based on vortex ring size and drone vulnerability
- **Range limitations**: Accounts for vortex ring decay

### Configurable Parameters
- **Cannon specifications**: Easily modify barrel dimensions, pressure, etc.
- **Drone characteristics**: Different size/vulnerability classes
- **Environmental factors**: Air density, wind effects (future)

### Visualization Output
- **3D engagement plot**: Cannon position, vortex trajectory, target location
- **Trajectory analysis**: Range vs elevation plots
- **Coverage maps**: Effective engagement envelope

## Core Calculations

### Vortex Ring Physics
```python
# src/vortex_ring.py
class VortexRing:
    def __init__(self, initial_velocity, diameter, formation_number):
        self.velocity = initial_velocity
        self.diameter = diameter
        self.formation_number = formation_number
    
    def trajectory(self, time):
        """Calculate vortex ring position at given time"""
        
    def diameter_at_range(self, distance):
        """Calculate ring diameter expansion with distance"""
        
    def kill_probability(self, target_size, vulnerability):
        """Calculate probability of drone disruption"""
```

### Engagement Calculation
```python
# src/engagement.py
class EngagementCalculator:
    def optimal_firing_solution(self, target_pos, drone_type):
        """Calculate best elevation/azimuth for target"""
        
    def multiple_target_sequence(self, target_list):
        """Optimize engagement sequence for multiple drones"""
```

## Example Outputs

### Engagement Analysis
```
Target: Small drone at (30, 15, 20)m
Optimal firing solution:
  - Elevation: 28.3°
  - Azimuth: 26.6°  
  - Flight time: 0.89s
  - Ring diameter at impact: 0.52m
  - Kill probability: 0.78
```

### Visualization
- 3D plot showing cannon, ballistic trajectory, vortex ring expansion
- Color-coded kill probability zones
- Range and elevation coverage maps

## Dependencies
```txt
numpy>=1.21.0
matplotlib>=3.5.0
scipy>=1.7.0
pyyaml>=6.0
```

## Paper Support

This toolkit generates the computational results and figures for:
- **Section 4**: Vortex ring trajectory calculations and drone interaction analysis
- **Section 5**: System performance characteristics and operational effectiveness  
- **Section 6**: Multi-target engagement scenarios and deployment applications
- **Figures 5-8**: Performance plots, engagement envelopes, and effectiveness visualization

The tool validates theoretical models through computational analysis while providing practical engagement calculations for defense applications. All paper results are reproducible using the commands listed in the "Usage Examples for Paper" section above.

## Citation

If you use this code or reference the associated research, please cite:

```
[Your paper citation format will go here]
```