# Air Vortex Cannon for Drone Defense - Simulation Toolkit

A comprehensive physics-based simulation toolkit for analyzing vortex cannon systems designed for drone defense applications. This toolkit provides both single-cannon and multi-cannon array analysis capabilities with realistic performance modeling.

## Key Findings Summary

- **Primary Effectiveness:** Small consumer drones (≤0.3m) with kill probabilities 0.7-0.9
- **Range Optimization:** 15-35m optimal engagement envelope  
- **Multi-Cannon Benefits:** 3-4x coverage improvement, enables medium drone engagement
- **Vehicle Integration:** Compact 2x2 arrays suitable for mobile platforms
- **Physical Limitations:** Large drone engagement requires alternative approaches

## Repository Structure

```
├── config/
│   └── cannon_specs.yaml          # System configuration and drone models
├── src/                            # Core physics modules
│   ├── cannon.py                   # Vortex cannon hardware model
│   ├── vortex_ring.py             # Ring physics and Monte Carlo analysis  
│   ├── engagement.py              # Target engagement calculations
│   └── multi_cannon_array.py     # Coordinated array systems
├── scripts/
│   ├── engage.py                  # Single engagement simulation
│   └── visualize.py              # Visualization suite (needs publication update)
├── examples/                      # Complete analysis scripts
│   ├── single_drone.py           # Individual target analysis
│   ├── multiple_targets.py       # Swarm engagement scenarios
│   ├── parametric_study.py       # Design optimization
│   └── multi_cannon_analysis.py  # Array performance comparison
├── results/                       # Analysis outputs
│   ├── single_drone_analysis.txt
│   ├── multi_cannon_analysis.txt
│   ├── parametric_analysis.txt
│   └── comprehensive_analysis_*.txt
├── figs/                          # Generated visualizations
│   ├── paper_figures/            # Publication-ready figures
│   ├── comparisons/              # Performance comparisons
│   └── arrays/                   # Multi-cannon visualizations
└── run_complete_analysis.py      # Full analysis suite
```

## Quick Start

### System Requirements

```bash
# Python 3.8+ with required packages
pip install -r requirements.txt
```

### Basic Analysis

```bash
# Single drone engagement
python examples/single_drone.py

# Multi-target scenarios  
python examples/multiple_targets.py

# Design optimization
python examples/parametric_study.py

# Complete multi-cannon analysis
python examples/multi_cannon_analysis.py
```

### Comprehensive Analysis Suite

```bash
# Run complete analysis with all visualizations
python run_complete_analysis.py --multi-cannon --visualizations --paper-figures

# Multi-cannon only analysis
python run_multi_cannon_complete.py
```

## Core Capabilities

### Physics Modeling
- **Vortex Ring Dynamics:** Formation number optimization, trajectory decay, energy transfer
- **Monte Carlo Analysis:** Statistical engagement modeling with atmospheric variations
- **Target Interaction:** Size-dependent vulnerability and damage assessment
- **Ballistic Solutions:** Moving target interception and lead angle calculation

### Multi-Cannon Arrays
- **Topology Support:** Linear, 2x2 grid, 3x3 grid, circular, networked arrangements
- **Coordination Modes:** Sequential, simultaneous, coordinated timing, adaptive targeting  
- **Combined Effects:** Energy combination modeling for enhanced lethality
- **Coverage Analysis:** Engagement envelope overlap and optimization

### Design Optimization
- **Parameter Sweeps:** Barrel dimensions, pressure settings, formation numbers
- **Performance Metrics:** Kill probability, energy efficiency, coverage area
- **Environmental Sensitivity:** Air density, wind effects, range limitations
- **Multi-Objective Optimization:** Balanced performance across threat categories

## Configuration

Edit `config/cannon_specs.yaml` for system parameters:

```yaml
cannon:
  barrel_length: 2.0      # meters
  barrel_diameter: 0.5    # meters  
  chamber_pressure: 240000 # Pa (realistic operating pressure)
  max_chamber_pressure: 300000 # Pa (3 bar safety limit)

vortex_ring:
  formation_number: 4.0   # Optimal stroke-to-diameter ratio
  initial_velocity: 80    # m/s (achievable with compressed air)
  effective_range: 45     # meters (physics-limited)

drone_models:
  small:                  # Consumer drones, racing quads
    mass: 0.5
    size: 0.3
    vulnerability: 0.65   # Realistic vulnerability assessment
  medium:                 # Professional UAVs  
    mass: 2.0
    size: 0.6
    vulnerability: 0.45   # More robust platforms
  large:                  # Fixed-wing tactical
    mass: 8.0
    size: 1.2
    vulnerability: 0.1    # Limited effectiveness
```

## Performance Characteristics

### Single Cannon (Baseline)
- **Effective Range:** 15-35m optimal, 45m maximum
- **Small Drone Kill Rate:** 70-90% within optimal range
- **Medium Drone Limitation:** <30% effectiveness, requires alternatives
- **Reload Capability:** ~2 seconds between shots
- **Angular Coverage:** ±85° elevation, 360° traverse

### 2x2 Multi-Cannon Array (Recommended)
- **Coverage Area:** 3-4x single cannon
- **Small Drone Performance:** 90%+ kill probability  
- **Medium Drone Capability:** 50-70% with coordinated fire
- **Simultaneous Targets:** 3-4 effective engagements
- **Vehicle Integration:** Compatible with truck/trailer mounting

### 3x3 Large Array (High-Value Asset Protection)
- **Coverage Area:** 6-8x single cannon
- **Large Drone Capability:** Limited but measurable effect
- **Swarm Defense:** 8-12 simultaneous small targets
- **Resource Intensive:** High power/space requirements
- **Fixed Installation:** Best for permanent base defense

## Technical Validation

### Physics Model Verification
- **Vortex Ring Theory:** Based on Gharib et al. formation number analysis
- **Energy Transfer:** Momentum-based damage modeling with empirical factors
- **Atmospheric Effects:** Reynolds number scaling and viscous decay
- **Range Limitations:** Consistent with experimental vortex cannon studies

### Monte Carlo Validation
- **1000+ Trial Analysis:** Statistical confidence in performance metrics
- **Environmental Uncertainty:** ±10% atmospheric variation modeling
- **Manufacturing Tolerances:** ±5% system parameter variation
- **Target Behavior:** Realistic drone response and vulnerability factors

## Visualization Suite

### Current Capabilities
- 3D engagement visualizations
- Performance envelope analysis  
- Multi-cannon coordination plots
- Parameter sensitivity analysis
- Array topology comparisons

## Data and Results

### Completed Analysis
- **13,623 bytes:** Single drone comprehensive analysis
- **15,340 bytes:** Multiple target engagement scenarios
- **13,360 bytes:** Parametric optimization study
- **61,887 bytes:** Complete multi-cannon array analysis

### Key Performance Data
- Small drone effectiveness: 88.5% average kill probability
- Multi-cannon improvement: 200% success rate increase
- Optimal configuration: 2.0m barrel, 0.5m diameter, 240kPa pressure
- Vehicle-mounted feasibility: Confirmed for 2x2 arrays

## Development Roadmap

### Known Limitations
- **Large Target Ineffectiveness:** Physics-limited against robust platforms
- **Range Constraints:** 45m maximum effective range
- **Weather Sensitivity:** Performance degrades in high winds
- **Power Requirements:** Continuous operation needs significant compressed air

## Contributing

This toolkit supports ongoing research into non-kinetic drone defense systems. The code is structured for easy modification and extension of analysis capabilities.

### Code Organization
- **Modular Physics:** Each component (cannon, vortex, engagement) separate
- **Configuration Driven:** YAML-based parameter management
- **Extensible Analysis:** Template-based scenario development
- **Comprehensive Testing:** Built-in validation and verification

### Analysis Extensions
- Custom drone models in `config/cannon_specs.yaml`
- New engagement scenarios in `examples/`
- Additional array topologies in `multi_cannon_array.py`
- Enhanced visualizations in `scripts/visualize.py`