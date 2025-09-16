#!/usr/bin/env python3
"""
Publication-Ready Vortex Cannon Visualizer

Generates journal-quality figures with grayscale compatibility and professional
formatting for academic publication. Focus on vehicle-mounted systems and
realistic small drone defense capabilities.

Usage:
    python scripts/visualize.py --figure-type envelope --drone-type small --output fig1_envelope.png
    python scripts/visualize.py --figure-type array-comparison --output fig2_arrays.png
    python scripts/visualize.py --figure-type performance --output fig3_performance.png
    python scripts/visualize.py --figure-type trajectory --output fig4_trajectory.png
    python scripts/visualize.py --figure-type vehicle --output fig5_vehicle.png

Note: Files are automatically saved to the 'figs' folder relative to the script location.
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Wedge
import matplotlib.gridspec as gridspec
from pathlib import Path
import yaml

# Publication settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Computer Modern Roman'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'axes.linewidth': 0.8,
    'grid.linewidth': 0.5,
    'lines.linewidth': 1.0,
    'patch.linewidth': 0.8,
    'text.usetex': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from cannon import VortexCannon, CannonConfiguration
    from engagement import EngagementCalculator, Target
    from vortex_ring import VortexRing
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure the src directory contains the required modules")
    sys.exit(1)

# Grayscale patterns and styles
PATTERNS = ['', '///', '...', '+++', 'xxx', '|||', '---', 'ooo']
GRAYS = ['#000000', '#333333', '#666666', '#999999', '#cccccc', '#e6e6e6', '#f0f0f0']
LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (3, 3))]


def load_config_with_defaults():
    """Load configuration with fallback defaults"""
    config_path = "config/cannon_specs.yaml"
    
    default_config = {
        'cannon': {
            'barrel_length': 2.0,
            'barrel_diameter': 0.5,
            'max_chamber_pressure': 300000,
            'chamber_pressure': 240000,
            'position': [0.0, 0.0, 2.0]
        },
        'vortex_ring': {
            'formation_number': 4.0,
            'initial_velocity': 80,
            'effective_range': 45
        },
        'environment': {
            'air_density': 1.225
        },
        'drone_models': {
            'small': {'mass': 0.5, 'size': 0.3, 'vulnerability': 0.65},
            'medium': {'mass': 2.0, 'size': 0.6, 'vulnerability': 0.45},
            'large': {'mass': 8.0, 'size': 1.2, 'vulnerability': 0.1}
        }
    }
    
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            for section in default_config:
                if section in user_config:
                    default_config[section].update(user_config[section])
        except Exception as e:
            print(f"Warning: Using defaults due to config error: {e}")
    
    return default_config


def create_cannon_from_config(config):
    """Create cannon instance from configuration"""
    cannon_config = config['cannon']
    vortex_config = config.get('vortex_ring', {})
    env_config = config.get('environment', {})
    
    config_obj = CannonConfiguration(
        barrel_length=cannon_config['barrel_length'],
        barrel_diameter=cannon_config['barrel_diameter'],
        max_chamber_pressure=cannon_config['max_chamber_pressure'],
        max_elevation=85.0,
        max_traverse=360.0,
        formation_number=vortex_config.get('formation_number', 4.0),
        air_density=env_config.get('air_density', 1.225),
        chamber_pressure=cannon_config['chamber_pressure']
    )
    
    cannon = VortexCannon.__new__(VortexCannon)
    cannon.config = config_obj
    cannon.position = np.array(cannon_config.get('position', [0.0, 0.0, 2.0]))
    cannon.orientation = {'elevation': 0.0, 'azimuth': 0.0}
    cannon.chamber_pressure = cannon_config['chamber_pressure']
    cannon.ready_to_fire = True
    cannon.last_shot_time = 0.0
    cannon.reload_time = 0.5
    cannon.pressure_buildup_time = 2.0
    
    return cannon


def create_engagement_envelope_figure(config, drone_type='small'):
    """Create publication-quality engagement envelope with grayscale contours"""
    cannon = create_cannon_from_config(config)
    calc = EngagementCalculator(cannon)
    
    # Calculate envelope data
    ranges = np.arange(5, 61, 2)  # 5-60m in 2m steps
    elevations = np.arange(0, 61, 5)  # 0-60° in 5° steps
    
    drone_spec = config['drone_models'][drone_type]
    kill_matrix = np.zeros((len(elevations), len(ranges)))
    
    print(f"Calculating envelope for {drone_type} drone...")
    for i, elev in enumerate(elevations):
        for j, range_val in enumerate(ranges):
            # Calculate target position
            elev_rad = np.radians(elev)
            target_pos = cannon.position + np.array([
                range_val * np.cos(elev_rad),
                0,
                range_val * np.sin(elev_rad)
            ])
            
            # Create test target
            target = Target(
                id=f"test_{range_val}_{elev}",
                position=target_pos,
                velocity=np.zeros(3),
                size=drone_spec['size'],
                vulnerability=drone_spec['vulnerability'],
                priority=1,
                detected_time=0.0
            )
            
            # Calculate engagement
            solution = calc.single_target_engagement(target)
            kill_matrix[i, j] = solution.kill_probability
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), 
                                   gridspec_kw={'width_ratios': [3, 1]})
    
    # Main envelope plot
    X, Y = np.meshgrid(ranges, elevations)
    levels = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    colors = ['#ffffff', '#d0d0d0', '#a0a0a0', '#707070', '#404040', '#000000']
    
    cs = ax1.contourf(X, Y, kill_matrix, levels=levels, colors=colors, alpha=0.8)
    cs_lines = ax1.contour(X, Y, kill_matrix, levels=levels[1:-1], 
                          colors='black', linewidths=0.8)
    ax1.clabel(cs_lines, inline=True, fontsize=8, fmt='%.1f')
    
    # Mark optimal point
    optimal_idx = np.unravel_index(np.argmax(kill_matrix), kill_matrix.shape)
    optimal_range = ranges[optimal_idx[1]]
    optimal_elev = elevations[optimal_idx[0]]
    ax1.plot(optimal_range, optimal_elev, 'k*', markersize=12, 
             markeredgewidth=1, markerfacecolor='white', label='Optimal Point')
    
    # Vehicle operational zone
    operational_zone = Rectangle((20, 10), 20, 35, fill=False, 
                               edgecolor='black', linewidth=2, linestyle='--',
                               label='Vehicle Operational Zone')
    ax1.add_patch(operational_zone)
    
    ax1.set_xlabel('Range (m)')
    ax1.set_ylabel('Elevation Angle (°)')
    ax1.set_title(f'Engagement Envelope - {drone_type.title()} Drone')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(5, 60)
    ax1.set_ylim(0, 60)
    
    # Performance summary
    ax2.axis('off')
    max_kill_prob = np.max(kill_matrix)
    effective_points = np.sum(kill_matrix >= 0.5)
    total_points = kill_matrix.size
    coverage_pct = (effective_points / total_points) * 100
    
    summary_text = f"""PERFORMANCE SUMMARY
                   
Max Kill Probability: {max_kill_prob:.2f}
Optimal: {optimal_range}m @ {optimal_elev}°
Effective Coverage: {coverage_pct:.1f}%

OPERATIONAL ZONES:
• Close (5-20m): High effectiveness
• Optimal (20-40m): Best performance
• Extended (40-60m): Limited effect

VEHICLE INTEGRATION:
• Compatible with truck mount
• 20-40m engagement envelope
• Suitable for mobile deployment"""
    
    ax2.text(0.05, 0.95, summary_text, transform=ax2.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor=GRAYS[5], alpha=0.8))
    
    # Add colorbar
    cbar = plt.colorbar(cs, ax=ax1, orientation='horizontal', 
                       pad=0.15, aspect=30, shrink=0.8)
    cbar.set_label('Kill Probability')
    
    plt.suptitle(f'Vortex Cannon Analysis - {drone_type.title()} Drone', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def create_array_comparison_figure(config):
    """Create vehicle-mounted array comparison figure"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Vehicle-Mounted Array Configurations', fontsize=14, fontweight='bold')
    
    configurations = [
        {'name': 'Single Cannon', 'positions': [[0, 0]], 'pattern': PATTERNS[0]},
        {'name': '2×2 Grid Array', 'positions': [[-10, -10], [10, -10], [-10, 10], [10, 10]], 'pattern': PATTERNS[1]},
        {'name': 'Linear Array', 'positions': [[-15, 0], [-5, 0], [5, 0], [15, 0]], 'pattern': PATTERNS[2]},
        {'name': 'Triangular Array', 'positions': [[0, 0], [-12, -15], [12, -15]], 'pattern': PATTERNS[3]}
    ]
    
    for idx, (ax, config_data) in enumerate(zip(axes.flat, configurations)):
        positions = np.array(config_data['positions'])
        
        # Draw vehicle outline
        vehicle = Rectangle((-20, -25), 40, 50, fill=True, 
                          facecolor=GRAYS[5], edgecolor='black', linewidth=1.5)
        ax.add_patch(vehicle)
        
        # Draw cab
        cab = Rectangle((-15, 20), 30, 10, fill=True,
                       facecolor=GRAYS[4], edgecolor='black', linewidth=1.5)
        ax.add_patch(cab)
        
        # Draw cannons
        for pos in positions:
            cannon = Circle(pos, 2.5, fill=True, facecolor='white', 
                          edgecolor='black', linewidth=2,
                          hatch=config_data['pattern'])
            ax.add_patch(cannon)
            
            # Engagement zone
            engagement_circle = Circle(pos, 30, fill=False, 
                                     edgecolor=GRAYS[2], linewidth=1, 
                                     linestyle='--', alpha=0.7)
            ax.add_patch(engagement_circle)
        
        # Test targets
        test_targets = [[25, 0], [35, 15], [30, -20]]
        for target_pos in test_targets:
            target = Circle(target_pos, 1, fill=True, facecolor='black')
            ax.add_patch(target)
        
        ax.set_xlim(-50, 50)
        ax.set_ylim(-40, 40)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Distance (m)')
        ax.set_ylabel('Distance (m)')
        ax.set_title(f'{config_data["name"]}\n{len(positions)} cannon(s)')
    
    plt.tight_layout()
    return fig


def create_performance_comparison_figure(config):
    """Create performance comparison: single vs multi-cannon"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Performance Comparison: Single vs Multi-Cannon Systems', 
                fontsize=14, fontweight='bold')
    
    # Target performance data
    targets = ['Small\nDrone', 'Medium\nDrone', 'Small\nSwarm', 'Mixed\nThreat']
    single_performance = [0.89, 0.0, 0.33, 0.33]
    multi_performance = [0.95, 0.45, 1.0, 0.67]
    
    x = np.arange(len(targets))
    width = 0.35
    
    # Performance bars
    bars1 = ax1.bar(x - width/2, single_performance, width, 
                   color='white', edgecolor='black', linewidth=1, 
                   hatch=PATTERNS[1], label='Single Cannon')
    bars2 = ax1.bar(x + width/2, multi_performance, width,
                   color=GRAYS[3], edgecolor='black', linewidth=1,
                   hatch=PATTERNS[2], label='2×2 Array')
    
    ax1.set_xlabel('Target Type')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('Engagement Success Rate')
    ax1.set_xticks(x)
    ax1.set_xticklabels(targets)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.1)
    
    # Coverage comparison
    config_names = ['Single', '2×2 Grid', 'Linear', 'Triangular']
    coverage_areas = [1, 4, 2.5, 3]
    
    ax2.bar(config_names, coverage_areas, 
           color=[GRAYS[i] for i in range(4)], 
           edgecolor='black', linewidth=1)
    ax2.set_ylabel('Relative Coverage Area')
    ax2.set_title('Coverage Area Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Range effectiveness
    ranges = np.arange(10, 61, 5)
    single_eff = np.exp(-ranges/40) * 0.9
    multi_eff = np.minimum(single_eff * 1.8, 1.0)
    
    ax3.plot(ranges, single_eff, 'k-', linewidth=2, marker='o', 
            markersize=4, label='Single Cannon')
    ax3.plot(ranges, multi_eff, 'k--', linewidth=2, marker='s', 
            markersize=4, markerfacecolor='white', label='2×2 Array')
    ax3.axhline(y=0.5, color=GRAYS[2], linestyle=':', alpha=0.8, 
               label='Effectiveness Threshold')
    
    ax3.set_xlabel('Range (m)')
    ax3.set_ylabel('Kill Probability')
    ax3.set_title('Range vs Effectiveness')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(10, 60)
    ax3.set_ylim(0, 1)
    
    # Resource efficiency
    scenarios = ['Light\nLoad', 'Balanced\nLoad', 'Heavy\nLoad']
    single_efficiency = [0.6, 0.4, 0.2]
    multi_efficiency = [0.8, 0.7, 0.5]
    
    x_eff = np.arange(len(scenarios))
    ax4.plot(x_eff, single_efficiency, 'ko-', linewidth=2, markersize=8,
            markerfacecolor='white', markeredgewidth=2, label='Single Cannon')
    ax4.plot(x_eff, multi_efficiency, 'ks--', linewidth=2, markersize=8,
            markerfacecolor=GRAYS[3], markeredgewidth=2, label='2×2 Array')
    
    ax4.set_xlabel('Threat Scenario')
    ax4.set_ylabel('System Efficiency')
    ax4.set_title('Resource Efficiency')
    ax4.set_xticks(x_eff)
    ax4.set_xticklabels(scenarios)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    plt.tight_layout()
    return fig


def create_trajectory_analysis_figure(config):
    """Create vortex ring trajectory analysis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Vortex Ring Trajectory Analysis', fontsize=14, fontweight='bold')
    
    cannon = create_cannon_from_config(config)
    vr = cannon.generate_vortex_ring()
    
    # Time points
    times = np.linspace(0, 2.0, 100)
    states = [vr.trajectory(t) for t in times]
    
    positions = np.array([s.position[0] for s in states])
    velocities = np.array([s.velocity for s in states])
    diameters = np.array([s.diameter for s in states])
    energies = np.array([s.energy for s in states])
    
    # Trajectory path
    ax1.plot(positions, np.full_like(positions, 5), 'k-', linewidth=2)
    ax1.plot(0, 5, 'ro', markersize=8, label='Cannon Position')
    
    # Ring size visualization
    for i in range(0, len(positions), 15):
        circle = Circle((positions[i], 5), diameters[i]/4, 
                       fill=False, edgecolor=GRAYS[2], alpha=0.6)
        ax1.add_patch(circle)
    
    ax1.set_xlabel('Range (m)')
    ax1.set_ylabel('Height (m)')
    ax1.set_title('Vortex Ring Trajectory Path')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(-5, 50)
    ax1.set_ylim(0, 10)
    
    # Velocity decay
    ax2.plot(positions, velocities, 'k-', linewidth=2)
    ax2.fill_between(positions, velocities, alpha=0.3, color=GRAYS[3])
    ax2.set_xlabel('Range (m)')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_title('Velocity vs Range')
    ax2.grid(True, alpha=0.3)
    
    # Ring expansion
    ax3.plot(positions, diameters, 'k-', linewidth=2)
    ax3.axhline(y=diameters[0], color=GRAYS[1], linestyle='--', 
               label=f'Initial: {diameters[0]:.2f}m')
    ax3.set_xlabel('Range (m)')
    ax3.set_ylabel('Ring Diameter (m)')
    ax3.set_title('Ring Expansion')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Energy decay
    ax4.plot(positions, energies, 'k-', linewidth=2)
    ax4.fill_between(positions, energies, alpha=0.3, color=GRAYS[2])
    
    # Energy thresholds
    thresholds = [100, 500, 1000]
    labels = ['Large Drone', 'Medium Drone', 'Small Drone']
    for threshold, label in zip(thresholds, labels):
        ax4.axhline(y=threshold, color=GRAYS[1], linestyle=':', alpha=0.8)
        ax4.text(2, threshold + 50, label, fontsize=8)
    
    ax4.set_xlabel('Range (m)')
    ax4.set_ylabel('Kinetic Energy (J)')
    ax4.set_title('Energy vs Range')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def create_vehicle_integration_figure(config):
    """Create vehicle integration analysis figure"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Vehicle Integration Analysis', fontsize=14, fontweight='bold')
    
    # Platform comparison
    platforms = ['Light Truck', 'Medium Truck', 'Trailer', 'Fixed Site']
    cannon_capacity = [1, 2, 4, 9]
    mobility_score = [10, 8, 6, 0]
    
    x = np.arange(len(platforms))
    width = 0.35
    
    ax1.bar(x - width/2, cannon_capacity, width, 
           color='white', edgecolor='black', linewidth=1,
           hatch=PATTERNS[1], label='Cannon Capacity')
    ax1.bar(x + width/2, mobility_score, width,
           color=GRAYS[3], edgecolor='black', linewidth=1,
           hatch=PATTERNS[2], label='Mobility Score')
    
    ax1.set_xlabel('Platform Type')
    ax1.set_ylabel('Capability Score')
    ax1.set_title('Platform Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(platforms, rotation=15)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Engagement geometry
    vehicle_rect = Rectangle((-4, -1.25), 8, 2.5, fill=True, 
                           facecolor=GRAYS[4], edgecolor='black', linewidth=2)
    ax2.add_patch(vehicle_rect)
    
    # Cannon positions
    cannon_positions = [[-2, -1], [2, -1], [-2, 1], [2, 1]]
    for pos in cannon_positions:
        cannon = Circle(pos, 0.3, fill=True, facecolor='white',
                       edgecolor='black', linewidth=1.5)
        ax2.add_patch(cannon)
        
        # Engagement arc
        wedge = Wedge(pos, 25, -45, 45, fill=False, 
                     edgecolor=GRAYS[2], linewidth=1, alpha=0.6)
        ax2.add_patch(wedge)
    
    ax2.set_xlim(-30, 30)
    ax2.set_ylim(-15, 15)
    ax2.set_aspect('equal')
    ax2.set_xlabel('Forward Distance (m)')
    ax2.set_ylabel('Lateral Distance (m)')
    ax2.set_title('Vehicle-Mounted Geometry')
    ax2.grid(True, alpha=0.3)
    
    # Timeline
    phases = ['Detection', 'Targeting', 'Firing', 'Assessment']
    single_times = [0.5, 1.0, 0.1, 0.5]
    multi_times = [0.3, 0.8, 0.2, 0.3]
    
    y_pos = np.arange(len(phases))
    ax3.barh(y_pos - 0.2, single_times, 0.4,
            color='white', edgecolor='black', linewidth=1,
            hatch=PATTERNS[1], label='Single Cannon')
    ax3.barh(y_pos + 0.2, multi_times, 0.4,
            color=GRAYS[3], edgecolor='black', linewidth=1,
            hatch=PATTERNS[2], label='Multi-Cannon')
    
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(phases)
    ax3.set_xlabel('Time (seconds)')
    ax3.set_title('Engagement Timeline')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Resource requirements
    systems = ['1 Cannon', '2×2 Array', 'Linear']
    power_req = [5, 18, 16]
    air_consumption = [20, 70, 60]
    
    x_res = np.arange(len(systems))
    ax4.bar(x_res - width/2, power_req, width,
           color='white', edgecolor='black', linewidth=1,
           hatch=PATTERNS[1], label='Power (kW)')
    ax4.bar(x_res + width/2, np.array(air_consumption)/5, width,
           color=GRAYS[3], edgecolor='black', linewidth=1,
           hatch=PATTERNS[2], label='Air (L/min ÷5)')
    
    ax4.set_xlabel('System Configuration')
    ax4.set_ylabel('Resource Requirement')
    ax4.set_title('Power & Air Requirements')
    ax4.set_xticks(x_res)
    ax4.set_xticklabels(systems)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Publication-Ready Vortex Cannon Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Figure Types:
  envelope        - Engagement envelope analysis (grayscale contours)
  array-comparison - Vehicle-mounted array configurations  
  performance     - Single vs multi-cannon performance comparison
  trajectory      - Vortex ring physics and trajectory analysis
  vehicle         - Vehicle integration and deployment analysis
        """
    )
    
    parser.add_argument('--figure-type', 
                       choices=['envelope', 'array-comparison', 'performance', 
                               'trajectory', 'vehicle'],
                       required=True,
                       help='Type of figure to generate')
    
    parser.add_argument('--drone-type', 
                       choices=['small', 'medium', 'large'], 
                       default='small',
                       help='Drone type for envelope analysis')
    
    parser.add_argument('--output', type=str, required=True,
                       help='Output file path')
    
    parser.add_argument('--dpi', type=int, default=300,
                       help='Output resolution')
    
    parser.add_argument('--format', 
                       choices=['png', 'pdf', 'svg', 'eps'],
                       default='png',
                       help='Output format')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config_with_defaults()
    
    # Automatically save to figs folder
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up from scripts to root
    figs_dir = os.path.join(script_dir, 'figs')
    output_path = os.path.join(figs_dir, args.output)
    
    # Ensure figs directory exists
    os.makedirs(figs_dir, exist_ok=True)
    
    try:
        print(f"Generating {args.figure_type} figure...")
        
        # Generate figure
        if args.figure_type == 'envelope':
            fig = create_engagement_envelope_figure(config, args.drone_type)
        elif args.figure_type == 'array-comparison':
            fig = create_array_comparison_figure(config)
        elif args.figure_type == 'performance':
            fig = create_performance_comparison_figure(config)
        elif args.figure_type == 'trajectory':
            fig = create_trajectory_analysis_figure(config)
        elif args.figure_type == 'vehicle':
            fig = create_vehicle_integration_figure(config)
        
        # Save figure
        fig.savefig(output_path, format=args.format, dpi=args.dpi, 
                   bbox_inches='tight', pad_inches=0.1,
                   facecolor='white', edgecolor='none')
        
        print(f"Figure saved: {output_path}")
        print(f"Format: {args.format.upper()}, DPI: {args.dpi}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())