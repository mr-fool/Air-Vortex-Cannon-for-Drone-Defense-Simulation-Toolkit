#!/usr/bin/env python3
"""
Vortex Cannon Engagement Visualizer

This script generates 3D visualizations of vortex cannon engagements showing
cannon placement, vortex ring trajectory, target positions, and engagement
effectiveness zones for drone defense analysis.

Usage:
    python scripts/visualize.py --target-x 30 --target-y 10 --target-z 15 --output figs/engagement.png
    python scripts/visualize.py --envelope-plot --drone-type small --output figs/envelope.png
    python scripts/visualize.py --trajectory-analysis --output figs/trajectory.png
"""

import argparse
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
import yaml
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import yaml
# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cannon import VortexCannon, CannonConfiguration
from engagement import EngagementCalculator, Target
from vortex_ring import VortexRing


def load_config_with_defaults(config_path: Optional[str] = None) -> Dict:
    """Load configuration with fallback to defaults"""
    default_config = {
        'cannon': {
            'barrel_length': 2.0,
            'barrel_diameter': 0.5,
            'max_chamber_pressure': 100000,
            'max_elevation': 85.0,
            'max_traverse': 360.0,
            'position': [0.0, 0.0, 2.0],
            'chamber_pressure': 80000
        },
        'vortex_ring': {
            'formation_number': 4.0,
            'initial_velocity': 50.0,
            'effective_range': 50.0
        },
        'environment': {
            'air_density': 1.225
        },
        'drone_models': {
            'small': {'mass': 0.5, 'size': 0.3, 'vulnerability': 0.9},
            'medium': {'mass': 2.0, 'size': 0.6, 'vulnerability': 0.7},
            'large': {'mass': 8.0, 'size': 1.2, 'vulnerability': 0.5}
        }
    }
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f)
            for section in default_config:
                if section in user_config:
                    default_config[section].update(user_config[section])
        except Exception as e:
            print(f"Warning: Could not load config file: {e}")
    
    return default_config


def create_cannon_from_config(config: Dict) -> VortexCannon:
    """Create cannon from configuration dictionary"""
    cannon_config = config['cannon']
    vortex_config = config.get('vortex_ring', {})
    env_config = config.get('environment', {})
    
    # Get chamber pressure from config
    max_pressure = cannon_config['max_chamber_pressure']
    chamber_pressure = cannon_config.get('chamber_pressure', max_pressure * 0.8)
    
    config_obj = CannonConfiguration(
        barrel_length=cannon_config['barrel_length'],
        barrel_diameter=cannon_config['barrel_diameter'],
        max_chamber_pressure=max_pressure,
        max_elevation=cannon_config.get('max_elevation', 85.0),
        max_traverse=cannon_config.get('max_traverse', 360.0),
        formation_number=vortex_config.get('formation_number', 4.0),
        air_density=env_config.get('air_density', 1.225),
        chamber_pressure=chamber_pressure
    )
    
    cannon = VortexCannon.__new__(VortexCannon)
    cannon.config = config_obj
    cannon.position = np.array(cannon_config.get('position', [0.0, 0.0, 2.0]))
    cannon.orientation = {'elevation': 0.0, 'azimuth': 0.0}
    cannon.chamber_pressure = chamber_pressure
    cannon.ready_to_fire = True
    cannon.last_shot_time = 0.0
    cannon.reload_time = 0.5
    cannon.pressure_buildup_time = 2.0
    
    return cannon


def plot_cannon_3d(ax, cannon_pos: np.ndarray, elevation: float, azimuth: float, 
                   barrel_length: float, barrel_diameter: float):
    """Draw 3D cannon representation"""
    # Cannon base (cylinder)
    base_height = 0.8
    base_radius = barrel_diameter * 0.8
    
    # Base cylinder
    theta = np.linspace(0, 2*np.pi, 20)
    z_base = np.linspace(cannon_pos[2] - base_height/2, cannon_pos[2] + base_height/2, 10)
    theta_mesh, z_mesh = np.meshgrid(theta, z_base)
    x_base = cannon_pos[0] + base_radius * np.cos(theta_mesh)
    y_base = cannon_pos[1] + base_radius * np.sin(theta_mesh)
    
    ax.plot_surface(x_base, y_base, z_mesh, alpha=0.6, color='darkgray')
    
    # Barrel direction vector
    elev_rad = np.radians(elevation)
    azim_rad = np.radians(azimuth)
    
    barrel_end = cannon_pos + barrel_length * np.array([
        np.cos(elev_rad) * np.cos(azim_rad),
        np.cos(elev_rad) * np.sin(azim_rad),
        np.sin(elev_rad)
    ])
    
    # Barrel (line)
    ax.plot([cannon_pos[0], barrel_end[0]], 
            [cannon_pos[1], barrel_end[1]], 
            [cannon_pos[2], barrel_end[2]], 
            'k-', linewidth=8, label='Cannon Barrel')
    
    # Cannon position marker
    ax.scatter(cannon_pos[0], cannon_pos[1], cannon_pos[2], 
               c='red', s=100, marker='s', label='Cannon')
    
    return barrel_end


def plot_vortex_trajectory(ax, cannon_pos: np.ndarray, target_pos: np.ndarray, 
                          vortex_ring: VortexRing, solution_data: Dict):
    """Plot vortex ring trajectory with size expansion"""
    # Calculate trajectory points
    flight_time = solution_data.get('flight_time', 2.0)
    time_points = np.linspace(0, flight_time, 50)
    
    # Direction vector from cannon to target
    direction = target_pos - cannon_pos
    direction = direction / np.linalg.norm(direction)
    
    trajectory_points = []
    ring_sizes = []
    velocities = []
    
    for t in time_points:
        state = vortex_ring.trajectory(t)
        # Project trajectory in correct direction
        pos = cannon_pos + direction * state.position[0]
        trajectory_points.append(pos)
        ring_sizes.append(state.diameter)
        velocities.append(state.velocity)
    
    trajectory_points = np.array(trajectory_points)
    
    # Plot trajectory line
    ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2],
            'b-', linewidth=2, alpha=0.8, label='Vortex Ring Trajectory')
    
    # Plot vortex rings at key points (every 10th point)
    ring_colors = plt.cm.plasma(np.linspace(0, 1, len(trajectory_points[::10])))
    
    for i, (point, size, color) in enumerate(zip(trajectory_points[::10], 
                                                 ring_sizes[::10], ring_colors)):
        # Draw ring as circle perpendicular to trajectory
        ring_radius = size / 2
        
        # Create circle in XY plane, then rotate to trajectory direction
        circle_theta = np.linspace(0, 2*np.pi, 20)
        circle_x = ring_radius * np.cos(circle_theta)
        circle_y = ring_radius * np.sin(circle_theta)
        circle_z = np.zeros_like(circle_x)
        
        # Simple rotation for visualization (assumes mostly horizontal trajectory)
        ring_x = point[0] + circle_x
        ring_y = point[1] + circle_y  
        ring_z = point[2] + circle_z
        
        alpha = 0.6 - i * 0.05  # Fade with distance
        alpha = max(alpha, 0.1)
        
        ax.plot(ring_x, ring_y, ring_z, color=color, alpha=alpha, linewidth=1)
    
    return trajectory_points


def plot_target_3d(ax, target_pos: np.ndarray, drone_size: float, 
                   velocity: Optional[np.ndarray] = None, target_id: str = "Target"):
    """Draw 3D drone target representation"""
    # Drone body (sphere)
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 10)
    radius = drone_size / 2
    
    x = target_pos[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = target_pos[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = target_pos[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, alpha=0.7, color='orange')
    
    # Target marker
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
               c='orange', s=150, marker='o', label=target_id)
    
    # Velocity vector if provided
    if velocity is not None and np.linalg.norm(velocity) > 0.1:
        vel_scale = 3.0  # Scale factor for visibility
        vel_end = target_pos + velocity * vel_scale
        ax.quiver(target_pos[0], target_pos[1], target_pos[2],
                 velocity[0], velocity[1], velocity[2],
                 length=vel_scale, normalize=True, color='red', 
                 arrow_length_ratio=0.1, label='Velocity')


def plot_engagement_zones(ax, cannon_pos: np.ndarray, max_range: float = 60):
    """Plot engagement effectiveness zones"""
    # Create range rings
    ranges = [15, 30, 45, 60]
    colors = ['green', 'yellow', 'orange', 'red']
    alphas = [0.3, 0.2, 0.15, 0.1]
    labels = ['Optimal', 'Effective', 'Marginal', 'Maximum']
    
    for r, color, alpha, label in zip(ranges, colors, alphas, labels):
        # Hemisphere for 3D effect
        phi = np.linspace(0, np.pi/2, 20)  # 0 to 90 degrees elevation
        theta = np.linspace(0, 2*np.pi, 40)  # Full rotation
        
        phi_mesh, theta_mesh = np.meshgrid(phi, theta)
        
        x = cannon_pos[0] + r * np.sin(phi_mesh) * np.cos(theta_mesh)
        y = cannon_pos[1] + r * np.sin(phi_mesh) * np.sin(theta_mesh)
        z = cannon_pos[2] + r * np.cos(phi_mesh)
        
        # Only plot upper hemisphere (above cannon)
        mask = z >= cannon_pos[2]
        x_masked = np.where(mask, x, np.nan)
        y_masked = np.where(mask, y, np.nan)
        z_masked = np.where(mask, z, np.nan)
        
        ax.plot_surface(x_masked, y_masked, z_masked, 
                       alpha=alpha, color=color, label=f'{label} Range ({r}m)')


def create_engagement_plot(cannon: VortexCannon, target: Target, 
                          solution_data: Dict, config: Dict) -> plt.Figure:
    """Create complete 3D engagement visualization"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract solution data
    elevation = solution_data.get('elevation', 0)
    azimuth = solution_data.get('azimuth', 0)
    success = solution_data.get('success', False)
    
    # Plot engagement zones
    plot_engagement_zones(ax, cannon.position)
    
    # Plot cannon
    barrel_end = plot_cannon_3d(ax, cannon.position, elevation, azimuth,
                               cannon.config.barrel_length, cannon.config.barrel_diameter)
    
    # Plot target
    plot_target_3d(ax, target.position, target.size, target.velocity, target.id)
    
    # Plot trajectory if successful
    if success and 'vortex_ring' in solution_data:
        vr = solution_data['vortex_ring']
        plot_vortex_trajectory(ax, cannon.position, target.position, vr, solution_data)
    
    # Formatting
    ax.set_xlabel('X Distance (m)')
    ax.set_ylabel('Y Distance (m)')
    ax.set_zlabel('Z Altitude (m)')
    
    # Set axis limits
    all_points = np.vstack([cannon.position, target.position])
    margin = 10
    ax.set_xlim(np.min(all_points[:, 0]) - margin, np.max(all_points[:, 0]) + margin)
    ax.set_ylim(np.min(all_points[:, 1]) - margin, np.max(all_points[:, 1]) + margin)
    ax.set_zlim(0, np.max(all_points[:, 2]) + margin)
    
    # Title and info
    status = "SUCCESSFUL" if success else "FAILED"
    kill_prob = solution_data.get('kill_probability', 0)
    range_val = solution_data.get('target_range', 0)
    
    title = f"Vortex Cannon Engagement - {status}\n"
    title += f"Target: {target.id} | Range: {range_val:.1f}m | P_kill: {kill_prob:.3f}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Grid
    ax.grid(True, alpha=0.3)
    
    return fig


def create_envelope_plot(config: Dict, drone_type: str = 'small') -> plt.Figure:
    """Create engagement envelope heatmap"""
    cannon = create_cannon_from_config(config)
    calc = EngagementCalculator(cannon)
    
    print(f"Calculating engagement envelope for {drone_type} drones...")
    envelope_data = calc.engagement_envelope_analysis(drone_type)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Kill probability heatmap
    ranges = np.array(envelope_data['ranges'])
    elevations = np.array(envelope_data['elevations'])
    kill_matrix = np.array(envelope_data['kill_probability_matrix'])
    
    im1 = ax1.imshow(kill_matrix, extent=[ranges.min(), ranges.max(), 
                                         elevations.min(), elevations.max()],
                     aspect='auto', origin='lower', cmap='RdYlGn', vmin=0, vmax=1)
    
    ax1.set_xlabel('Range (m)')
    ax1.set_ylabel('Elevation (degrees)')
    ax1.set_title(f'Kill Probability - {drone_type.title()} Drone')
    ax1.grid(True, alpha=0.3)
    
    # Add contour lines
    contour1 = ax1.contour(ranges, elevations, kill_matrix, 
                          levels=[0.3, 0.5, 0.7, 0.9], colors='black', alpha=0.5)
    ax1.clabel(contour1, inline=True, fontsize=8)
    
    # Colorbar
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Kill Probability')
    
    # Hit probability heatmap
    hit_matrix = np.array(envelope_data['hit_probability_matrix'])
    im2 = ax2.imshow(hit_matrix, extent=[ranges.min(), ranges.max(), 
                                        elevations.min(), elevations.max()],
                     aspect='auto', origin='lower', cmap='Blues', vmin=0, vmax=1)
    
    ax2.set_xlabel('Range (m)')
    ax2.set_ylabel('Elevation (degrees)')
    ax2.set_title(f'Hit Probability - {drone_type.title()} Drone')
    ax2.grid(True, alpha=0.3)
    
    # Add contour lines
    contour2 = ax2.contour(ranges, elevations, hit_matrix, 
                          levels=[0.3, 0.5, 0.7, 0.9], colors='black', alpha=0.5)
    ax2.clabel(contour2, inline=True, fontsize=8)
    
    # Colorbar
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Hit Probability')
    
    # Add performance summary
    max_range = envelope_data['max_effective_range']
    optimal_range = envelope_data['optimal_range']
    optimal_elev = envelope_data['optimal_elevation']
    
    fig.suptitle(f'Engagement Envelope Analysis - {drone_type.title()} Drone\n'
                f'Max Effective Range: {max_range}m | Optimal: {optimal_range}m @ {optimal_elev}°',
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def create_trajectory_analysis(config: Dict) -> plt.Figure:
    """Create trajectory analysis plots"""
    cannon = create_cannon_from_config(config)
    vr = cannon.generate_vortex_ring()
    
    fig = vr.plot_trajectory(max_time=3.0)
    fig.suptitle('Vortex Ring Trajectory Analysis', fontsize=14, fontweight='bold')
    
    return fig


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Vortex Cannon Engagement Visualizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single target engagement
  python scripts/visualize.py --target-x 30 --target-y 10 --target-z 15 --drone-size small
  
  # Engagement envelope analysis
  python scripts/visualize.py --envelope-plot --drone-type medium
  
  # Trajectory analysis
  python scripts/visualize.py --trajectory-analysis
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument('--envelope-plot', action='store_true',
                           help='Generate engagement envelope heatmap')
    mode_group.add_argument('--trajectory-analysis', action='store_true',
                           help='Generate vortex ring trajectory analysis')
    
    # Target parameters (for engagement plot)
    parser.add_argument('--target-x', type=float,
                       help='Target X coordinate (meters)')
    parser.add_argument('--target-y', type=float, default=0.0,
                       help='Target Y coordinate (meters)')
    parser.add_argument('--target-z', type=float,
                       help='Target Z coordinate (meters)')
    parser.add_argument('--drone-size', choices=['small', 'medium', 'large'], default='small',
                       help='Drone size category')
    
    # Target velocity
    parser.add_argument('--velocity-x', type=float, default=0.0,
                       help='Target X velocity (m/s)')
    parser.add_argument('--velocity-y', type=float, default=0.0,
                       help='Target Y velocity (m/s)')
    parser.add_argument('--velocity-z', type=float, default=0.0,
                       help='Target Z velocity (m/s)')
    
    # Envelope analysis
    parser.add_argument('--drone-type', choices=['small', 'medium', 'large'], default='small',
                       help='Drone type for envelope analysis')
    
    # Configuration and output
    parser.add_argument('--config', type=str, default='config/cannon_specs.yaml',
                       help='Configuration file path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file path (PNG, PDF, etc.)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Output resolution (DPI)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.envelope_plot and not args.trajectory_analysis:
        if args.target_x is None or args.target_z is None:
            parser.error("Engagement plot requires --target-x and --target-z")
    
    # Load configuration
    config = load_config_with_defaults(args.config)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        if args.envelope_plot:
            fig = create_envelope_plot(config, args.drone_type)
        elif args.trajectory_analysis:
            fig = create_trajectory_analysis(config)
        else:
            # Single target engagement plot
            cannon = create_cannon_from_config(config)
            calc = EngagementCalculator(cannon)
            
            # Create target
            drone_spec = config['drone_models'][args.drone_size]
            target = Target(
                id=f"{args.drone_size}_drone",
                position=np.array([args.target_x, args.target_y, args.target_z]),
                velocity=np.array([args.velocity_x, args.velocity_y, args.velocity_z]),
                size=drone_spec['size'],
                vulnerability=drone_spec['vulnerability'],
                priority=1,
                detected_time=0.0
            )
            
            # Calculate engagement
            solution = calc.single_target_engagement(target)
            
            # Add vortex ring to solution data
            if solution.success:
                vr = cannon.generate_vortex_ring(target.position)
                solution_dict = solution.__dict__.copy()
                solution_dict['vortex_ring'] = vr
            else:
                solution_dict = solution.__dict__.copy()
            
            fig = create_engagement_plot(cannon, target, solution_dict, config)
        
        # Save figure
        fig.savefig(args.output, dpi=args.dpi, bbox_inches='tight')
        print(f"Visualization saved to: {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())