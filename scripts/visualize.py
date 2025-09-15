#!/usr/bin/env python3
"""
Enhanced Vortex Cannon Engagement Visualizer with Multi-Array Support

This script generates 3D visualizations of both single and multi-cannon array
engagements, showing array topologies, coordinated firing patterns, and
combined vortex effects for drone defense analysis.

Usage:
    # Single cannon visualization
    python scripts/visualize.py --target-x 30 --target-y 10 --target-z 15 --output figs/engagement.png
    
    # Multi-cannon array visualization
    python scripts/visualize.py --multi-array --topology grid_2x2 --targets 2 --output figs/array_engagement.png
    
    # Array topology comparison
    python scripts/visualize.py --array-comparison --output figs/topology_comparison.png
    
    # Envelope analysis for arrays
    python scripts/visualize.py --envelope-plot --drone-type small --array-size 4 --output figs/array_envelope.png
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

# Multi-cannon imports
try:
    from multi_cannon_array import MultiCannonArray, ArrayTopology, FiringMode, create_test_array
    MULTI_CANNON_AVAILABLE = True
except ImportError:
    MULTI_CANNON_AVAILABLE = False
    print("Warning: Multi-cannon modules not available. Single-cannon mode only.")


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
        },
        'multi_cannon': {
            'default_spacing': 20.0,
            'coordination_delay': 0.1,
            'max_simultaneous_targets': 5
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
                   barrel_length: float, barrel_diameter: float, cannon_id: str = "Cannon",
                   color: str = 'red', alpha: float = 1.0):
    """Draw 3D cannon representation with customizable appearance"""
    # Cannon base (cylinder)
    base_height = 0.8
    base_radius = barrel_diameter * 0.8
    
    # Base cylinder
    theta = np.linspace(0, 2*np.pi, 20)
    z_base = np.linspace(cannon_pos[2] - base_height/2, cannon_pos[2] + base_height/2, 10)
    theta_mesh, z_mesh = np.meshgrid(theta, z_base)
    x_base = cannon_pos[0] + base_radius * np.cos(theta_mesh)
    y_base = cannon_pos[1] + base_radius * np.sin(theta_mesh)
    
    ax.plot_surface(x_base, y_base, z_mesh, alpha=0.6*alpha, color='darkgray')
    
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
            'k-', linewidth=8*alpha, alpha=alpha)
    
    # Cannon position marker
    ax.scatter(cannon_pos[0], cannon_pos[1], cannon_pos[2], 
               c=color, s=100*alpha, marker='s', alpha=alpha, label=cannon_id)
    
    return barrel_end


def plot_multi_cannon_array(ax, array: 'MultiCannonArray', targets: List[Target], 
                           results: Optional[List[Dict]] = None):
    """Plot multi-cannon array with coordinated engagement visualization"""
    if not MULTI_CANNON_AVAILABLE:
        raise ImportError("Multi-cannon functionality not available")
    
    # Color scheme for cannons
    cannon_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'cyan']
    
    # Plot each cannon in the array
    cannon_positions = []
    for i, cannon in enumerate(array.cannons):
        color = cannon_colors[i % len(cannon_colors)]
        
        # Get cannon orientation (simplified - assume targeting nearest target)
        if targets:
            nearest_target = min(targets, key=lambda t: np.linalg.norm(t.position - cannon.position))
            direction = nearest_target.position - cannon.position
            elevation = np.degrees(np.arcsin(direction[2] / np.linalg.norm(direction)))
            azimuth = np.degrees(np.arctan2(direction[1], direction[0]))
        else:
            elevation, azimuth = 0, 0
        
        plot_cannon_3d(ax, cannon.position, elevation, azimuth,
                      cannon.config.barrel_length, cannon.config.barrel_diameter,
                      f"Cannon {i+1}", color, alpha=0.8)
        cannon_positions.append(cannon.position)
    
    # Plot array connection lines to show topology
    cannon_positions = np.array(cannon_positions)
    
    # Connect cannons based on topology
    if array.topology == ArrayTopology.LINE:
        for i in range(len(cannon_positions) - 1):
            ax.plot([cannon_positions[i][0], cannon_positions[i+1][0]],
                   [cannon_positions[i][1], cannon_positions[i+1][1]],
                   [cannon_positions[i][2], cannon_positions[i+1][2]],
                   'k--', alpha=0.3, linewidth=1)
    
    elif array.topology == ArrayTopology.GRID_2x2:
        # Connect grid neighbors
        for i in range(2):
            for j in range(2):
                idx = i * 2 + j
                if idx < len(cannon_positions):
                    # Horizontal connections
                    if j < 1 and idx + 1 < len(cannon_positions):
                        ax.plot([cannon_positions[idx][0], cannon_positions[idx+1][0]],
                               [cannon_positions[idx][1], cannon_positions[idx+1][1]],
                               [cannon_positions[idx][2], cannon_positions[idx+1][2]],
                               'k--', alpha=0.3, linewidth=1)
                    # Vertical connections  
                    if i < 1 and idx + 2 < len(cannon_positions):
                        ax.plot([cannon_positions[idx][0], cannon_positions[idx+2][0]],
                               [cannon_positions[idx][1], cannon_positions[idx+2][1]],
                               [cannon_positions[idx][2], cannon_positions[idx+2][2]],
                               'k--', alpha=0.3, linewidth=1)
    
    elif array.topology == ArrayTopology.CIRCLE:
        # Connect in circle
        for i in range(len(cannon_positions)):
            next_i = (i + 1) % len(cannon_positions)
            ax.plot([cannon_positions[i][0], cannon_positions[next_i][0]],
                   [cannon_positions[i][1], cannon_positions[next_i][1]],
                   [cannon_positions[i][2], cannon_positions[next_i][2]],
                   'k--', alpha=0.3, linewidth=1)
    
    # Plot coordinated engagement zones if results provided
    if results:
        plot_coordinated_engagement_zones(ax, array, results)
    
    # Add array info text
    array_center = np.mean(cannon_positions, axis=0)
    info_text = f"Array: {array.topology.name}\nMode: {array.firing_mode.name}\nCannons: {len(array.cannons)}"
    ax.text(array_center[0], array_center[1], array_center[2] + 5,
           info_text, fontsize=10, ha='center',
           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))


def plot_coordinated_engagement_zones(ax, array: 'MultiCannonArray', results: List[Dict]):
    """Plot combined engagement zones from coordinated fire"""
    # Create combined effectiveness field
    x_range = np.linspace(-60, 60, 30)
    y_range = np.linspace(-60, 60, 30)
    z_range = np.linspace(5, 40, 20)
    
    # Sample a subset for visualization
    sample_points = []
    effectiveness_values = []
    
    for x in x_range[::3]:
        for y in y_range[::3]:
            for z in z_range[::2]:
                point = np.array([x, y, z])
                
                # Calculate combined effectiveness from all cannons
                combined_effectiveness = 0
                for cannon in array.cannons:
                    distance = np.linalg.norm(point - cannon.position)
                    if distance < 60:  # Within range
                        # Simple effectiveness model
                        base_effectiveness = max(0, (60 - distance) / 60)
                        combined_effectiveness += base_effectiveness
                
                if combined_effectiveness > 0.1:  # Only plot significant zones
                    sample_points.append(point)
                    effectiveness_values.append(min(combined_effectiveness, 1.0))
    
    if sample_points:
        sample_points = np.array(sample_points)
        effectiveness_values = np.array(effectiveness_values)
        
        # Plot as scatter with color coding
        scatter = ax.scatter(sample_points[:, 0], sample_points[:, 1], sample_points[:, 2],
                           c=effectiveness_values, cmap='plasma', alpha=0.3, s=20)
        
        # Add colorbar
        plt.colorbar(scatter, ax=ax, label='Combined Effectiveness', shrink=0.5)


def plot_vortex_trajectory(ax, cannon_pos: np.ndarray, target_pos: np.ndarray, 
                          vortex_ring: VortexRing, solution_data: Dict, 
                          color: str = 'blue', alpha: float = 0.8):
    """Plot vortex ring trajectory with customizable appearance"""
    # Calculate trajectory points
    flight_time = solution_data.get('flight_time', 2.0)
    time_points = np.linspace(0, flight_time, 50)
    
    # Direction vector from cannon to target
    direction = target_pos - cannon_pos
    direction = direction / np.linalg.norm(direction)
    
    trajectory_points = []
    ring_sizes = []
    
    for t in time_points:
        state = vortex_ring.trajectory(t)
        # Project trajectory in correct direction
        pos = cannon_pos + direction * state.position[0]
        trajectory_points.append(pos)
        ring_sizes.append(state.diameter)
    
    trajectory_points = np.array(trajectory_points)
    
    # Plot trajectory line
    ax.plot(trajectory_points[:, 0], trajectory_points[:, 1], trajectory_points[:, 2],
            color=color, linewidth=2, alpha=alpha, label='Vortex Ring Trajectory')
    
    # Plot vortex rings at key points
    for i, (point, size) in enumerate(zip(trajectory_points[::10], ring_sizes[::10])):
        ring_radius = size / 2
        
        # Create circle for ring visualization
        circle_theta = np.linspace(0, 2*np.pi, 20)
        circle_x = ring_radius * np.cos(circle_theta)
        circle_y = ring_radius * np.sin(circle_theta)
        circle_z = np.zeros_like(circle_x)
        
        ring_x = point[0] + circle_x
        ring_y = point[1] + circle_y  
        ring_z = point[2] + circle_z
        
        ring_alpha = alpha * (0.6 - i * 0.05)
        ring_alpha = max(ring_alpha, 0.1)
        
        ax.plot(ring_x, ring_y, ring_z, color=color, alpha=ring_alpha, linewidth=1)


def plot_target_3d(ax, target_pos: np.ndarray, drone_size: float, 
                   velocity: Optional[np.ndarray] = None, target_id: str = "Target",
                   color: str = 'orange'):
    """Draw 3D drone target representation"""
    # Drone body (sphere)
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 10)
    radius = drone_size / 2
    
    x = target_pos[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = target_pos[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = target_pos[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
    
    ax.plot_surface(x, y, z, alpha=0.7, color=color)
    
    # Target marker
    ax.scatter(target_pos[0], target_pos[1], target_pos[2], 
               c=color, s=150, marker='o', label=target_id)
    
    # Velocity vector if provided
    if velocity is not None and np.linalg.norm(velocity) > 0.1:
        vel_scale = 3.0
        ax.quiver(target_pos[0], target_pos[1], target_pos[2],
                 velocity[0], velocity[1], velocity[2],
                 length=vel_scale, normalize=True, color='red', 
                 arrow_length_ratio=0.1, alpha=0.8)


def create_multi_array_engagement_plot(config: Dict, topology: str = 'grid_2x2', 
                                     num_targets: int = 2) -> plt.Figure:
    """Create multi-cannon array engagement visualization"""
    if not MULTI_CANNON_AVAILABLE:
        raise ImportError("Multi-cannon functionality not available")
    
    # Create array
    topology_map = {
        'line': ArrayTopology.LINE,
        'grid_2x2': ArrayTopology.GRID_2x2,
        'grid_3x3': ArrayTopology.GRID_3x3,
        'circle': ArrayTopology.CIRCLE
    }
    
    array_topology = topology_map.get(topology, ArrayTopology.GRID_2x2)
    array = create_test_array(array_topology, FiringMode.COORDINATED)
    
    # Create test targets
    targets = []
    target_positions = [
        [25, 0, 15],
        [35, 20, 18],
        [40, -15, 12],
        [30, 30, 25],
        [20, -20, 20]
    ]
    
    drone_types = ['small', 'medium', 'large']
    for i in range(min(num_targets, len(target_positions))):
        drone_type = drone_types[i % len(drone_types)]
        drone_spec = config['drone_models'][drone_type]
        
        target = Target(
            id=f"{drone_type}_drone_{i+1}",
            position=np.array(target_positions[i]),
            velocity=np.array([np.random.uniform(-3, 3), np.random.uniform(-3, 3), 0]),
            size=drone_spec['size'],
            vulnerability=drone_spec['vulnerability'],
            priority=i+1,
            detected_time=0.0
        )
        targets.append(target)
    
    # Execute engagement
    results = array.execute_engagement_sequence(targets)
    
    # Create visualization
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the multi-cannon array
    plot_multi_cannon_array(ax, array, targets, results)
    
    # Plot targets
    for i, target in enumerate(targets):
        colors = ['orange', 'red', 'purple', 'brown', 'pink']
        plot_target_3d(ax, target.position, target.size, target.velocity, 
                      target.id, colors[i % len(colors)])
    
    # Plot trajectories for successful engagements
    trajectory_colors = ['blue', 'green', 'cyan', 'magenta', 'yellow']
    for i, result in enumerate(results):
        if result.get('success', False) and 'assigned_cannons' in result:
            for j, cannon_id in enumerate(result['assigned_cannons']):
                if cannon_id < len(array.cannons):
                    cannon = array.cannons[cannon_id]
                    target = targets[result.get('target_index', i)]
                    
                    # Create simplified vortex ring for visualization
                    vr = cannon.generate_vortex_ring(target.position)
                    color = trajectory_colors[(i*2 + j) % len(trajectory_colors)]
                    
                    plot_vortex_trajectory(ax, cannon.position, target.position, 
                                         vr, {'flight_time': 2.0}, color, alpha=0.6)
    
    # Formatting
    ax.set_xlabel('X Distance (m)')
    ax.set_ylabel('Y Distance (m)')
    ax.set_zlabel('Z Altitude (m)')
    
    # Set axis limits based on array and targets
    all_positions = [cannon.position for cannon in array.cannons] + [target.position for target in targets]
    all_positions = np.array(all_positions)
    
    margin = 15
    ax.set_xlim(np.min(all_positions[:, 0]) - margin, np.max(all_positions[:, 0]) + margin)
    ax.set_ylim(np.min(all_positions[:, 1]) - margin, np.max(all_positions[:, 1]) + margin)
    ax.set_zlim(0, np.max(all_positions[:, 2]) + margin)
    
    # Title and statistics
    successful_engagements = sum(1 for r in results if r.get('success', False))
    total_targets = len(targets)
    success_rate = successful_engagements / total_targets if total_targets > 0 else 0
    
    title = f"Multi-Cannon Array Engagement - {topology.upper()}\n"
    title += f"Array: {len(array.cannons)} cannons | Targets: {total_targets} | "
    title += f"Success Rate: {success_rate:.1%}"
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    return fig


def create_array_topology_comparison(config: Dict) -> plt.Figure:
    """Create comparison of different array topologies"""
    if not MULTI_CANNON_AVAILABLE:
        raise ImportError("Multi-cannon functionality not available")
    
    fig = plt.figure(figsize=(20, 12))
    
    topologies = [
        (ArrayTopology.LINE, 'Linear Array'),
        (ArrayTopology.GRID_2x2, '2x2 Grid'),
        (ArrayTopology.GRID_3x3, '3x3 Grid'),
        (ArrayTopology.CIRCLE, 'Circular Array')
    ]
    
    for i, (topology, title) in enumerate(topologies):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
        try:
            # Create array
            array = create_test_array(topology, FiringMode.COORDINATED)
            
            # Create a single test target
            target = Target(
                id="test_target",
                position=np.array([30, 15, 18]),
                velocity=np.array([-2, 0, 0]),
                size=0.6,
                vulnerability=0.7,
                priority=1,
                detected_time=0.0
            )
            
            # Plot array and target
            plot_multi_cannon_array(ax, array, [target])
            plot_target_3d(ax, target.position, target.size, target.velocity, "Target")
            
            # Calculate coverage metrics
            cannon_positions = np.array([cannon.position for cannon in array.cannons])
            array_span = np.max(cannon_positions, axis=0) - np.min(cannon_positions, axis=0)
            coverage_area = array_span[0] * array_span[1]
            
            ax.set_title(f"{title}\nCannons: {len(array.cannons)} | Coverage: {coverage_area:.0f}m²", 
                        fontsize=12, fontweight='bold')
            
        except Exception as e:
            ax.text(0.5, 0.5, 0.5, f"Error: {str(e)}", transform=ax.transData, 
                   ha='center', va='center', fontsize=10)
            ax.set_title(f"{title} - Error", fontsize=12)
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.grid(True, alpha=0.3)
        
        # Set consistent axis limits
        ax.set_xlim(-40, 40)
        ax.set_ylim(-40, 40)
        ax.set_zlim(0, 30)
    
    fig.suptitle('Multi-Cannon Array Topology Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    return fig


def create_enhanced_envelope_plot(config: Dict, drone_type: str = 'small', 
                                array_size: int = 1) -> plt.Figure:
    """Create enhanced envelope plot with array performance comparison"""
    if array_size == 1:
        # Use original single-cannon envelope
        cannon = create_cannon_from_config(config)
        calc = EngagementCalculator(cannon)
        envelope_data = calc.engagement_envelope_analysis(drone_type)
        
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111)
        
        ranges = np.array(envelope_data['ranges'])
        elevations = np.array(envelope_data['elevations'])
        kill_matrix = np.array(envelope_data['kill_probability_matrix'])
        
        contour_levels = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
        cs = ax.contourf(ranges, elevations, kill_matrix, 
                        levels=contour_levels, cmap='RdYlGn', alpha=0.8)
        
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Elevation (degrees)')
        ax.set_title(f'Single Cannon Engagement Envelope - {drone_type.title()} Drone')
        plt.colorbar(cs, label='Kill Probability')
        
    else:
        # Multi-cannon envelope analysis
        if not MULTI_CANNON_AVAILABLE:
            raise ImportError("Multi-cannon functionality not available")
        
        fig = plt.figure(figsize=(18, 10))
        
        # Compare single vs multi-cannon
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        # Single cannon baseline
        cannon = create_cannon_from_config(config)
        calc = EngagementCalculator(cannon)
        single_envelope = calc.engagement_envelope_analysis(drone_type)
        
        ranges = np.array(single_envelope['ranges'])
        elevations = np.array(single_envelope['elevations'])
        single_kill_matrix = np.array(single_envelope['kill_probability_matrix'])
        
        # Multi-cannon simulation (simplified)
        multi_kill_matrix = np.minimum(single_kill_matrix * array_size * 0.7, 1.0)
        
        contour_levels = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
        
        cs1 = ax1.contourf(ranges, elevations, single_kill_matrix, 
                          levels=contour_levels, cmap='RdYlGn', alpha=0.8)
        ax1.set_title(f'Single Cannon - {drone_type.title()} Drone')
        ax1.set_xlabel('Range (m)')
        ax1.set_ylabel('Elevation (degrees)')
        
        cs2 = ax2.contourf(ranges, elevations, multi_kill_matrix, 
                          levels=contour_levels, cmap='RdYlGn', alpha=0.8)
        ax2.set_title(f'{array_size}-Cannon Array - {drone_type.title()} Drone')
        ax2.set_xlabel('Range (m)')
        ax2.set_ylabel('Elevation (degrees)')
        
        # Shared colorbar
        fig.colorbar(cs2, ax=[ax1, ax2], label='Kill Probability', shrink=0.8)
    
    plt.tight_layout()
    return fig


# Keep original functions for backward compatibility
def plot_engagement_zones(ax, cannon_pos: np.ndarray, max_range: float = 60):
    """Plot engagement effectiveness zones"""
    ranges = [15, 30, 45, 60]
    colors = ['green', 'yellow', 'orange', 'red']
    alphas = [0.3, 0.2, 0.15, 0.1]
    labels = ['Optimal', 'Effective', 'Marginal', 'Maximum']
    
    for r, color, alpha, label in zip(ranges, colors, alphas, labels):
        phi = np.linspace(0, np.pi/2, 20)
        theta = np.linspace(0, 2*np.pi, 40)
        
        phi_mesh, theta_mesh = np.meshgrid(phi, theta)
        
        x = cannon_pos[0] + r * np.sin(phi_mesh) * np.cos(theta_mesh)
        y = cannon_pos[1] + r * np.sin(phi_mesh) * np.sin(theta_mesh)
        z = cannon_pos[2] + r * np.cos(phi_mesh)
        
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
    ax.grid(True, alpha=0.3)
    
    return fig


def create_envelope_plot(config: Dict, drone_type: str = 'small') -> plt.Figure:
    """Create improved engagement envelope visualization with better readability"""
    cannon = create_cannon_from_config(config)
    calc = EngagementCalculator(cannon)
    
    print(f"Calculating engagement envelope for {drone_type} drones...")
    envelope_data = calc.engagement_envelope_analysis(drone_type)
    
    # Create figure with better layout
    fig = plt.figure(figsize=(18, 8))
    
    # Kill probability contour plot (left)
    ax1 = plt.subplot(131)
    ranges = np.array(envelope_data['ranges'])
    elevations = np.array(envelope_data['elevations'])
    kill_matrix = np.array(envelope_data['kill_probability_matrix'])
    
    # Create filled contour plot instead of heatmap
    contour_levels = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    colors = ['#d73027', '#fc8d59', '#fee08b', '#d9ef8b', '#91d1c2', '#4575b4']
    
    cs1 = ax1.contourf(ranges, elevations, kill_matrix, 
                       levels=contour_levels, colors=colors, alpha=0.8)
    
    # Add clear contour lines
    cs1_lines = ax1.contour(ranges, elevations, kill_matrix, 
                           levels=contour_levels[1:-1], colors='black', linewidths=1.5)
    ax1.clabel(cs1_lines, inline=True, fontsize=10, fmt='%.1f')
    
    ax1.set_xlabel('Range (m)', fontsize=12)
    ax1.set_ylabel('Elevation (degrees)', fontsize=12)
    ax1.set_title(f'Kill Probability - {drone_type.title()} Drone', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Mark optimal point
    optimal_range = envelope_data['optimal_range']
    optimal_elev = envelope_data['optimal_elevation']
    ax1.plot(optimal_range, optimal_elev, 'r*', markersize=15, markeredgecolor='black', 
             markeredgewidth=1, label=f'Optimal: {optimal_range}m @ {optimal_elev}°')
    ax1.legend(loc='upper right')
    
    # Effectiveness zones (middle)
    ax2 = plt.subplot(132)
    
    # Create discrete effectiveness zones
    effectiveness = np.zeros_like(kill_matrix)
    effectiveness[kill_matrix >= 0.7] = 3  # High effectiveness
    effectiveness[(kill_matrix >= 0.5) & (kill_matrix < 0.7)] = 2  # Medium
    effectiveness[(kill_matrix >= 0.3) & (kill_matrix < 0.5)] = 1  # Low
    effectiveness[kill_matrix < 0.3] = 0  # Ineffective
    
    zone_colors = ['red', 'orange', 'yellow', 'green']
    zone_labels = ['Ineffective\n(<0.3)', 'Low\n(0.3-0.5)', 'Medium\n(0.5-0.7)', 'High\n(≥0.7)']
    
    im2 = ax2.imshow(effectiveness, extent=[ranges.min(), ranges.max(), 
                                          elevations.min(), elevations.max()],
                     aspect='auto', origin='lower', cmap='RdYlGn', vmin=0, vmax=3)
    
    ax2.set_xlabel('Range (m)', fontsize=12)
    ax2.set_ylabel('Elevation (degrees)', fontsize=12)
    ax2.set_title('Effectiveness Zones', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, color='white')
    
    # Custom colorbar for effectiveness zones
    cbar2 = plt.colorbar(im2, ax=ax2, ticks=[0.375, 1.125, 1.875, 2.625])
    cbar2.ax.set_yticklabels(zone_labels)
    cbar2.set_label('Effectiveness Level', fontsize=12)
    
    # Performance summary (right)
    ax3 = plt.subplot(133)
    ax3.axis('off')
    
    # Performance metrics text
    max_range = envelope_data['max_effective_range']
    max_kill_prob = envelope_data['max_kill_probability']
    
    # Calculate zone statistics
    total_points = kill_matrix.size
    high_eff_points = np.sum(kill_matrix >= 0.7)
    medium_eff_points = np.sum((kill_matrix >= 0.5) & (kill_matrix < 0.7))
    low_eff_points = np.sum((kill_matrix >= 0.3) & (kill_matrix < 0.5))
    ineffective_points = np.sum(kill_matrix < 0.3)
    
    summary_text = f"""
ENGAGEMENT ENVELOPE SUMMARY
{drone_type.title()} Drone Performance

KEY METRICS:
• Max Effective Range: {max_range:.0f} m
• Optimal Point: {optimal_range:.0f}m @ {optimal_elev:.0f}°
• Peak Kill Probability: {max_kill_prob:.3f}

COVERAGE ANALYSIS:
• High Effectiveness: {high_eff_points/total_points*100:.1f}%
  (P_kill ≥ 0.7)
• Medium Effectiveness: {medium_eff_points/total_points*100:.1f}%
  (P_kill 0.5-0.7)
• Low Effectiveness: {low_eff_points/total_points*100:.1f}%
  (P_kill 0.3-0.5)
• Ineffective: {ineffective_points/total_points*100:.1f}%
  (P_kill < 0.3)

OPERATIONAL ENVELOPE:
• Close Range (5-20m): Optimal zone
• Medium Range (20-40m): Effective zone  
• Extended Range (40-60m): Marginal zone
• Maximum Range (60-80m): Limited effectiveness

DEPLOYMENT GUIDANCE:
• Position targets in 20-40m range for best results
• Elevation angles 20-40° provide optimal performance
• System shows clear operational boundaries
    """
    
    ax3.text(0.05, 0.95, summary_text, transform=ax3.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Overall title
    fig.suptitle(f'Vortex Cannon Engagement Envelope Analysis - {drone_type.title()} Drone\n'
                f'Single Cannon Configuration', fontsize=16, fontweight='bold')
    
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
        description="Enhanced Vortex Cannon Engagement Visualizer with Multi-Array Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single target engagement
  python scripts/visualize.py --target-x 30 --target-y 10 --target-z 15 --drone-size small
  
  # Multi-cannon array engagement
  python scripts/visualize.py --multi-array --topology grid_2x2 --targets 3
  
  # Array topology comparison
  python scripts/visualize.py --array-comparison
  
  # Enhanced envelope analysis
  python scripts/visualize.py --envelope-plot --drone-type medium --array-size 4
  
  # Trajectory analysis
  python scripts/visualize.py --trajectory-analysis
        """
    )
    
    # Mode selection - enhanced with multi-cannon options
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument('--envelope-plot', action='store_true',
                           help='Generate engagement envelope heatmap')
    mode_group.add_argument('--trajectory-analysis', action='store_true',
                           help='Generate vortex ring trajectory analysis')
    mode_group.add_argument('--multi-array', action='store_true',
                           help='Generate multi-cannon array engagement visualization')
    mode_group.add_argument('--array-comparison', action='store_true',
                           help='Generate array topology comparison')
    
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
    
    # Multi-cannon array parameters
    parser.add_argument('--topology', choices=['line', 'grid_2x2', 'grid_3x3', 'circle'], 
                       default='grid_2x2', help='Array topology for multi-cannon visualization')
    parser.add_argument('--targets', type=int, default=2,
                       help='Number of targets for multi-cannon engagement')
    
    # Enhanced envelope analysis
    parser.add_argument('--drone-type', choices=['small', 'medium', 'large'], default='small',
                       help='Drone type for envelope analysis')
    parser.add_argument('--array-size', type=int, default=1,
                       help='Number of cannons in array for envelope analysis')
    
    # Configuration and output
    parser.add_argument('--config', type=str, default='config/cannon_specs.yaml',
                       help='Configuration file path')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file path (PNG, PDF, etc.)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Output resolution (DPI)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not any([args.envelope_plot, args.trajectory_analysis, args.multi_array, args.array_comparison]):
        if args.target_x is None or args.target_z is None:
            parser.error("Single engagement plot requires --target-x and --target-z")
    
    # Check multi-cannon availability for relevant modes
    if (args.multi_array or args.array_comparison or args.array_size > 1) and not MULTI_CANNON_AVAILABLE:
        parser.error("Multi-cannon functionality not available. Please ensure multi_cannon_array.py is accessible.")
    
    # Load configuration
    config = load_config_with_defaults(args.config)
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    try:
        if args.envelope_plot:
            if args.array_size > 1:
                fig = create_enhanced_envelope_plot(config, args.drone_type, args.array_size)
            else:
                fig = create_envelope_plot(config, args.drone_type)
        elif args.trajectory_analysis:
            fig = create_trajectory_analysis(config)
        elif args.multi_array:
            fig = create_multi_array_engagement_plot(config, args.topology, args.targets)
        elif args.array_comparison:
            fig = create_array_topology_comparison(config)
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