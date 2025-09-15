#!/usr/bin/env python3
"""
Vortex Cannon Engagement Calculator - Command Line Interface

This script provides a command-line interface for calculating optimal engagement
solutions for drone targets using vortex cannon systems. Supports single targets,
multiple targets, and performance analysis.

Usage:
    python scripts/engage.py --target-x 30 --target-y 10 --target-z 15 --drone-size small
    python scripts/engage.py --config examples/multi_target_scenario.yaml
    python scripts/engage.py --envelope-analysis --drone-type medium
"""

import argparse
import sys
import os
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

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
            'position': [0.0, 0.0, 2.0],  # 2m elevation
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
            
            # Merge user config with defaults
            for section in default_config:
                if section in user_config:
                    default_config[section].update(user_config[section])
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
            print("Using default configuration.")
    
    return default_config


def create_cannon_from_config(config: Dict) -> VortexCannon:
    """Create and configure cannon from config dictionary"""
    cannon_config = config['cannon']
    vortex_config = config['vortex_ring']
    env_config = config['environment']
    
    # Create configuration object
    config_obj = CannonConfiguration(
        barrel_length=cannon_config['barrel_length'],
        barrel_diameter=cannon_config['barrel_diameter'],
        max_chamber_pressure=cannon_config['max_chamber_pressure'],
        max_elevation=cannon_config['max_elevation'],
        max_traverse=cannon_config['max_traverse'],
        formation_number=vortex_config['formation_number'],
        air_density=env_config['air_density']
    )
    
    # Create cannon object
    cannon = VortexCannon.__new__(VortexCannon)
    cannon.config = config_obj
    cannon.position = np.array(cannon_config.get('position', [0.0, 0.0, 2.0]))
    cannon.orientation = {'elevation': 0.0, 'azimuth': 0.0}
    cannon.chamber_pressure = cannon_config.get('chamber_pressure', 80000)
    cannon.ready_to_fire = True
    cannon.last_shot_time = 0.0
    cannon.reload_time = 0.5
    cannon.pressure_buildup_time = 2.0
    
    return cannon


def print_engagement_results(solution, verbose: bool = False):
    """Print engagement solution results in formatted output"""
    print("=" * 60)
    print(f"ENGAGEMENT SOLUTION: {solution.target_id}")
    print("=" * 60)
    
    if solution.success:
        print("✓ ENGAGEMENT FEASIBLE")
        print(f"  Reason: {solution.reason}")
        print()
        
        print("FIRING SOLUTION:")
        print(f"  Elevation: {solution.elevation:8.2f}°")
        print(f"  Azimuth:   {solution.azimuth:8.2f}°")
        print(f"  Range:     {solution.target_range:8.1f} m")
        print()
        
        print("TIMING:")
        print(f"  Flight time:    {solution.flight_time:6.2f} s")
        print(f"  Impact time:    {solution.impact_time:6.2f} s")
        print(f"  Muzzle velocity: {solution.muzzle_velocity:5.1f} m/s")
        print()
        
        print("EFFECTIVENESS:")
        print(f"  Hit probability:  {solution.hit_probability:6.3f}")
        print(f"  Kill probability: {solution.kill_probability:6.3f}")
        print(f"  Impact energy:    {solution.impact_energy:6.1f} J")
        print(f"  Ring diameter:    {solution.ring_size_at_impact:6.3f} m")
        
        if verbose:
            print()
            print("INTERCEPT DETAILS:")
            print(f"  Intercept position: [{solution.intercept_position[0]:6.2f}, "
                  f"{solution.intercept_position[1]:6.2f}, {solution.intercept_position[2]:6.2f}]")
    else:
        print("✗ ENGAGEMENT NOT FEASIBLE")
        print(f"  Reason: {solution.reason}")
        
        if solution.elevation != 0 or solution.azimuth != 0:
            print()
            print("ATTEMPTED SOLUTION:")
            print(f"  Elevation: {solution.elevation:8.2f}°")
            print(f"  Azimuth:   {solution.azimuth:8.2f}°")
            print(f"  Range:     {solution.target_range:8.1f} m")
            if solution.kill_probability > 0:
                print(f"  Kill prob: {solution.kill_probability:8.3f}")
    
    print("=" * 60)


def single_target_engagement(args, config: Dict):
    """Handle single target engagement calculation"""
    # Create cannon
    cannon = create_cannon_from_config(config)
    calc = EngagementCalculator(cannon)
    
    # Get drone specifications
    drone_models = config['drone_models']
    if args.drone_size not in drone_models:
        print(f"Error: Unknown drone size '{args.drone_size}'")
        print(f"Available sizes: {list(drone_models.keys())}")
        return 1
    
    drone_spec = drone_models[args.drone_size]
    
    # Create target
    target_position = np.array([args.target_x, args.target_y, args.target_z])
    target_velocity = np.array([args.velocity_x, args.velocity_y, args.velocity_z])
    
    target = Target(
        id="user_target",
        position=target_position,
        velocity=target_velocity,
        size=drone_spec['size'],
        vulnerability=drone_spec['vulnerability'],
        priority=1,
        detected_time=0.0
    )
    
    # Print target information
    print("TARGET SPECIFICATION:")
    print(f"  Position: [{target.position[0]:6.1f}, {target.position[1]:6.1f}, {target.position[2]:6.1f}] m")
    if np.linalg.norm(target.velocity) > 0.1:
        print(f"  Velocity: [{target.velocity[0]:6.1f}, {target.velocity[1]:6.1f}, {target.velocity[2]:6.1f}] m/s")
        print(f"  Speed:    {np.linalg.norm(target.velocity):6.1f} m/s")
    else:
        print("  Status:   Stationary")
    print(f"  Size:     {target.size:6.1f} m")
    print(f"  Type:     {args.drone_size} drone")
    print()
    
    print("CANNON CONFIGURATION:")
    print(f"  Position: [{cannon.position[0]:6.1f}, {cannon.position[1]:6.1f}, {cannon.position[2]:6.1f}] m")
    print(f"  Pressure: {cannon.chamber_pressure:6.0f} Pa")
    print(f"  Max velocity: {cannon.calculate_muzzle_velocity():6.1f} m/s")
    print()
    
    # Calculate engagement
    solution = calc.single_target_engagement(target)
    
    # Print results
    print_engagement_results(solution, verbose=args.verbose)
    
    return 0 if solution.success else 1


def multi_target_engagement(args, config: Dict):
    """Handle multiple target engagement from file"""
    if not os.path.exists(args.targets_file):
        print(f"Error: Targets file not found: {args.targets_file}")
        return 1
    
    # Load targets from file
    try:
        with open(args.targets_file, 'r') as f:
            targets_data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading targets file: {e}")
        return 1
    
    # Create cannon
    cannon = create_cannon_from_config(config)
    calc = EngagementCalculator(cannon)
    
    # Create target objects
    targets = []
    drone_models = config['drone_models']
    
    for i, target_data in enumerate(targets_data.get('targets', [])):
        drone_type = target_data.get('drone_size', 'small')
        if drone_type not in drone_models:
            print(f"Warning: Unknown drone type '{drone_type}' for target {i+1}, using 'small'")
            drone_type = 'small'
        
        drone_spec = drone_models[drone_type]
        
        target = Target(
            id=target_data.get('id', f"target_{i+1}"),
            position=np.array(target_data['position']),
            velocity=np.array(target_data.get('velocity', [0, 0, 0])),
            size=drone_spec['size'],
            vulnerability=drone_spec['vulnerability'],
            priority=target_data.get('priority', 1),
            detected_time=target_data.get('detected_time', 0.0)
        )
        targets.append(target)
    
    print(f"MULTI-TARGET ENGAGEMENT: {len(targets)} targets")
    print("=" * 60)
    
    # Calculate engagement sequence
    solutions = calc.multi_target_engagement(targets)
    
    # Print summary
    successful_engagements = sum(1 for sol in solutions if sol.success)
    total_kill_probability = sum(sol.kill_probability for sol in solutions if sol.success)
    
    print("ENGAGEMENT SEQUENCE SUMMARY:")
    print(f"  Total targets:      {len(targets)}")
    print(f"  Successful engages: {successful_engagements}")
    print(f"  Success rate:       {successful_engagements/len(targets)*100:.1f}%")
    if successful_engagements > 0:
        print(f"  Average kill prob:  {total_kill_probability/successful_engagements:.3f}")
    print()
    
    # Print individual results
    for i, solution in enumerate(solutions):
        print(f"SEQUENCE {i+1}:")
        print_engagement_results(solution, verbose=False)
        print()
    
    return 0


def envelope_analysis(args, config: Dict):
    """Perform engagement envelope analysis"""
    cannon = create_cannon_from_config(config)
    calc = EngagementCalculator(cannon)
    
    print(f"ENGAGEMENT ENVELOPE ANALYSIS: {args.drone_type} drone")
    print("=" * 60)
    print("Calculating engagement envelope... (this may take a moment)")
    
    # Perform analysis
    envelope = calc.engagement_envelope_analysis(args.drone_type)
    
    # Print results
    print()
    print("ENGAGEMENT ENVELOPE RESULTS:")
    print(f"  Maximum effective range: {envelope['max_effective_range']:5.0f} m")
    print(f"  Optimal range:          {envelope['optimal_range']:5.0f} m")
    print(f"  Optimal elevation:      {envelope['optimal_elevation']:5.0f}°")
    print(f"  Maximum kill prob:      {envelope['max_kill_probability']:5.3f}")
    print()
    
    # Print range performance
    print("RANGE PERFORMANCE (at optimal elevation):")
    opt_elev_idx = envelope['elevations'].index(envelope['optimal_elevation'])
    kill_probs = envelope['kill_probability_matrix'][opt_elev_idx]
    
    for i, (range_val, kill_prob) in enumerate(zip(envelope['ranges'], kill_probs)):
        if i % 4 == 0:  # Print every 4th value to keep output manageable
            status = "EFFECTIVE" if kill_prob >= calc.min_kill_probability else "marginal"
            print(f"  {range_val:3.0f}m: P_kill = {kill_prob:.3f} ({status})")
    
    return 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Vortex Cannon Engagement Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single stationary target
  python scripts/engage.py --target-x 30 --target-y 10 --target-z 15 --drone-size small
  
  # Moving target
  python scripts/engage.py --target-x 40 --target-y 0 --target-z 20 --drone-size medium \\
                          --velocity-x -5 --velocity-y 2 --velocity-z 0
  
  # Multiple targets from file
  python scripts/engage.py --targets-file examples/swarm_scenario.yaml
  
  # Engagement envelope analysis
  python scripts/engage.py --envelope-analysis --drone-type small
        """
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=False)
    mode_group.add_argument('--targets-file', type=str,
                           help='YAML file containing multiple target definitions')
    mode_group.add_argument('--envelope-analysis', action='store_true',
                           help='Perform engagement envelope analysis')
    
    # Single target parameters
    parser.add_argument('--target-x', type=float, 
                       help='Target X coordinate (meters)')
    parser.add_argument('--target-y', type=float, default=0.0,
                       help='Target Y coordinate (meters, default: 0)')
    parser.add_argument('--target-z', type=float,
                       help='Target Z coordinate (meters, altitude)')
    parser.add_argument('--drone-size', choices=['small', 'medium', 'large'], default='small',
                       help='Drone size category (default: small)')
    
    # Target velocity (for moving targets)
    parser.add_argument('--velocity-x', type=float, default=0.0,
                       help='Target X velocity (m/s, default: 0)')
    parser.add_argument('--velocity-y', type=float, default=0.0,
                       help='Target Y velocity (m/s, default: 0)')
    parser.add_argument('--velocity-z', type=float, default=0.0,
                       help='Target Z velocity (m/s, default: 0)')
    
    # Envelope analysis
    parser.add_argument('--drone-type', choices=['small', 'medium', 'large'], default='small',
                       help='Drone type for envelope analysis (default: small)')
    
    # Configuration
    parser.add_argument('--config', type=str, default='config/cannon_specs.yaml',
                       help='Configuration file path (default: config/cannon_specs.yaml)')
    
    # Output options
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with detailed information')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='Quiet mode - minimal output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.targets_file and not args.envelope_analysis:
        if args.target_x is None or args.target_z is None:
            parser.error("Single target mode requires --target-x and --target-z")
    
    # Load configuration
    config = load_config_with_defaults(args.config)
    
    try:
        # Route to appropriate handler
        if args.envelope_analysis:
            return envelope_analysis(args, config)
        elif args.targets_file:
            return multi_target_engagement(args, config)
        else:
            return single_target_engagement(args, config)
            
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())