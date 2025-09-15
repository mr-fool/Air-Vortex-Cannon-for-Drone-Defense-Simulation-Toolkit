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
from datetime import datetime

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
    
    # Get chamber pressure from config, fall back to 80% of max if not specified
    max_pressure = cannon_config['max_chamber_pressure']
    chamber_pressure = cannon_config.get('chamber_pressure', max_pressure * 0.8)
    
    # Create configuration object with all required parameters
    config_obj = CannonConfiguration(
        barrel_length=cannon_config['barrel_length'],
        barrel_diameter=cannon_config['barrel_diameter'],
        max_chamber_pressure=max_pressure,
        max_elevation=cannon_config.get('max_elevation', 85.0),
        max_traverse=cannon_config.get('max_traverse', 360.0),
        formation_number=vortex_config['formation_number'],
        air_density=env_config['air_density'],
        chamber_pressure=chamber_pressure  # Add the missing parameter
    )
    
    # Create cannon object
    cannon = VortexCannon.__new__(VortexCannon)
    cannon.config = config_obj
    cannon.position = np.array(cannon_config.get('position', [0.0, 0.0, 2.0]))
    cannon.orientation = {'elevation': 0.0, 'azimuth': 0.0}
    cannon.chamber_pressure = chamber_pressure  # Use the configured pressure
    cannon.ready_to_fire = True
    cannon.last_shot_time = 0.0
    cannon.reload_time = 0.5
    cannon.pressure_buildup_time = 2.0
    
    return cannon


def generate_output_filename(mode: str, args) -> str:
    """Generate output filename based on mode and parameters"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if mode == "single":
        return f"results/engagement_single_{timestamp}.txt"
    elif mode == "multi":
        targets_file = os.path.basename(args.targets_file).replace('.yaml', '').replace('.yml', '')
        return f"results/engagement_multi_{targets_file}_{timestamp}.txt"
    elif mode == "envelope":
        return f"results/engagement_envelope_{args.drone_type}_{timestamp}.txt"
    else:
        return f"results/engagement_{timestamp}.txt"


def save_results_to_file(content: str, filename: str):
    """Save results to file, creating directories if needed"""
    try:
        # Create results directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"\nResults saved to: {filename}")
        
    except Exception as e:
        print(f"Warning: Could not save results to file: {e}")


def print_engagement_results(solution, verbose: bool = False, output_lines: List[str] = None):
    """Print engagement solution results in formatted output"""
    lines = []
    
    lines.append("=" * 60)
    lines.append(f"ENGAGEMENT SOLUTION: {solution.target_id}")
    lines.append("=" * 60)
    
    if solution.success:
        lines.append("[SUCCESS] ENGAGEMENT FEASIBLE")
        lines.append(f"  Reason: {solution.reason}")
        lines.append("")
        
        lines.append("FIRING SOLUTION:")
        lines.append(f"  Elevation: {solution.elevation:8.2f} degrees")
        lines.append(f"  Azimuth:   {solution.azimuth:8.2f} degrees")
        lines.append(f"  Range:     {solution.target_range:8.1f} m")
        lines.append("")
        
        lines.append("TIMING:")
        lines.append(f"  Flight time:    {solution.flight_time:6.2f} s")
        lines.append(f"  Impact time:    {solution.impact_time:6.2f} s")
        lines.append(f"  Muzzle velocity: {solution.muzzle_velocity:5.1f} m/s")
        lines.append("")
        
        lines.append("EFFECTIVENESS:")
        lines.append(f"  Hit probability:  {solution.hit_probability:6.3f}")
        lines.append(f"  Kill probability: {solution.kill_probability:6.3f}")
        lines.append(f"  Impact energy:    {solution.impact_energy:6.1f} J")
        lines.append(f"  Ring diameter:    {solution.ring_size_at_impact:6.3f} m")
        
        if verbose:
            lines.append("")
            lines.append("INTERCEPT DETAILS:")
            lines.append(f"  Intercept position: [{solution.intercept_position[0]:6.2f}, "
                        f"{solution.intercept_position[1]:6.2f}, {solution.intercept_position[2]:6.2f}]")
    else:
        lines.append("[FAILED] ENGAGEMENT NOT FEASIBLE")
        lines.append(f"  Reason: {solution.reason}")
        
        if solution.elevation != 0 or solution.azimuth != 0:
            lines.append("")
            lines.append("ATTEMPTED SOLUTION:")
            lines.append(f"  Elevation: {solution.elevation:8.2f} degrees")
            lines.append(f"  Azimuth:   {solution.azimuth:8.2f} degrees")
            lines.append(f"  Range:     {solution.target_range:8.1f} m")
            if solution.kill_probability > 0:
                lines.append(f"  Kill prob: {solution.kill_probability:8.3f}")
    
    lines.append("=" * 60)
    
    # Print to console
    for line in lines:
        print(line)
    
    # Add to output lines if provided
    if output_lines is not None:
        output_lines.extend(lines)


def single_target_engagement(args, config: Dict):
    """Handle single target engagement calculation"""
    output_lines = []
    
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
    
    # Add header info to output
    output_lines.append(f"VORTEX CANNON ENGAGEMENT ANALYSIS")
    output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"Mode: Single Target Engagement")
    output_lines.append("")
    
    # Print target information
    target_info = [
        "TARGET SPECIFICATION:",
        f"  Position: [{target.position[0]:6.1f}, {target.position[1]:6.1f}, {target.position[2]:6.1f}] m"
    ]
    
    if np.linalg.norm(target.velocity) > 0.1:
        target_info.extend([
            f"  Velocity: [{target.velocity[0]:6.1f}, {target.velocity[1]:6.1f}, {target.velocity[2]:6.1f}] m/s",
            f"  Speed:    {np.linalg.norm(target.velocity):6.1f} m/s"
        ])
    else:
        target_info.append("  Status:   Stationary")
    
    target_info.extend([
        f"  Size:     {target.size:6.1f} m",
        f"  Type:     {args.drone_size} drone",
        ""
    ])
    
    for line in target_info:
        print(line)
    output_lines.extend(target_info)
    
    cannon_info = [
        "CANNON CONFIGURATION:",
        f"  Position: [{cannon.position[0]:6.1f}, {cannon.position[1]:6.1f}, {cannon.position[2]:6.1f}] m",
        f"  Pressure: {cannon.chamber_pressure:6.0f} Pa",
        f"  Max velocity: {cannon.calculate_muzzle_velocity():6.1f} m/s",
        ""
    ]
    
    for line in cannon_info:
        print(line)
    output_lines.extend(cannon_info)
    
    # Calculate engagement
    solution = calc.single_target_engagement(target)
    
    # Print results
    print_engagement_results(solution, verbose=args.verbose, output_lines=output_lines)
    
    # Save results to file
    if not args.quiet:
        filename = generate_output_filename("single", args)
        save_results_to_file('\n'.join(output_lines), filename)
    
    return 0 if solution.success else 1


def multi_target_engagement(args, config: Dict):
    """Handle multiple target engagement from file"""
    output_lines = []
    
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
    
    # Add header info to output
    output_lines.append(f"VORTEX CANNON MULTI-TARGET ENGAGEMENT ANALYSIS")
    output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"Targets file: {args.targets_file}")
    output_lines.append("")
    
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
    
    header = f"MULTI-TARGET ENGAGEMENT: {len(targets)} targets"
    print(header)
    print("=" * 60)
    output_lines.append(header)
    output_lines.append("=" * 60)
    
    # Calculate engagement sequence
    solutions = calc.multi_target_engagement(targets)
    
    # Print summary
    successful_engagements = sum(1 for sol in solutions if sol.success)
    total_kill_probability = sum(sol.kill_probability for sol in solutions if sol.success)
    
    summary = [
        "ENGAGEMENT SEQUENCE SUMMARY:",
        f"  Total targets:      {len(targets)}",
        f"  Successful engages: {successful_engagements}",
        f"  Success rate:       {successful_engagements/len(targets)*100:.1f}%"
    ]
    
    if successful_engagements > 0:
        summary.append(f"  Average kill prob:  {total_kill_probability/successful_engagements:.3f}")
    
    summary.append("")
    
    for line in summary:
        print(line)
    output_lines.extend(summary)
    
    # Print individual results
    for i, solution in enumerate(solutions):
        sequence_header = f"SEQUENCE {i+1}:"
        print(sequence_header)
        output_lines.append(sequence_header)
        print_engagement_results(solution, verbose=False, output_lines=output_lines)
        print()
        output_lines.append("")
    
    # Save results to file
    if not args.quiet:
        filename = generate_output_filename("multi", args)
        save_results_to_file('\n'.join(output_lines), filename)
    
    return 0


def envelope_analysis(args, config: Dict):
    """Perform engagement envelope analysis"""
    output_lines = []
    
    cannon = create_cannon_from_config(config)
    calc = EngagementCalculator(cannon)
    
    # Add header info to output
    output_lines.append(f"VORTEX CANNON ENGAGEMENT ENVELOPE ANALYSIS")
    output_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    output_lines.append(f"Drone type: {args.drone_type}")
    output_lines.append("")
    
    header = f"ENGAGEMENT ENVELOPE ANALYSIS: {args.drone_type} drone"
    print(header)
    print("=" * 60)
    print("Calculating engagement envelope... (this may take a moment)")
    
    output_lines.append(header)
    output_lines.append("=" * 60)
    output_lines.append("Calculating engagement envelope... (this may take a moment)")
    
    # Perform analysis
    envelope = calc.engagement_envelope_analysis(args.drone_type)
    
    # Print results
    print()
    output_lines.append("")
    
    results = [
        "ENGAGEMENT ENVELOPE RESULTS:",
        f"  Maximum effective range: {envelope['max_effective_range']:5.0f} m",
        f"  Optimal range:          {envelope['optimal_range']:5.0f} m",
        f"  Optimal elevation:      {envelope['optimal_elevation']:5.0f} degrees",
        f"  Maximum kill prob:      {envelope['max_kill_probability']:5.3f}",
        ""
    ]
    
    for line in results:
        print(line)
    output_lines.extend(results)
    
    # Print range performance
    range_header = "RANGE PERFORMANCE (at optimal elevation):"
    print(range_header)
    output_lines.append(range_header)
    
    opt_elev_idx = envelope['elevations'].index(envelope['optimal_elevation'])
    kill_probs = envelope['kill_probability_matrix'][opt_elev_idx]
    
    for i, (range_val, kill_prob) in enumerate(zip(envelope['ranges'], kill_probs)):
        if i % 4 == 0:  # Print every 4th value to keep output manageable
            status = "EFFECTIVE" if kill_prob >= calc.min_kill_probability else "marginal"
            line = f"  {range_val:3.0f}m: P_kill = {kill_prob:.3f} ({status})"
            print(line)
            output_lines.append(line)
    
    # Save results to file
    if not args.quiet:
        filename = generate_output_filename("envelope", args)
        save_results_to_file('\n'.join(output_lines), filename)
    
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
                       help='Quiet mode - minimal output, no file saving')
    
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
