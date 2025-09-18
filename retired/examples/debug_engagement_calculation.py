#!/usr/bin/env python3
"""
Debug script to trace engagement calculation failures
"""

import sys
import os
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multi_cannon_array import create_test_array, ArrayTopology, FiringMode
from engagement import Target, EngagementCalculator
from vortex_ring import VortexRing

def debug_engagement_calculation():
    """Debug why individual engagement calculations are failing"""
    
    print("DEBUGGING ENGAGEMENT CALCULATION FAILURE")
    print("="*60)
    
    # Create array and get a cannon
    array = create_test_array(ArrayTopology.GRID_2x2, FiringMode.ADAPTIVE)
    cannon_unit = array.cannons[2]  # cannon_03 from debug output
    cannon = cannon_unit.cannon
    
    print(f"Testing cannon: {cannon_unit.id}")
    print(f"Cannon position: {cannon_unit.position}")
    print(f"Cannon config:")
    print(f"  Barrel length: {cannon.config.barrel_length}m")
    print(f"  Barrel diameter: {cannon.config.barrel_diameter}m")
    print(f"  Chamber pressure: {cannon.config.chamber_pressure/1000:.0f} kPa")
    print(f"  Max chamber pressure: {cannon.config.max_chamber_pressure/1000:.0f} kPa")
    
    # Test targets
    targets = [
        Target("small", np.array([30, 0, 18]), np.array([0, 0, 0]), 0.3, 0.65, 1, 0.0),
        Target("medium", np.array([30, 0, 18]), np.array([0, 0, 0]), 0.6, 0.45, 1, 0.0),
        Target("large", np.array([30, 0, 18]), np.array([0, 0, 0]), 1.2, 0.1, 1, 0.0)
    ]
    
    for target in targets:
        print(f"\n--- DEBUGGING TARGET: {target.id} (size={target.size}, vuln={target.vulnerability}) ---")
        
        range_to_target = np.linalg.norm(target.position - cannon_unit.position)
        print(f"Range to target: {range_to_target:.1f}m")
        
        # Test basic capability
        can_engage, reason = cannon.can_engage_target(target.position)
        print(f"Can engage: {can_engage} ({reason})")
        
        if can_engage:
            # Step 1: Test vortex ring generation
            print("\n1. Testing vortex ring generation:")
            try:
                vr = cannon.generate_vortex_ring(target.position)
                print(f"   Initial velocity: {vr.velocity:.1f} m/s")
                print(f"   Initial energy: {vr.kinetic_energy:.0f} J")
                print(f"   Formation number: {vr.formation_number:.1f}")
                
                # Test trajectory to target
                flight_time = vr.time_to_range(range_to_target)
                if flight_time > 0:
                    impact_velocity = vr.velocity_at_time(flight_time)
                    impact_energy = 0.5 * vr.mass * impact_velocity**2
                    print(f"   Flight time: {flight_time:.2f} s")
                    print(f"   Impact velocity: {impact_velocity:.1f} m/s")
                    print(f"   Impact energy: {impact_energy:.0f} J")
                else:
                    print(f"   ERROR: Invalid flight time: {flight_time}")
                    continue
            except Exception as e:
                print(f"   ERROR in vortex ring generation: {e}")
                continue
            
            # Step 2: Test trajectory calculation
            print("\n2. Testing trajectory calculation:")
            try:
                calc = EngagementCalculator(cannon)
                
                # Check if calc has the trajectory method
                if hasattr(calc, 'calculate_trajectory'):
                    trajectory = calc.calculate_trajectory(target.position, 0.0)
                    print(f"   Trajectory calculated: {trajectory is not None}")
                    if trajectory:
                        print(f"   Launch angle: {np.degrees(trajectory.launch_angle):.1f}°")
                        print(f"   Time of flight: {trajectory.time_of_flight:.2f} s")
                else:
                    print("   No calculate_trajectory method found")
            except Exception as e:
                print(f"   ERROR in trajectory calculation: {e}")
            
            # Step 3: Test full engagement calculation
            print("\n3. Testing full engagement calculation:")
            try:
                calc = EngagementCalculator(cannon)
                solution = calc.single_target_engagement(target, 0.0)
                
                print(f"   Solution success: {solution.success}")
                print(f"   Kill probability: {solution.kill_probability:.3f}")
                print(f"   Hit probability: {solution.hit_probability:.3f}")
                print(f"   Impact energy: {solution.impact_energy:.0f} J")
                print(f"   Impact time: {solution.impact_time:.2f} s")
                
                if not solution.success:
                    print(f"   Failure reason: {getattr(solution, 'failure_reason', 'Unknown')}")
                
                # Check energy vs vulnerability requirements
                required_energy = 100 / target.vulnerability  # Rough estimate
                print(f"   Required energy estimate: {required_energy:.0f} J")
                print(f"   Energy ratio: {solution.impact_energy / required_energy:.2f}")
                
            except Exception as e:
                print(f"   ERROR in engagement calculation: {e}")
                import traceback
                traceback.print_exc()

def debug_energy_thresholds():
    """Debug energy threshold calculations"""
    
    print("\n" + "="*60)
    print("DEBUGGING ENERGY THRESHOLDS")
    print("="*60)
    
    # Test different target parameters
    test_cases = [
        {"size": 0.3, "vuln": 0.65, "desc": "Small drone"},
        {"size": 0.6, "vuln": 0.45, "desc": "Medium drone"},
        {"size": 1.2, "vuln": 0.1, "desc": "Large drone"},
    ]
    
    print(f"{'Target':<12} {'Size':<6} {'Vuln':<6} {'Energy Req':<12} {'Kill@1000J':<10} {'Kill@2000J':<10}")
    print("-" * 70)
    
    for case in test_cases:
        # Simple energy requirement calculation
        base_energy = 100  # Base energy for 100% vulnerable 1m target
        size_factor = case["size"] ** 2  # Area scaling
        vuln_factor = 1.0 / case["vuln"]  # Vulnerability scaling
        
        required_energy = base_energy * size_factor * vuln_factor
        
        # Kill probability at different energies
        kill_prob_1000 = min(1.0, (1000 / required_energy) * case["vuln"])
        kill_prob_2000 = min(1.0, (2000 / required_energy) * case["vuln"])
        
        print(f"{case['desc']:<12} {case['size']:<6.1f} {case['vuln']:<6.2f} "
              f"{required_energy:<12.0f} {kill_prob_1000:<10.3f} {kill_prob_2000:<10.3f}")

def debug_vortex_physics():
    """Debug vortex ring physics parameters"""
    
    print("\n" + "="*60)
    print("DEBUGGING VORTEX RING PHYSICS")
    print("="*60)
    
    # Create cannon
    array = create_test_array(ArrayTopology.GRID_2x2, FiringMode.ADAPTIVE)
    cannon = array.cannons[0].cannon
    
    print(f"Cannon chamber pressure: {cannon.config.chamber_pressure/1000:.0f} kPa")
    print(f"Formation number: {cannon.config.formation_number:.1f}")
    
    # Test vortex ring at different ranges
    test_ranges = [20, 30, 40, 50]
    
    print(f"\n{'Range(m)':<8} {'Initial V':<10} {'Init Energy':<12} {'Impact V':<10} {'Impact E':<10} {'Flight T':<8}")
    print("-" * 70)
    
    for range_m in test_ranges:
        target_pos = np.array([range_m, 0, 18])
        
        try:
            vr = cannon.generate_vortex_ring(target_pos)
            flight_time = vr.time_to_range(range_m)
            
            if flight_time > 0:
                impact_velocity = vr.velocity_at_time(flight_time)
                impact_energy = 0.5 * vr.mass * impact_velocity**2
                
                print(f"{range_m:<8} {vr.velocity:<10.1f} {vr.kinetic_energy:<12.0f} "
                      f"{impact_velocity:<10.1f} {impact_energy:<10.0f} {flight_time:<8.2f}")
            else:
                print(f"{range_m:<8} {'ERROR':<10} {'ERROR':<12} {'ERROR':<10} {'ERROR':<10} {'ERROR':<8}")
        except Exception as e:
            print(f"{range_m:<8} ERROR: {e}")

if __name__ == "__main__":
    debug_engagement_calculation()
    debug_energy_thresholds()
    debug_vortex_physics()
