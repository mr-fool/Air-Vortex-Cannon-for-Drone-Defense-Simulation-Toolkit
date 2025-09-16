#!/usr/bin/env python3
"""
Debug script to trace multi-cannon assignment issues
"""

import sys
import os
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from multi_cannon_array import create_test_array, ArrayTopology, FiringMode
from engagement import Target

def debug_assignment_issue():
    """Debug why medium/large targets get 0 cannons assigned"""
    
    print("DEBUGGING MULTI-CANNON ASSIGNMENT ISSUE")
    print("="*60)
    
    # Create array
    array = create_test_array(ArrayTopology.GRID_2x2, FiringMode.ADAPTIVE)
    
    # Create test targets
    targets = [
        Target("small_target", np.array([30, 0, 18]), np.array([0, 0, 0]), 0.3, 0.65, 1, 0.0),
        Target("medium_target", np.array([30, 0, 18]), np.array([0, 0, 0]), 0.6, 0.45, 1, 0.0),
        Target("large_target", np.array([30, 0, 18]), np.array([0, 0, 0]), 1.2, 0.1, 1, 0.0)
    ]
    
    print(f"Array has {len(array.cannons)} cannons")
    for i, cannon in enumerate(array.cannons):
        print(f"  Cannon {i}: position {cannon.position}")
    
    print("\nTesting each target individually...")
    
    for target in targets:
        print(f"\n--- DEBUGGING TARGET: {target.id} (size={target.size}) ---")
        
        # Test assignment
        assignment_result = array.assign_targets([target])
        assignments = assignment_result['assignments']
        
        print(f"Assignment result: {assignments}")
        
        if target.id in assignments:
            assigned_cannon_ids = assignments[target.id]
            if isinstance(assigned_cannon_ids, list):
                print(f"Assigned cannons: {assigned_cannon_ids}")
            else:
                print(f"Assigned cannon: {assigned_cannon_ids}")
                assigned_cannon_ids = [assigned_cannon_ids]
            
            # Test individual cannon engagement capabilities
            print("Testing individual cannon capabilities:")
            for cannon_id in assigned_cannon_ids:
                cannon_unit = next((c for c in array.cannons if c.id == cannon_id), None)
                if cannon_unit:
                    range_to_target = np.linalg.norm(target.position - cannon_unit.position)
                    print(f"  {cannon_id}: range={range_to_target:.1f}m")
                    
                    # Test can_engage_target
                    try:
                        can_engage, reason = cannon_unit.cannon.can_engage_target(target.position)
                        print(f"    can_engage_target: {can_engage} (reason: {reason})")
                    except Exception as e:
                        print(f"    can_engage_target ERROR: {e}")
                    
                    # Test actual engagement calculation
                    from engagement import EngagementCalculator
                    try:
                        calc = EngagementCalculator(cannon_unit.cannon)
                        solution = calc.single_target_engagement(target, 0.0)
                        print(f"    Engagement solution: success={solution.success}, kill_prob={solution.kill_probability:.3f}")
                    except Exception as e:
                        print(f"    Engagement calculation ERROR: {e}")
        else:
            print("No cannons assigned!")
        
        # Test combined engagement
        print("\nTesting combined engagement:")
        try:
            results = array.execute_engagement_sequence([target])
            if results:
                result = results[0]
                print(f"  Success: {result['success']}")
                print(f"  Participating cannons: {result.get('participating_cannons', 0)}")
                print(f"  Kill probability: {result.get('combined_kill_probability', 0):.3f}")
                if 'individual_solutions' in result:
                    print(f"  Individual solutions: {len(result['individual_solutions'])}")
                    for i, sol in enumerate(result['individual_solutions']):
                        print(f"    Solution {i}: success={sol.success}, energy={sol.impact_energy:.0f}J")
        except Exception as e:
            print(f"  Combined engagement ERROR: {e}")

def debug_cannon_capabilities():
    """Debug individual cannon engagement capabilities"""
    
    print("\n" + "="*60)
    print("DEBUGGING CANNON CAPABILITIES")
    print("="*60)
    
    # Create single cannon
    array = create_test_array(ArrayTopology.GRID_2x2, FiringMode.ADAPTIVE)
    cannon = array.cannons[0]
    
    # Test positions at different ranges and elevations
    test_positions = [
        np.array([20, 0, 15]),   # Close, medium elevation
        np.array([30, 0, 18]),   # Medium range
        np.array([40, 0, 20]),   # Far range
        np.array([50, 0, 25]),   # Very far
        np.array([30, 0, 5]),    # Low elevation
        np.array([30, 0, 35]),   # High elevation
    ]
    
    print(f"Cannon position: {cannon.position}")
    print(f"Cannon config - Max elevation: {cannon.cannon.config.max_elevation}°")
    print(f"Cannon config - Barrel length: {cannon.cannon.config.barrel_length}m")
    
    print("\nTesting engagement capability at different positions:")
    print(f"{'Position':<20} {'Range':<8} {'Elevation':<10} {'Can Engage':<12} {'Reason':<30}")
    print("-" * 80)
    
    for pos in test_positions:
        range_to_pos = np.linalg.norm(pos - cannon.position)
        
        # Calculate elevation angle
        dx = range_to_pos
        dz = pos[2] - cannon.position[2]
        elevation_angle = np.degrees(np.arctan2(dz, dx))
        
        try:
            can_engage, reason = cannon.cannon.can_engage_target(pos)
            print(f"{str(pos):<20} {range_to_pos:<8.1f} {elevation_angle:<10.1f} {str(can_engage):<12} {reason:<30}")
        except Exception as e:
            print(f"{str(pos):<20} {range_to_pos:<8.1f} {elevation_angle:<10.1f} {'ERROR':<12} {str(e):<30}")

if __name__ == "__main__":
    debug_assignment_issue()
    debug_cannon_capabilities()
