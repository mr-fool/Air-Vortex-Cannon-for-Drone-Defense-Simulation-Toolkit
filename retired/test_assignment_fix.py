#!/usr/bin/env python3
"""
Test script to verify multi-cannon assignment fix is working
"""

import sys
import os
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Test the fixed assignment logic
def test_multicanno_assignment():
    """Test that large targets get multiple cannons assigned"""
    print("TESTING MULTI-CANNON ASSIGNMENT FIX")
    print("="*50)
    
    try:
        # Create test array
        from multi_cannon_array import create_test_array, ArrayTopology, FiringMode
        from engagement import Target
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure you've applied the fixes to src/multi_cannon_array.py first")
        return False
    
    # Create 3x3 array (9 cannons)
    array = create_test_array(ArrayTopology.GRID_3x3, FiringMode.ADAPTIVE)
    print(f"Created array with {len(array.cannons)} cannons")
    
    # Create test targets of different sizes
    targets = [
        Target("small_1", np.array([25, 5, 15]), np.zeros(3), 0.3, 0.65, 1, 0.0),  # Small
        Target("medium_1", np.array([30, 0, 18]), np.zeros(3), 0.6, 0.45, 1, 0.0),  # Medium  
        Target("large_1", np.array([32, -8, 20]), np.zeros(3), 1.2, 0.1, 1, 0.0),   # Large
        Target("xl_1", np.array([35, 5, 22]), np.zeros(3), 2.0, 0.05, 1, 0.0),      # Extra Large
    ]
    
    print(f"Testing with {len(targets)} targets:")
    for target in targets:
        print(f"  {target.id}: size={target.size}m")
    
    print("\nExecuting engagement sequence...")
    results = array.execute_engagement_sequence(targets)
    
    print(f"\nRESULTS:")
    print(f"{'Target':<10} {'Size':<6} {'Cannons':<8} {'Success':<8} {'P_kill':<8}")
    print("-" * 50)
    
    for result in results:
        target = next(t for t in targets if t.id == result['target_id'])
        print(f"{result['target_id']:<10} {target.size:<6.1f} "
              f"{result.get('participating_cannons', 0):<8} "
              f"{result.get('success', False):<8} "
              f"{result.get('combined_kill_probability', 0.0):<8.3f}")
    
    # Verify the fix worked
    large_result = next((r for r in results if 'large' in r['target_id']), None)
    xl_result = next((r for r in results if 'xl' in r['target_id']), None)
    
    print(f"\nVERIFICATION:")
    if large_result and large_result.get('participating_cannons', 0) > 1:
        print(f"✓ FIXED: Large target got {large_result['participating_cannons']} cannons")
    else:
        print(f"✗ STILL BROKEN: Large target got {large_result.get('participating_cannons', 0) if large_result else 'no'} cannons")
    
    if xl_result and xl_result.get('participating_cannons', 0) > 1:
        print(f"✓ FIXED: XL target got {xl_result['participating_cannons']} cannons")
    else:
        print(f"✗ STILL BROKEN: XL target got {xl_result.get('participating_cannons', 0) if xl_result else 'no'} cannons")

if __name__ == "__main__":
    test_multicanno_assignment()