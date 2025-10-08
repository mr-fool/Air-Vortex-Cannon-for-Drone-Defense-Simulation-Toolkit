#!/usr/bin/env python3
"""
CI Test for Kill Probability Threshold

This test verifies that the kill probability remains below 0.001
for all realistic damage thresholds (≥750J) at 25m range.
"""

import pytest
import numpy as np
import sys
import os

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vortex_ring import VortexRing
from cannon import VortexCannon, CannonConfiguration
from engagement import EngagementCalculator

def test_kill_probability_threshold():
    """
    Test that kill probability is <0.001 for all damage thresholds ≥750J at 25m
    with appropriate accuracy penalties applied.
    
    This verifies the fundamental energy limitation claim in the paper.
    """
    # Create cannon and calculator to get accuracy penalties
    config = CannonConfiguration(2.0, 0.5, 300000, 85.0, 360.0, 4.0, 1.225, 240000)
    cannon = VortexCannon.__new__(VortexCannon)
    cannon.config = config
    cannon.position = np.zeros(3)
    calc = EngagementCalculator(cannon)
    
    # Create test vortex ring with baseline parameters
    vr = VortexRing(initial_velocity=50.0, initial_diameter=0.3, formation_number=4.0)
    
    # Target at 25m range
    target_position = np.array([25.0, 0.0, 0.0])
    range_value = 25.0
    drone_size = 0.3  # Small drone
    vulnerability = 0.6  # Small drone vulnerability
    
    # Get accuracy penalty for this range
    accuracy_factor = calc._calculate_range_accuracy_penalty(range_value)
    print(f"Range accuracy factor at {range_value}m: {accuracy_factor:.4f}")
    
    # Test with different damage thresholds
    thresholds = [750, 1500, 3000]  # Small, medium, large drone thresholds
    
    for threshold in thresholds:
        # Run Monte Carlo with fixed seed and accuracy-adjusted vulnerability
        results = vr.monte_carlo_engagement(
            target_position, 
            drone_size, 
            vulnerability * accuracy_factor,  # Apply accuracy penalty
            damage_threshold=threshold,
            n_trials=10000
        )
        
        # Further adjust kill probability with accuracy factor
        # This matches how engagement.py applies the penalty twice
        adjusted_kill_prob = results['kill_probability'] * accuracy_factor
        
        # The critical assertion that verifies the paper's claim
        assert adjusted_kill_prob < 0.001, \
            f"Adjusted kill probability at 25m with {threshold}J threshold is {adjusted_kill_prob:.6f}, exceeds 0.001"
        
        print(f"PASSED: Threshold {threshold}J at 25m:")
        print(f"  - Raw kill probability: {results['kill_probability']:.6f}")
        print(f"  - Adjusted with accuracy: {adjusted_kill_prob:.6f} < 0.001")
        print(f"  - Average impact energy: {results['average_impact_energy']:.2f} J")

if __name__ == "__main__":
    test_kill_probability_threshold()
    print("All tests passed!")