"""
Target Engagement Calculator

This module handles target engagement calculations, trajectory optimization,
and multi-target engagement sequencing for vortex cannon drone defense systems.
Provides ballistic solutions and engagement effectiveness analysis.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import yaml
from scipy.optimize import minimize_scalar, minimize
import time

# Import cannon and vortex ring classes
from cannon import VortexCannon
from vortex_ring import VortexRing


@dataclass
class Target:
    """Represents a drone target with position and characteristics"""
    id: str
    position: np.ndarray  # [x, y, z] coordinates
    velocity: np.ndarray  # [vx, vy, vz] velocity vector (for moving targets)
    size: float          # Characteristic dimension in meters
    vulnerability: float  # Vulnerability factor (0-1)
    priority: int        # Engagement priority (1=highest)
    detected_time: float # Time when target was detected
    
    def position_at_time(self, future_time: float) -> np.ndarray:
        """Calculate target position at future time (assumes constant velocity)"""
        time_delta = future_time - self.detected_time
        return self.position + self.velocity * time_delta


@dataclass
class EngagementSolution:
    """Complete engagement solution for a target"""
    target_id: str
    success: bool
    reason: str
    
    # Firing solution
    elevation: float
    azimuth: float
    fire_time: float
    impact_time: float
    
    # Predicted performance
    hit_probability: float
    kill_probability: float
    impact_energy: float
    ring_size_at_impact: float
    
    # Ballistic data
    muzzle_velocity: float
    flight_time: float
    target_range: float
    intercept_position: np.ndarray


class EngagementCalculator:
    """
    Calculates optimal engagement solutions for drone targets.
    
    Handles ballistic trajectory calculation, intercept timing,
    and multi-target engagement optimization.
    """
    
    def __init__(self, cannon: VortexCannon, config_path: str = "config/cannon_specs.yaml"):
        """
        Initialize engagement calculator with cannon system.
        
        Args:
            cannon: VortexCannon instance
            config_path: Path to configuration file for drone models
        """
        self.cannon = cannon
        self.drone_models = self._load_drone_models(config_path)
        
        # Engagement parameters
        self.max_engagement_range = 60.0  # meters
        self.min_kill_probability = 0.3   # Minimum acceptable kill probability
        self.prediction_time_limit = 10.0 # Maximum prediction time for moving targets
        
    def _load_drone_models(self, config_path: str) -> Dict:
        """Load drone vulnerability models from configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            return config.get('drone_models', {
                'small': {'mass': 0.5, 'size': 0.3, 'vulnerability': 0.9},
                'medium': {'mass': 2.0, 'size': 0.6, 'vulnerability': 0.7},
                'large': {'mass': 8.0, 'size': 1.2, 'vulnerability': 0.5}
            })
        except FileNotFoundError:
            # Default drone models if config not found
            return {
                'small': {'mass': 0.5, 'size': 0.3, 'vulnerability': 0.9},
                'medium': {'mass': 2.0, 'size': 0.6, 'vulnerability': 0.7},
                'large': {'mass': 8.0, 'size': 1.2, 'vulnerability': 0.5}
            }
    
    def ballistic_solution(self, target_position: np.ndarray) -> Tuple[float, float, float]:
        """
        Calculate ballistic firing solution for stationary target.
        
        Args:
            target_position: Target coordinates [x, y, z]
            
        Returns:
            Tuple of (elevation, azimuth, time_to_target)
        """
        # Generate vortex ring for ballistic calculation
        vr = self.cannon.generate_vortex_ring()
        
        # Calculate direct line solution
        elevation, azimuth = self.cannon.aim_at_target(target_position)
        
        # Calculate time to target
        target_range = np.linalg.norm(target_position - self.cannon.position)
        time_to_target = vr.time_to_range(target_range)
        
        return elevation, azimuth, time_to_target
    
    def intercept_solution(self, target: Target, current_time: float = 0.0) -> Tuple[np.ndarray, float]:
        """
        Calculate intercept point for moving target.
        
        Args:
            target: Target object with position and velocity
            current_time: Current time reference
            
        Returns:
            Tuple of (intercept_position, intercept_time)
        """
        def intercept_error(fire_time):
            """Optimization function to minimize intercept error"""
            # Predict target position at impact time
            vr = self.cannon.generate_vortex_ring()
            
            # Estimate flight time (iterative solution)
            target_pos_at_fire = target.position_at_time(current_time + fire_time)
            range_estimate = np.linalg.norm(target_pos_at_fire - self.cannon.position)
            flight_time = vr.time_to_range(range_estimate)
            
            impact_time = current_time + fire_time + flight_time
            target_pos_at_impact = target.position_at_time(impact_time)
            
            # Calculate actual range to intercept point
            actual_range = np.linalg.norm(target_pos_at_impact - self.cannon.position)
            actual_flight_time = vr.time_to_range(actual_range)
            
            # Error between predicted and actual flight time
            return abs(flight_time - actual_flight_time)
        
        # Optimize fire time to minimize intercept error
        result = minimize_scalar(intercept_error, bounds=(0, self.prediction_time_limit), 
                               method='bounded')
        
        optimal_fire_time = result.x
        
        # Calculate final intercept solution
        vr = self.cannon.generate_vortex_ring()
        target_pos_at_fire = target.position_at_time(current_time + optimal_fire_time)
        range_to_target = np.linalg.norm(target_pos_at_fire - self.cannon.position)
        flight_time = vr.time_to_range(range_to_target)
        
        intercept_time = current_time + optimal_fire_time + flight_time
        intercept_position = target.position_at_time(intercept_time)
        
        return intercept_position, intercept_time
    
    def engagement_effectiveness(self, target: Target, 
                               intercept_position: np.ndarray,
                               n_trials: int = 500) -> Dict:
        """
        Calculate engagement effectiveness using Monte Carlo analysis.
        
        Args:
            target: Target object
            intercept_position: Predicted intercept point
            n_trials: Number of Monte Carlo trials
            
        Returns:
            Dictionary with effectiveness metrics
        """
        # Generate vortex ring for this engagement
        vr = self.cannon.generate_vortex_ring(intercept_position)
        
        # Run Monte Carlo analysis
        relative_position = intercept_position - self.cannon.position
        results = vr.monte_carlo_engagement(
            relative_position,
            target.size,
            target.vulnerability,
            n_trials
        )
        
        return results
    
    def single_target_engagement(self, target: Target, 
                                current_time: float = 0.0) -> EngagementSolution:
        """
        Calculate complete engagement solution for single target.
        
        Args:
            target: Target to engage
            current_time: Current time reference
            
        Returns:
            Complete engagement solution
        """
        try:
            # Check if target is moving
            target_speed = np.linalg.norm(target.velocity)
            
            if target_speed < 0.1:  # Stationary target
                intercept_position = target.position
                intercept_time = current_time
                elevation, azimuth, flight_time = self.ballistic_solution(target.position)
                fire_time = current_time
            else:  # Moving target
                intercept_position, intercept_time = self.intercept_solution(target, current_time)
                elevation, azimuth = self.cannon.aim_at_target(intercept_position)
                flight_time = intercept_time - current_time
                fire_time = current_time
            
            # Check engagement feasibility
            can_engage, reason = self.cannon.can_engage_target(intercept_position)
            if not can_engage:
                return EngagementSolution(
                    target_id=target.id,
                    success=False,
                    reason=reason,
                    elevation=0, azimuth=0, fire_time=0, impact_time=0,
                    hit_probability=0, kill_probability=0, impact_energy=0,
                    ring_size_at_impact=0, muzzle_velocity=0, flight_time=0,
                    target_range=0, intercept_position=np.zeros(3)
                )
            
            # Calculate effectiveness
            effectiveness = self.engagement_effectiveness(target, intercept_position)
            
            # Check if effectiveness meets minimum requirements
            if effectiveness['kill_probability'] < self.min_kill_probability:
                return EngagementSolution(
                    target_id=target.id,
                    success=False,
                    reason=f"Kill probability {effectiveness['kill_probability']:.3f} below minimum {self.min_kill_probability}",
                    elevation=elevation, azimuth=azimuth, fire_time=fire_time, impact_time=intercept_time,
                    hit_probability=effectiveness['hit_probability'],
                    kill_probability=effectiveness['kill_probability'],
                    impact_energy=effectiveness['average_impact_energy'],
                    ring_size_at_impact=effectiveness['average_ring_size_at_impact'],
                    muzzle_velocity=self.cannon.calculate_muzzle_velocity(),
                    flight_time=flight_time,
                    target_range=effectiveness['target_range'],
                    intercept_position=intercept_position
                )
            
            # Successful engagement solution
            return EngagementSolution(
                target_id=target.id,
                success=True,
                reason="Engagement solution calculated",
                elevation=elevation,
                azimuth=azimuth,
                fire_time=fire_time,
                impact_time=intercept_time,
                hit_probability=effectiveness['hit_probability'],
                kill_probability=effectiveness['kill_probability'],
                impact_energy=effectiveness['average_impact_energy'],
                ring_size_at_impact=effectiveness['average_ring_size_at_impact'],
                muzzle_velocity=self.cannon.calculate_muzzle_velocity(),
                flight_time=flight_time,
                target_range=effectiveness['target_range'],
                intercept_position=intercept_position
            )
            
        except Exception as e:
            return EngagementSolution(
                target_id=target.id,
                success=False,
                reason=f"Calculation error: {str(e)}",
                elevation=0, azimuth=0, fire_time=0, impact_time=0,
                hit_probability=0, kill_probability=0, impact_energy=0,
                ring_size_at_impact=0, muzzle_velocity=0, flight_time=0,
                target_range=0, intercept_position=np.zeros(3)
            )
    
    def multi_target_engagement(self, targets: List[Target], 
                              current_time: float = 0.0) -> List[EngagementSolution]:
        """
        Calculate engagement sequence for multiple targets.
        
        Prioritizes targets by threat level and engagement feasibility.
        
        Args:
            targets: List of targets to engage
            current_time: Current time reference
            
        Returns:
            List of engagement solutions in optimal sequence
        """
        solutions = []
        available_targets = targets.copy()
        engagement_time = current_time
        
        # Sort targets by priority and engagement effectiveness
        def target_priority_score(target):
            # Quick effectiveness estimate
            range_to_target = np.linalg.norm(target.position - self.cannon.position)
            range_factor = max(0, 1 - range_to_target / self.max_engagement_range)
            priority_factor = 1.0 / target.priority  # Lower priority number = higher priority
            vulnerability_factor = target.vulnerability
            
            return range_factor * priority_factor * vulnerability_factor
        
        available_targets.sort(key=target_priority_score, reverse=True)
        
        # Engage targets in sequence
        while available_targets and engagement_time < current_time + 30.0:  # 30 second time limit
            # Get next target
            target = available_targets.pop(0)
            
            # Calculate engagement solution
            solution = self.single_target_engagement(target, engagement_time)
            solutions.append(solution)
            
            if solution.success:
                # Update engagement time (include reload time)
                engagement_time = solution.impact_time + self.cannon.reload_time
            else:
                # Try next target immediately if this one failed
                pass
        
        return solutions
    
    def engagement_envelope_analysis(self, drone_type: str = 'small') -> Dict:
        """
        Analyze engagement envelope for specified drone type.
        
        Args:
            drone_type: Type of drone ('small', 'medium', 'large')
            
        Returns:
            Dictionary with engagement envelope data
        """
        if drone_type not in self.drone_models:
            raise ValueError(f"Unknown drone type: {drone_type}")
        
        drone_spec = self.drone_models[drone_type]
        
        # Test engagement at various ranges and elevations
        ranges = np.arange(5, 81, 5)  # 5 to 80 meters
        elevations = np.arange(0, 91, 10)  # 0 to 90 degrees
        
        results = {
            'ranges': ranges.tolist(),
            'elevations': elevations.tolist(),
            'kill_probability_matrix': [],
            'hit_probability_matrix': [],
            'max_effective_range': 0,
            'optimal_elevation': 0
        }
        
        max_kill_prob = 0
        optimal_range = 0
        optimal_elev = 0
        
        for elevation in elevations:
            kill_prob_row = []
            hit_prob_row = []
            
            for range_val in ranges:
                # Calculate target position
                elev_rad = np.radians(elevation)
                target_pos = self.cannon.position + np.array([
                    range_val * np.cos(elev_rad),
                    0,
                    range_val * np.sin(elev_rad)
                ])
                
                # Create test target
                test_target = Target(
                    id=f"test_{range_val}_{elevation}",
                    position=target_pos,
                    velocity=np.zeros(3),
                    size=drone_spec['size'],
                    vulnerability=drone_spec['vulnerability'],
                    priority=1,
                    detected_time=0.0
                )
                
                # Calculate engagement
                solution = self.single_target_engagement(test_target)
                
                kill_prob_row.append(solution.kill_probability)
                hit_prob_row.append(solution.hit_probability)
                
                # Track maximum effective range
                if solution.kill_probability > self.min_kill_probability:
                    results['max_effective_range'] = max(results['max_effective_range'], range_val)
                
                # Track optimal conditions
                if solution.kill_probability > max_kill_prob:
                    max_kill_prob = solution.kill_probability
                    optimal_range = range_val
                    optimal_elev = elevation
            
            results['kill_probability_matrix'].append(kill_prob_row)
            results['hit_probability_matrix'].append(hit_prob_row)
        
        results['optimal_range'] = optimal_range
        results['optimal_elevation'] = optimal_elev
        results['max_kill_probability'] = max_kill_prob
        
        return results


def test_engagement():
    """Test function to validate engagement calculations"""
    print("Testing Engagement Calculator...")
    
    try:
        # Create test cannon (using same approach as cannon.py test)
        from cannon import CannonConfiguration
        
        config_obj = CannonConfiguration(
            barrel_length=2.0,
            barrel_diameter=0.5,
            max_chamber_pressure=100000,
            max_elevation=85.0,
            max_traverse=360.0,
            formation_number=4.0,
            air_density=1.225
        )
        
        cannon = VortexCannon.__new__(VortexCannon)
        cannon.config = config_obj
        cannon.position = np.array([0.0, 0.0, 2.0])  # 2m elevation
        cannon.orientation = {'elevation': 0.0, 'azimuth': 0.0}
        cannon.chamber_pressure = 80000.0
        cannon.ready_to_fire = True
        cannon.last_shot_time = 0.0
        cannon.reload_time = 0.5
        cannon.pressure_buildup_time = 2.0
        
        # Create engagement calculator
        calc = EngagementCalculator(cannon)
        
        # Test stationary target
        stationary_target = Target(
            id="drone_001",
            position=np.array([30.0, 10.0, 15.0]),
            velocity=np.zeros(3),
            size=0.5,
            vulnerability=0.8,
            priority=1,
            detected_time=0.0
        )
        
        print("Testing stationary target engagement...")
        solution = calc.single_target_engagement(stationary_target)
        print(f"Success: {solution.success}")
        print(f"Kill probability: {solution.kill_probability:.3f}")
        print(f"Elevation: {solution.elevation:.1f}°")
        print(f"Flight time: {solution.flight_time:.2f}s")
        
        # Test moving target
        moving_target = Target(
            id="drone_002",
            position=np.array([40.0, 0.0, 20.0]),
            velocity=np.array([-5.0, 2.0, 0.0]),  # Moving target
            size=0.3,
            vulnerability=0.9,
            priority=1,
            detected_time=0.0
        )
        
        print("\nTesting moving target engagement...")
        solution2 = calc.single_target_engagement(moving_target)
        print(f"Success: {solution2.success}")
        print(f"Kill probability: {solution2.kill_probability:.3f}")
        print(f"Intercept position: {solution2.intercept_position}")
        
        # Test multi-target engagement
        targets = [stationary_target, moving_target]
        print("\nTesting multi-target engagement...")
        solutions = calc.multi_target_engagement(targets)
        print(f"Engagement sequence for {len(solutions)} targets:")
        for i, sol in enumerate(solutions):
            print(f"  Target {i+1}: {sol.target_id}, Success: {sol.success}, P_kill: {sol.kill_probability:.3f}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_engagement()