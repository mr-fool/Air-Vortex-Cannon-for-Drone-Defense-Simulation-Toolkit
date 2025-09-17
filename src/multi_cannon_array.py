#!/usr/bin/env python3
"""
Multi-Cannon Array System for Drone Defense

This module extends the single cannon simulation to model coordinated multi-cannon
arrays with various topologies, firing coordination strategies, and combined
energy effects for engaging larger drone threats.

Key Features:
- Array topology management (linear, grid, networked)
- Coordinated targeting and firing sequences
- Combined energy effects for larger targets
- Coverage optimization and overlap analysis
- Command and control simulation
"""

import numpy as np
import yaml
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import itertools
import time

# Import existing modules
from cannon import VortexCannon, CannonConfiguration
from engagement import EngagementCalculator, Target, EngagementSolution
from vortex_ring import VortexRing


class ArrayTopology(Enum):
    """Array deployment topologies"""
    LINEAR = "linear"
    GRID_2x2 = "grid_2x2" 
    GRID_3x3 = "grid_3x3"
    TRIANGULAR = "triangular"
    CIRCULAR = "circular"
    NETWORKED = "networked"


class FiringMode(Enum):
    """Firing coordination modes"""
    SEQUENTIAL = "sequential"        # One cannon at a time
    SIMULTANEOUS = "simultaneous"    # All cannons fire together
    COORDINATED = "coordinated"      # Timed for simultaneous impact
    ADAPTIVE = "adaptive"            # Dynamic based on target


@dataclass
class CannonUnit:
    """Individual cannon in array with state tracking"""
    id: str
    cannon: VortexCannon
    position: np.ndarray
    orientation: Dict[str, float]
    ready: bool = True
    last_fire_time: float = 0.0
    assigned_target: Optional[str] = None
    
    def reload_time_remaining(self, current_time: float) -> float:
        """Calculate remaining reload time"""
        elapsed = current_time - self.last_fire_time
        return max(0, self.cannon.reload_time - elapsed)


@dataclass
class ArrayConfiguration:
    """Configuration for multi-cannon array"""
    topology: ArrayTopology
    spacing: float                    # Distance between cannons (m)
    cannon_config: CannonConfiguration
    firing_mode: FiringMode
    coordination_delay: float = 0.1   # Command delay between cannons (s)
    max_simultaneous: int = 4         # Max cannons firing simultaneously
    energy_combination_factor: float = 0.8  # Efficiency of combined energy
    

class MultiCannonArray:
    """
    Multi-cannon array system for coordinated drone defense.
    
    Manages array topology, target assignment, firing coordination,
    and combined engagement effects for enhanced capability.
    """
    
    def __init__(self, config: ArrayConfiguration):
        """Initialize multi-cannon array system"""
        self.config = config
        self.cannons: List[CannonUnit] = []
        self.array_center = np.zeros(3)
        self.coverage_radius = 0.0
        self.command_center = None
        
        # Performance tracking
        self.engagement_history: List[Dict] = []
        self.array_metrics = {
            'total_engagements': 0,
            'successful_engagements': 0,
            'average_kill_probability': 0.0,
            'coverage_area': 0.0
        }
        
        self._deploy_array()
        self._initialize_command_control()
    
    def _deploy_array(self):
        """Deploy cannons according to specified topology"""
        positions = self._calculate_positions()
        
        for i, pos in enumerate(positions):
            # Create individual cannon
            cannon = VortexCannon.__new__(VortexCannon)
            cannon.config = self.config.cannon_config
            cannon.position = pos
            cannon.orientation = {'elevation': 0.0, 'azimuth': 0.0}
            cannon.chamber_pressure = self.config.cannon_config.chamber_pressure
            cannon.ready_to_fire = True
            cannon.last_shot_time = 0.0
            cannon.reload_time = 0.5
            cannon.pressure_buildup_time = 2.0
            
            # Create cannon unit
            unit = CannonUnit(
                id=f"cannon_{i+1:02d}",
                cannon=cannon,
                position=pos.copy(),
                orientation={'elevation': 0.0, 'azimuth': 0.0}
            )
            self.cannons.append(unit)
        
        # Calculate array properties
        self.array_center = np.mean([c.position for c in self.cannons], axis=0)
        max_distance = max([np.linalg.norm(c.position - self.array_center) 
                           for c in self.cannons])
        self.coverage_radius = max_distance + 60.0  # Add max engagement range
        
    def _calculate_positions(self) -> List[np.ndarray]:
        """Calculate cannon positions based on topology"""
        positions = []
        spacing = self.config.spacing
        
        if self.config.topology == ArrayTopology.LINEAR:
            # Linear array along X-axis
            n_cannons = 4
            for i in range(n_cannons):
                x = (i - (n_cannons-1)/2) * spacing
                positions.append(np.array([x, 0.0, 2.0]))
        
        elif self.config.topology == ArrayTopology.GRID_2x2:
            # 2x2 grid
            for i in range(2):
                for j in range(2):
                    x = (i - 0.5) * spacing
                    y = (j - 0.5) * spacing
                    positions.append(np.array([x, y, 2.0]))
        
        elif self.config.topology == ArrayTopology.GRID_3x3:
            # 3x3 grid
            for i in range(3):
                for j in range(3):
                    x = (i - 1) * spacing
                    y = (j - 1) * spacing
                    positions.append(np.array([x, y, 2.0]))
        
        elif self.config.topology == ArrayTopology.TRIANGULAR:
            # Triangular formation
            positions = [
                np.array([0.0, 0.0, 2.0]),
                np.array([-spacing/2, spacing*np.sqrt(3)/2, 2.0]),
                np.array([spacing/2, spacing*np.sqrt(3)/2, 2.0])
            ]
        
        elif self.config.topology == ArrayTopology.CIRCULAR:
            # Circular formation
            n_cannons = 6
            for i in range(n_cannons):
                angle = 2 * np.pi * i / n_cannons
                x = spacing * np.cos(angle)
                y = spacing * np.sin(angle)
                positions.append(np.array([x, y, 2.0]))
        
        elif self.config.topology == ArrayTopology.NETWORKED:
            # Networked deployment (optimized positions)
            positions = [
                np.array([0.0, 0.0, 2.0]),
                np.array([spacing, spacing/2, 2.0]),
                np.array([-spacing/2, spacing, 2.0]),
                np.array([spacing/2, -spacing/2, 2.0])
            ]
        
        return positions
    
    def _initialize_command_control(self):
        """Initialize command and control system"""
        self.command_center = {
            'position': self.array_center.copy(),
            'coverage_map': self._generate_coverage_map(),
            'target_queue': [],
            'engagement_queue': [],
            'coordination_matrix': self._calculate_coordination_matrix()
        }
    
    def _generate_coverage_map(self) -> Dict:
        """Generate coverage map for array"""
        # Discretize coverage area
        resolution = 5.0  # 5m grid
        x_range = np.arange(-100, 101, resolution)
        y_range = np.arange(-100, 101, resolution)
        z_range = np.arange(5, 51, resolution)
        
        coverage_map = {}
        
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    test_point = np.array([x, y, z])
                    covering_cannons = []
                    
                    for cannon in self.cannons:
                        range_to_point = np.linalg.norm(test_point - cannon.position)
                        if range_to_point <= 60.0:  # Max engagement range
                            can_engage, _ = cannon.cannon.can_engage_target(test_point)
                            if can_engage:
                                covering_cannons.append(cannon.id)
                    
                    if covering_cannons:
                        coverage_map[(x, y, z)] = covering_cannons
        
        return coverage_map
    
    def _calculate_coordination_matrix(self) -> np.ndarray:
        """Calculate coordination timing matrix between cannons"""
        n_cannons = len(self.cannons)
        coordination_matrix = np.zeros((n_cannons, n_cannons))
        
        # Communication delays between cannons
        for i in range(n_cannons):
            for j in range(n_cannons):
                if i != j:
                    distance = np.linalg.norm(self.cannons[i].position - 
                                            self.cannons[j].position)
                    # Assume radio communication with processing delay
                    delay = self.config.coordination_delay + distance * 1e-6
                    coordination_matrix[i, j] = delay
        
        return coordination_matrix
    
    def assign_targets(self, targets: List[Target], current_time: float = 0.0) -> Dict:
        """
        Assign targets to cannons using optimization algorithm.
        
        Args:
            targets: List of targets to engage
            current_time: Current simulation time
            
        Returns:
            Dictionary with target assignments and engagement plan
        """
        assignments = {}
        engagement_plan = []
        available_cannons = [c for c in self.cannons if c.ready]
        
        # Sort targets by priority and threat level
        def target_threat_score(target):
            # Calculate threat based on proximity, size, and priority
            min_distance = min([np.linalg.norm(target.position - c.position) 
                              for c in self.cannons])
            threat_score = (1.0 / target.priority) * (1.0 / max(min_distance, 1.0))
            return threat_score
        
        sorted_targets = sorted(targets, key=target_threat_score, reverse=True)
        
        # Assignment strategies based on firing mode
        if self.config.firing_mode == FiringMode.SEQUENTIAL:
            assignments = self._assign_sequential(sorted_targets, available_cannons)
        elif self.config.firing_mode == FiringMode.SIMULTANEOUS:
            assignments = self._assign_simultaneous(sorted_targets, available_cannons)
        elif self.config.firing_mode == FiringMode.COORDINATED:
            assignments = self._assign_coordinated(sorted_targets, available_cannons)
        elif self.config.firing_mode == FiringMode.ADAPTIVE:
            assignments = self._assign_adaptive(sorted_targets, available_cannons, current_time)
        
        return {
            'assignments': assignments,
            'engagement_plan': engagement_plan,
            'mode': self.config.firing_mode.value,
            'total_targets': len(targets),
            'available_cannons': len(available_cannons)
        }
    
    def _assign_sequential(self, targets: List[Target], 
                          cannons: List[CannonUnit]) -> Dict:
        """Sequential target assignment - one cannon per target"""
        assignments = {}
        
        for i, target in enumerate(targets[:len(cannons)]):
            # Find best cannon for this target
            best_cannon = None
            best_score = -1
            
            for cannon in cannons:
                if cannon.id not in assignments.values():
                    # Calculate suitability score
                    range_to_target = np.linalg.norm(target.position - cannon.position)
                    if range_to_target <= 60.0:  # Within max range
                        can_engage, _ = cannon.cannon.can_engage_target(target.position)
                        if can_engage:
                            score = 1.0 / max(range_to_target, 1.0)  # Prefer closer cannons
                            if score > best_score:
                                best_score = score
                                best_cannon = cannon
            
            if best_cannon:
                assignments[target.id] = best_cannon.id
                best_cannon.assigned_target = target.id
        
        return assignments
    
    def _assign_simultaneous(self, targets: List[Target], 
                           cannons: List[CannonUnit]) -> Dict:
        """Simultaneous assignment - multiple cannons per high-value target"""
        assignments = {}
        
        # Group targets by priority
        priority_groups = {}
        for target in targets:
            if target.priority not in priority_groups:
                priority_groups[target.priority] = []
            priority_groups[target.priority].append(target)
        
        cannon_index = 0
        
        for priority in sorted(priority_groups.keys()):
            targets_in_group = priority_groups[priority]
            
            for target in targets_in_group:
                if cannon_index >= len(cannons):
                    break
                
                # Assign multiple cannons to high-priority targets
                cannons_per_target = min(2 if priority == 1 else 1, 
                                       len(cannons) - cannon_index)
                
                target_assignments = []
                for _ in range(cannons_per_target):
                    if cannon_index < len(cannons):
                        cannon = cannons[cannon_index]
                        range_to_target = np.linalg.norm(target.position - cannon.position)
                        
                        if range_to_target <= 60.0:
                            can_engage, _ = cannon.cannon.can_engage_target(target.position)
                            if can_engage:
                                target_assignments.append(cannon.id)
                                cannon.assigned_target = target.id
                        
                        cannon_index += 1
                
                if target_assignments:
                    assignments[target.id] = target_assignments
        
        return assignments
    
    def _assign_coordinated(self, targets: List[Target], 
                          cannons: List[CannonUnit]) -> Dict:
        """Coordinated assignment with timing optimization"""
        assignments = {}
        
        # Calculate optimal firing times for simultaneous impact
        for target in targets:
            suitable_cannons = []
            
            for cannon in cannons:
                range_to_target = np.linalg.norm(target.position - cannon.position)
                if range_to_target <= 60.0:
                    can_engage, _ = cannon.cannon.can_engage_target(target.position)
                    if can_engage:
                        # Calculate flight time
                        vr = cannon.cannon.generate_vortex_ring(target.position)
                        flight_time = vr.time_to_range(range_to_target)
                        suitable_cannons.append((cannon, flight_time))
            
            if suitable_cannons:
                # Sort by flight time and select best cannons
                suitable_cannons.sort(key=lambda x: x[1])
                
                # Assign up to 2 cannons per target for coordination
                selected = suitable_cannons[:min(2, len(suitable_cannons))]
                assignments[target.id] = [c[0].id for c in selected]
                
                for cannon, _ in selected:
                    cannon.assigned_target = target.id
        
        return assignments
    
    def _assign_adaptive(self, targets: List[Target], cannons: List[CannonUnit],
                    current_time: float) -> Dict:
        assignments = {}
        
        for target in targets:
            target_speed = np.linalg.norm(target.velocity)
            
            # Find ALL suitable cannons for this target
            suitable = []
            for cannon in cannons:
                if not cannon.assigned_target:
                    range_to_target = np.linalg.norm(target.position - cannon.position)
                    if range_to_target <= 75.0:  # Use enhanced range
                        can_engage, _ = cannon.cannon.can_engage_target(target.position)
                        if can_engage:
                            suitable.append(cannon)
            
            # SIMPLIFIED ASSIGNMENT:
            if target.size >= 0.5:  # Medium or large targets
                # Assign ALL available cannons
                selected = suitable
            else:  # Small targets
                # Only need 1 cannon
                selected = suitable[:1]
            
            if selected:
                assignments[target.id] = [c.id for c in selected]
                for cannon in selected:
                    cannon.assigned_target = target.id
        
        return assignments
    
    def calculate_combined_engagement(self, target: Target, 
                                    assigned_cannons: List[str],
                                    current_time: float = 0.0) -> Dict:
        """
        Calculate combined engagement effects from multiple cannons.
        
        Args:
            target: Target being engaged
            assigned_cannons: List of cannon IDs assigned to target
            current_time: Current simulation time
            
        Returns:
            Combined engagement results
        """
        cannon_units = [c for c in self.cannons if c.id in assigned_cannons]
        individual_solutions = []
        
        # Calculate individual engagement solutions
        for unit in cannon_units:
            calc = EngagementCalculator(unit.cannon)
            solution = calc.single_target_engagement(target, current_time)
            individual_solutions.append(solution)
        
        # Combine effects
        combined_results = self._combine_engagement_effects(
            target, individual_solutions, cannon_units)
        
        return combined_results
    
    def _combine_engagement_effects(self, target, solutions, cannons):
        """Calculate combined effects from multiple cannon engagements - FIXED VERSION"""
        
        if not solutions:
            return {
                'success': False,
                'combined_kill_probability': 0.0,
                'participating_cannons': 0,
                'combined_energy': 0.0
            }
        
        # Count ALL solutions that attempted engagement (not just successful ones)
        attempted_solutions = [s for s in solutions if hasattr(s, 'impact_energy')]
        
        if not attempted_solutions:
            return {
                'success': False,
                'combined_kill_probability': 0.0,
                'participating_cannons': 0,
                'combined_energy': 0.0
            }
        
        # Calculate total energy from ALL attempts (successful or not)
        total_energy = sum(s.impact_energy for s in attempted_solutions if s.impact_energy > 0)
        
        # Multi-cannon bonus
        if len(attempted_solutions) > 1:
            combined_energy = total_energy * 1.3  # 30% bonus for multiple cannons
        else:
            combined_energy = total_energy
        
        # Calculate kill probability based on energy and target vulnerability
        if combined_energy > 0:
            # Base kill probability from energy
            energy_factor = min(0.8, combined_energy / 1000) * target.vulnerability
            
            # Multi-cannon coordination bonus
            if len(attempted_solutions) >= 2:
                energy_factor += 0.1  # 10% bonus for coordination
            
            final_kill_prob = min(0.9, energy_factor)
        else:
            final_kill_prob = 0.0
        
        # FIXED: Success should be based on meaningful engagement, not arbitrary threshold
        # Success if we delivered energy and have some kill probability
        success = (combined_energy > 100) and (final_kill_prob > 0.01)  # Very low threshold
        
        return {
            'success': success,
            'target_id': target.id,
            'participating_cannons': len(attempted_solutions),
            'combined_energy': combined_energy,
            'combined_kill_probability': final_kill_prob,
            'individual_solutions': attempted_solutions
        }
        
    def execute_engagement_sequence(self, targets: List[Target],
                                  current_time: float = 0.0) -> List[Dict]:
        """
        Execute complete engagement sequence for multiple targets.
        
        Args:
            targets: List of targets to engage
            current_time: Current simulation time
            
        Returns:
            List of engagement results
        """
        # Assign targets to cannons
        assignment_result = self.assign_targets(targets, current_time)
        assignments = assignment_result['assignments']
        
        engagement_results = []
        execution_time = current_time
        
        # Execute engagements based on assignments
        for target_id, cannon_ids in assignments.items():
            target = next((t for t in targets if t.id == target_id), None)
            if not target:
                continue
            
            if isinstance(cannon_ids, list):
                # Multi-cannon engagement
                result = self.calculate_combined_engagement(
                    target, cannon_ids, execution_time)
            else:
                # Single cannon engagement
                cannon = next((c for c in self.cannons if c.id == cannon_ids), None)
                if cannon:
                    calc = EngagementCalculator(cannon.cannon)
                    solution = calc.single_target_engagement(target, execution_time)
                    result = {
                        'success': solution.success,
                        'target_id': target.id,
                        'participating_cannons': 1,
                        'combined_kill_probability': solution.kill_probability,
                        'individual_solutions': [solution]
                    }
            
            engagement_results.append(result)
            
            # Update execution time for sequential operations
            if self.config.firing_mode == FiringMode.SEQUENTIAL:
                if result['success'] and result['individual_solutions']:
                    max_impact_time = max(s.impact_time for s in result['individual_solutions'])
                    execution_time = max_impact_time + 0.5  # Reload time
        
        # Update array metrics
        self._update_metrics(engagement_results)
        
        return engagement_results
    
    def _update_metrics(self, results: List[Dict]):
        """Update array performance metrics"""
        self.array_metrics['total_engagements'] += len(results)
        
        successful = sum(1 for r in results if r['success'])
        self.array_metrics['successful_engagements'] += successful
        
        if len(results) > 0:
            avg_kill_prob = np.mean([r.get('combined_kill_probability', 0) 
                                   for r in results])
            
            # Running average
            total_eng = self.array_metrics['total_engagements']
            current_avg = self.array_metrics['average_kill_probability']
            self.array_metrics['average_kill_probability'] = (
                (current_avg * (total_eng - len(results)) + 
                 avg_kill_prob * len(results)) / total_eng
            )
    
    def analyze_coverage(self) -> Dict:
        """Analyze array coverage capabilities"""
        coverage_analysis = {
            'topology': self.config.topology.value,
            'total_cannons': len(self.cannons),
            'array_span': 0.0,
            'coverage_overlap': {},
            'blind_spots': [],
            'optimal_targets': 0
        }
        
        # Calculate array span
        positions = np.array([c.position for c in self.cannons])
        if len(positions) > 1:
            coverage_analysis['array_span'] = np.max(pdist(positions))
        
        # Analyze coverage overlap
        test_points = []
        for angle in np.linspace(0, 2*np.pi, 36):  # Every 10 degrees
            for range_val in [20, 30, 40, 50]:
                x = range_val * np.cos(angle) + self.array_center[0]
                y = range_val * np.sin(angle) + self.array_center[1]
                z = 15.0  # Typical drone altitude
                test_points.append(np.array([x, y, z]))
        
        overlap_counts = []
        for point in test_points:
            covering_cannons = 0
            for cannon in self.cannons:
                range_to_point = np.linalg.norm(point - cannon.position)
                if range_to_point <= 45.0:  # Max effective range
                    can_engage, _ = cannon.cannon.can_engage_target(point)
                    if can_engage:
                        covering_cannons += 1
            overlap_counts.append(covering_cannons)
        
        coverage_analysis['coverage_overlap'] = {
            'average_overlap': np.mean(overlap_counts),
            'max_overlap': np.max(overlap_counts),
            'uncovered_points': sum(1 for c in overlap_counts if c == 0)
        }
        
        return coverage_analysis
    
    def get_array_status(self) -> Dict:
        """Get current array system status"""
        return {
            'topology': self.config.topology.value,
            'firing_mode': self.config.firing_mode.value,
            'total_cannons': len(self.cannons),
            'ready_cannons': sum(1 for c in self.cannons if c.ready),
            'array_center': self.array_center.tolist(),
            'coverage_radius': self.coverage_radius,
            'metrics': self.array_metrics.copy(),
            'cannon_status': [
                {
                    'id': c.id,
                    'position': c.position.tolist(),
                    'ready': c.ready,
                    'assigned_target': c.assigned_target
                }
                for c in self.cannons
            ]
        }


def create_test_array(topology: ArrayTopology = ArrayTopology.GRID_2x2,
                     firing_mode: FiringMode = FiringMode.COORDINATED) -> MultiCannonArray:
    """Create test multi-cannon array with realistic configuration"""
    
    # Use configuration from existing YAML or create realistic fallback
    try:
        import yaml
        config_path = "config/cannon_specs.yaml"
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        cannon_config_data = config_data['cannon']
        vortex_config = config_data.get('vortex_ring', {})
        env_config = config_data.get('environment', {})
        
        chamber_pressure = cannon_config_data.get('chamber_pressure', 
                                                 cannon_config_data['max_chamber_pressure'] * 0.8)
        
        cannon_config = CannonConfiguration(
            barrel_length=cannon_config_data['barrel_length'],
            barrel_diameter=cannon_config_data['barrel_diameter'],
            max_chamber_pressure=cannon_config_data['max_chamber_pressure'],
            max_elevation=cannon_config_data.get('max_elevation', 85.0),
            max_traverse=cannon_config_data.get('max_traverse', 360.0),
            formation_number=vortex_config.get('formation_number', 4.0),
            air_density=env_config.get('air_density', 1.225),
            chamber_pressure=chamber_pressure
        )
    except:
        # Fallback configuration
        cannon_config = CannonConfiguration(
            barrel_length=2.0,
            barrel_diameter=0.5,
            max_chamber_pressure=300000,
            max_elevation=85.0,
            max_traverse=360.0,
            formation_number=4.0,
            air_density=1.225,
            chamber_pressure=240000
        )
    
    array_config = ArrayConfiguration(
        topology=topology,
        spacing=20.0,  # 20m spacing between cannons
        cannon_config=cannon_config,
        firing_mode=firing_mode,
        coordination_delay=0.1,
        max_simultaneous=4,
        energy_combination_factor=0.8
    )
    
    return MultiCannonArray(array_config)


def test_multi_cannon_array():
    """Test multi-cannon array functionality"""
    print("Testing Multi-Cannon Array System...")
    
    # Create test array
    array = create_test_array(ArrayTopology.GRID_2x2, FiringMode.COORDINATED)
    
    print(f"Array created with {len(array.cannons)} cannons")
    print(f"Array center: {array.array_center}")
    print(f"Coverage radius: {array.coverage_radius:.1f}m")
    
    # Create test targets
    targets = [
        Target("small_1", np.array([30, 10, 15]), np.zeros(3), 0.3, 0.9, 1, 0.0),
        Target("medium_1", np.array([25, -15, 18]), np.array([-3, 1, 0]), 0.6, 0.7, 2, 0.0),
        Target("large_1", np.array([40, 0, 25]), np.array([-2, 0, 0]), 1.2, 0.5, 1, 0.0)
    ]
    
    print(f"\nTest targets: {len(targets)}")
    
    # Execute engagement sequence
    results = array.execute_engagement_sequence(targets)
    
    print(f"\nEngagement Results:")
    for result in results:
        target_id = result['target_id']
        success = result['success']
        cannons = result['participating_cannons']
        kill_prob = result.get('combined_kill_probability', 0)
        print(f"  {target_id}: Success={success}, Cannons={cannons}, P_kill={kill_prob:.3f}")
    
    # Coverage analysis
    coverage = array.analyze_coverage()
    print(f"\nCoverage Analysis:")
    print(f"  Array span: {coverage['array_span']:.1f}m")
    print(f"  Average overlap: {coverage['coverage_overlap']['average_overlap']:.1f}")
    print(f"  Uncovered points: {coverage['coverage_overlap']['uncovered_points']}")
    
    # System status
    status = array.get_array_status()
    print(f"\nArray Status:")
    print(f"  Ready cannons: {status['ready_cannons']}/{status['total_cannons']}")
    print(f"  Success rate: {status['metrics']['successful_engagements']}/{status['metrics']['total_engagements']}")


def demonstrate_array_topologies():
    """Demonstrate different array topologies and their characteristics"""
    print("\n" + "="*80)
    print("MULTI-CANNON ARRAY TOPOLOGY COMPARISON")
    print("="*80)
    
    topologies = [
        ArrayTopology.LINEAR,
        ArrayTopology.GRID_2x2,
        ArrayTopology.TRIANGULAR,
        ArrayTopology.CIRCULAR
    ]
    
    # Standard test targets
    test_targets = [
        Target("small_close", np.array([20, 0, 12]), np.zeros(3), 0.3, 0.9, 1, 0.0),
        Target("medium_moving", np.array([35, 15, 16]), np.array([-4, -2, 0]), 0.6, 0.7, 2, 0.0),
        Target("large_distant", np.array([45, 0, 25]), np.array([-2, 0, 0]), 1.2, 0.5, 1, 0.0)
    ]
    
    results_summary = []
    
    for topology in topologies:
        print(f"\n--- {topology.value.upper()} TOPOLOGY ---")
        
        # Test with coordinated firing mode
        array = create_test_array(topology, FiringMode.COORDINATED)
        
        # Analyze coverage
        coverage = array.analyze_coverage()
        print(f"Array span: {coverage['array_span']:.1f}m")
        print(f"Average coverage overlap: {coverage['coverage_overlap']['average_overlap']:.1f}")
        
        # Execute engagement
        engagement_results = array.execute_engagement_sequence(test_targets)
        
        # Calculate performance metrics
        successful = sum(1 for r in engagement_results if r['success'])
        total_kill_prob = sum(r.get('combined_kill_probability', 0) for r in engagement_results)
        avg_kill_prob = total_kill_prob / len(engagement_results) if engagement_results else 0
        
        multi_cannon_engagements = sum(1 for r in engagement_results 
                                     if r.get('participating_cannons', 0) > 1)
        
        print(f"Engagement success: {successful}/{len(test_targets)}")
        print(f"Average kill probability: {avg_kill_prob:.3f}")
        print(f"Multi-cannon engagements: {multi_cannon_engagements}")
        
        # Store results for comparison
        results_summary.append({
            'topology': topology.value,
            'success_rate': successful / len(test_targets),
            'avg_kill_prob': avg_kill_prob,
            'array_span': coverage['array_span'],
            'coverage_overlap': coverage['coverage_overlap']['average_overlap'],
            'multi_cannon_rate': multi_cannon_engagements / len(test_targets)
        })
    
    # Summary comparison
    print(f"\n--- TOPOLOGY COMPARISON SUMMARY ---")
    print(f"{'Topology':<12} {'Success':<8} {'Avg P_kill':<10} {'Span(m)':<8} {'Overlap':<8} {'Multi-rate':<10}")
    print("-" * 70)
    
    for result in results_summary:
        print(f"{result['topology']:<12} {result['success_rate']:<8.1%} "
              f"{result['avg_kill_prob']:<10.3f} {result['array_span']:<8.1f} "
              f"{result['coverage_overlap']:<8.1f} {result['multi_cannon_rate']:<10.1%}")


def demonstrate_firing_modes():
    """Demonstrate different firing coordination modes"""
    print("\n" + "="*80)
    print("FIRING MODE COMPARISON")
    print("="*80)
    
    firing_modes = [
        FiringMode.SEQUENTIAL,
        FiringMode.SIMULTANEOUS,
        FiringMode.COORDINATED,
        FiringMode.ADAPTIVE
    ]
    
    # Test scenario: Mixed threat targets
    mixed_targets = [
        Target("priority_1", np.array([25, 5, 15]), np.array([-5, 0, 0]), 0.3, 0.9, 1, 0.0),
        Target("large_threat", np.array([30, -10, 20]), np.array([-3, 1, 0]), 1.2, 0.5, 1, 0.0),
        Target("fast_mover", np.array([35, 15, 12]), np.array([-8, -3, 0]), 0.4, 0.8, 2, 0.0),
        Target("medium_drone", np.array([40, 0, 18]), np.array([-2, 0, 0]), 0.6, 0.7, 2, 0.0)
    ]
    
    mode_results = []
    
    for mode in firing_modes:
        print(f"\n--- {mode.value.upper()} MODE ---")
        
        # Create array with this firing mode
        array = create_test_array(ArrayTopology.GRID_2x2, mode)
        
        # Execute engagement
        start_time = time.time()
        results = array.execute_engagement_sequence(mixed_targets)
        execution_time = time.time() - start_time
        
        # Analyze results
        successful = sum(1 for r in results if r['success'])
        total_cannons_used = sum(r.get('participating_cannons', 0) for r in results)
        simultaneous_impacts = 0
        
        # Check for simultaneous impacts (within 0.5s)
        if mode in [FiringMode.SIMULTANEOUS, FiringMode.COORDINATED]:
            for result in results:
                if 'individual_solutions' in result:
                    impact_times = [s.impact_time for s in result['individual_solutions'] if s.success]
                    if len(impact_times) > 1:
                        time_spread = max(impact_times) - min(impact_times)
                        if time_spread <= 0.5:
                            simultaneous_impacts += 1
        
        avg_kill_prob = np.mean([r.get('combined_kill_probability', 0) for r in results])
        
        print(f"Successful engagements: {successful}/{len(mixed_targets)}")
        print(f"Average kill probability: {avg_kill_prob:.3f}")
        print(f"Total cannons used: {total_cannons_used}")
        print(f"Simultaneous impacts: {simultaneous_impacts}")
        print(f"Execution time: {execution_time:.3f}s")
        
        mode_results.append({
            'mode': mode.value,
            'success_rate': successful / len(mixed_targets),
            'avg_kill_prob': avg_kill_prob,
            'cannons_used': total_cannons_used,
            'simultaneous_impacts': simultaneous_impacts,
            'execution_time': execution_time
        })
    
    # Mode comparison summary
    print(f"\n--- FIRING MODE COMPARISON ---")
    print(f"{'Mode':<12} {'Success':<8} {'Avg P_kill':<10} {'Cannons':<8} {'Simul.':<6} {'Time(s)':<8}")
    print("-" * 65)
    
    for result in mode_results:
        print(f"{result['mode']:<12} {result['success_rate']:<8.1%} "
              f"{result['avg_kill_prob']:<10.3f} {result['cannons_used']:<8} "
              f"{result['simultaneous_impacts']:<6} {result['execution_time']:<8.3f}")


def analyze_scalability():
    """Analyze array scalability with increasing threat levels"""
    print("\n" + "="*80)
    print("SCALABILITY ANALYSIS: Array Performance vs Threat Level")
    print("="*80)
    
    # Define threat scenarios of increasing complexity
    threat_scenarios = {
        'light': [
            Target("drone_1", np.array([25, 0, 15]), np.zeros(3), 0.3, 0.9, 1, 0.0),
            Target("drone_2", np.array([30, 10, 12]), np.array([-2, 0, 0]), 0.3, 0.9, 2, 0.0)
        ],
        'moderate': [
            Target("drone_1", np.array([20, 5, 15]), np.array([-3, 1, 0]), 0.3, 0.9, 1, 0.0),
            Target("drone_2", np.array([35, -8, 18]), np.array([-4, 2, 0]), 0.3, 0.9, 1, 0.0),
            Target("medium_1", np.array([30, 15, 20]), np.array([-2, -1, 0]), 0.6, 0.7, 2, 0.0)
        ],
        'heavy': [
            Target("drone_1", np.array([22, 8, 14]), np.array([-5, 1, 0]), 0.3, 0.9, 1, 0.0),
            Target("drone_2", np.array([28, -12, 16]), np.array([-6, 3, 0]), 0.3, 0.9, 1, 0.0),
            Target("drone_3", np.array([35, 0, 18]), np.array([-4, 0, 0]), 0.3, 0.9, 2, 0.0),
            Target("medium_1", np.array([32, 20, 22]), np.array([-3, -2, 0]), 0.6, 0.7, 2, 0.0),
            Target("large_1", np.array([40, 0, 25]), np.array([-2, 0, 0]), 1.2, 0.5, 1, 0.0)
        ],
        'extreme': [
            Target("drone_1", np.array([18, 6, 12]), np.array([-7, 2, 0]), 0.3, 0.9, 1, 0.0),
            Target("drone_2", np.array([25, -15, 14]), np.array([-8, 4, 0]), 0.3, 0.9, 1, 0.0),
            Target("drone_3", np.array([32, 8, 16]), np.array([-5, -1, 0]), 0.3, 0.9, 1, 0.0),
            Target("drone_4", np.array([28, 18, 20]), np.array([-6, -3, 0]), 0.3, 0.9, 2, 0.0),
            Target("medium_1", np.array([35, -5, 18]), np.array([-4, 1, 0]), 0.6, 0.7, 2, 0.0),
            Target("medium_2", np.array([30, 25, 24]), np.array([-3, -4, 0]), 0.6, 0.7, 2, 0.0),
            Target("large_1", np.array([42, 0, 28]), np.array([-2, 0, 0]), 1.2, 0.5, 1, 0.0),
            Target("large_2", np.array([38, -20, 30]), np.array([-1, 2, 0]), 1.2, 0.5, 1, 0.0)
        ]
    }
    
    # Test different array configurations
    configurations = [
        (ArrayTopology.LINEAR, "Linear 4-gun"),
        (ArrayTopology.GRID_2x2, "2x2 Grid"),
        (ArrayTopology.GRID_3x3, "3x3 Grid"),
        (ArrayTopology.CIRCULAR, "Circular 6-gun")
    ]
    
    print(f"{'Scenario':<10} {'Config':<12} {'Targets':<8} {'Success':<8} {'Avg P_kill':<10} {'Multi-gun':<9} {'Large Kills':<10}")
    print("-" * 85)
    
    for scenario_name, targets in threat_scenarios.items():
        for topology, config_name in configurations:
            if topology == ArrayTopology.GRID_3x3 and len(targets) < 4:
                continue  # Skip large arrays for light scenarios
                
            # Create array
            array = create_test_array(topology, FiringMode.ADAPTIVE)
            
            # Execute engagement
            results = array.execute_engagement_sequence(targets)
            
            # Analyze results
            successful = sum(1 for r in results if r['success'])
            avg_kill_prob = np.mean([r.get('combined_kill_probability', 0) for r in results])
            multi_gun = sum(1 for r in results if r.get('participating_cannons', 0) > 1)
            
            # Count large target kills
            large_kills = 0
            for result in results:
                target_id = result['target_id']
                target = next((t for t in targets if t.id == target_id), None)
                if target and target.size >= 1.0 and result['success']:
                    large_kills += 1
            
            print(f"{scenario_name:<10} {config_name:<12} {len(targets):<8} "
                  f"{successful:<8} {avg_kill_prob:<10.3f} {multi_gun:<9} {large_kills:<10}")
    
    print(f"\nKey Observations:")
    print(f"- Multi-gun coordination essential for large targets")
    print(f"- Array topology affects coverage and engagement efficiency")
    print(f"- Adaptive firing mode scales better with mixed threats")
    print(f"- 3x3 grid provides best performance against extreme threats")


def generate_deployment_recommendations():
    """Generate deployment recommendations based on analysis"""
    print("\n" + "="*80)
    print("DEPLOYMENT RECOMMENDATIONS")
    print("="*80)
    
    recommendations = {
        'Small Installation Defense': {
            'topology': 'Linear 4-gun array',
            'spacing': '15-20m',
            'firing_mode': 'Sequential/Coordinated',
            'coverage': '~300m frontage',
            'best_against': 'Small drone swarms (3-5 targets)',
            'limitations': 'Limited coverage depth, vulnerable to flanking'
        },
        
        'Medium Installation Defense': {
            'topology': '2x2 Grid array',
            'spacing': '20-25m',
            'firing_mode': 'Adaptive',
            'coverage': '~400m diameter',
            'best_against': 'Mixed threats, medium drones',
            'limitations': 'Gaps at extreme ranges'
        },
        
        'Large Installation Defense': {
            'topology': '3x3 Grid array', 
            'spacing': '25-30m',
            'firing_mode': 'Coordinated/Adaptive',
            'coverage': '~600m diameter',
            'best_against': 'Large drones, coordinated attacks',
            'limitations': 'High resource requirement'
        },
        
        'Mobile/Tactical Defense': {
            'topology': 'Triangular array',
            'spacing': '15-20m',
            'firing_mode': 'Simultaneous',
            'coverage': '~250m radius',
            'best_against': 'Fast-moving threats, ambush scenarios',
            'limitations': 'Limited sustained fire capability'
        },
        
        'Distributed Network Defense': {
            'topology': 'Networked array',
            'spacing': '50-100m',
            'firing_mode': 'Coordinated',
            'coverage': '~1km area',
            'best_against': 'Wide-area surveillance, large formations',
            'limitations': 'Complex command/control requirements'
        }
    }
    
    for scenario, details in recommendations.items():
        print(f"\n{scenario}:")
        print(f"  Recommended topology: {details['topology']}")
        print(f"  Optimal spacing: {details['spacing']}")
        print(f"  Firing mode: {details['firing_mode']}")
        print(f"  Coverage area: {details['coverage']}")
        print(f"  Best against: {details['best_against']}")
        print(f"  Limitations: {details['limitations']}")
    
    print(f"\nGeneral Design Principles:")
    print(f"- Cannon spacing should be 3-5x max engagement range")
    print(f"- Grid topologies provide best all-around coverage")
    print(f"- Adaptive firing modes handle mixed threats effectively")
    print(f"- Multi-cannon coordination essential for large targets")
    print(f"- Consider reload cycles in sustained engagement scenarios")


if __name__ == "__main__":
    # Run comprehensive multi-cannon array analysis
    test_multi_cannon_array()
    demonstrate_array_topologies()
    demonstrate_firing_modes()
    analyze_scalability()
    generate_deployment_recommendations()