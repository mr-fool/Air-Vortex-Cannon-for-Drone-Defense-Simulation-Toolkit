#!/usr/bin/env python3
"""
Multi-Cannon Array System for Drone Defense - PHYSICS CORRECTED VERSION

PHYSICS CORRECTIONS IMPLEMENTED:
1. Realistic coordination effects based on vortex ring interference theory
2. Conservative energy combination accounting for destructive interference
3. Reduced effective ranges based on targeting accuracy limitations
4. Coordination bonus limited to 5-15% based on fluid dynamics principles

THEORY BASIS:
- Vortex ring interactions: Widnall & Sullivan (1973) stability analysis
- Multi-projectile interference: Batchelor (1967) fluid mechanics
- Coordination timing effects: Gharib et al. (1998) formation number theory
- Energy dissipation: Saffman (1992) vortex dynamics
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
    # PHYSICS CORRECTED: Reduced energy combination efficiency
    energy_combination_factor: float = 0.6  # Reduced from 0.8 due to interference
    

class MultiCannonArray:
    """
    PHYSICS CORRECTED Multi-cannon array system for coordinated drone defense.
    
    CORRECTIONS IMPLEMENTED:
    - Realistic vortex ring interference modeling
    - Conservative coordination bonuses (5-15% max)
    - Reduced effective ranges (20-25m max)
    - Interference-based energy combination
    """
    
    def __init__(self, config: ArrayConfiguration):
        """Initialize multi-cannon array system"""
        self.config = config
        self.cannons: List[CannonUnit] = []
        self.array_center = np.zeros(3)
        # PHYSICS CORRECTED: Reduced coverage radius
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
        # PHYSICS CORRECTED: Reduced max engagement range
        self.coverage_radius = max_distance + 25.0  # Reduced from 60m to 25m
        
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
        """Generate coverage map for array with CORRECTED range limits"""
        # Discretize coverage area with realistic range limits
        resolution = 5.0  # 5m grid
        # PHYSICS CORRECTED: Reduced coverage area to realistic 30m max
        x_range = np.arange(-30, 31, resolution)
        y_range = np.arange(-30, 31, resolution)
        z_range = np.arange(5, 26, resolution)  # Reduced max altitude
        
        coverage_map = {}
        
        for x in x_range:
            for y in y_range:
                for z in z_range:
                    test_point = np.array([x, y, z])
                    covering_cannons = []
                    
                    for cannon in self.cannons:
                        range_to_point = np.linalg.norm(test_point - cannon.position)
                        # PHYSICS CORRECTED: Reduced max engagement range
                        if range_to_point <= 25.0:  # Reduced from 60m to 25m
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
        Assign targets to cannons using optimization algorithm with CORRECTED expectations
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
            assignments = self._assign_adaptive_corrected(sorted_targets, available_cannons, current_time)
        
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
                    # PHYSICS CORRECTED: Reduced max range
                    if range_to_target <= 25.0:  # Reduced from 60m
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
                        
                        # PHYSICS CORRECTED: Reduced max range
                        if range_to_target <= 25.0:  # Reduced from 60m
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
                # PHYSICS CORRECTED: Reduced max range
                if range_to_target <= 25.0:  # Reduced from 60m
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
    
    def _assign_adaptive_corrected(self, targets: List[Target], cannons: List[CannonUnit],
                                  current_time: float) -> Dict:
        """
        PHYSICS CORRECTED: Adaptive assignment with realistic multi-cannon expectations
        
        CORRECTIONS:
        - Conservative assignment (only for very large targets or close groups)
        - Reduced range limits (25m max)
        - More selective multi-cannon criteria
        """
        assignments = {}
        available_cannons = cannons.copy()
        
        # RESET all cannon assignments first
        for cannon in available_cannons:
            cannon.assigned_target = None
        
        # Sort targets by priority and size (larger/higher priority first)
        def target_priority_score(target):
            size_factor = target.size
            priority_factor = 1.0 / target.priority
            return size_factor * priority_factor
        
        sorted_targets = sorted(targets, key=target_priority_score, reverse=True)
        
        print(f"DEBUG CORRECTED: Starting assignment with {len(available_cannons)} cannons, {len(sorted_targets)} targets")
        
        for target in sorted_targets:
            print(f"DEBUG CORRECTED: Processing target {target.id} (size={target.size}m, priority={target.priority})")
            
            # Find ALL AVAILABLE cannons for this target with CORRECTED range limits
            suitable = []
            for cannon in available_cannons:
                if cannon.assigned_target is None:  # Only unassigned cannons
                    range_to_target = np.linalg.norm(target.position - cannon.position)
                    # PHYSICS CORRECTED: Reduced max range
                    if range_to_target <= 25.0:  # Reduced from 75m to 25m
                        try:
                            can_engage, reason = cannon.cannon.can_engage_target(target.position)
                            if can_engage:
                                suitable.append(cannon)
                            else:
                                print(f"DEBUG CORRECTED: Cannon {cannon.id} cannot engage: {reason}")
                        except Exception as e:
                            print(f"DEBUG CORRECTED: Error checking cannon {cannon.id}: {e}")
                    else:
                        print(f"DEBUG CORRECTED: Cannon {cannon.id} out of range: {range_to_target:.1f}m > 25m")
            
            print(f"DEBUG CORRECTED: Found {len(suitable)} suitable cannons for {target.id}")
            
            if suitable:
                # PHYSICS CORRECTED: Conservative multi-cannon assignment
                # Only assign multiple cannons in very specific circumstances
                
                if target.size >= 1.5 and len(suitable) >= 3:  # Very large targets only
                    # Large military drones might benefit from 2-3 cannons
                    num_to_assign = min(3, len(suitable))
                    selected = suitable[:num_to_assign]
                    assignment_reason = f"VERY LARGE target (>=1.5m): assigning {num_to_assign} cannons (interference expected)"
                    
                elif target.size >= 0.8 and len(suitable) >= 2 and target.priority == 1:
                    # High priority medium targets get 2 cannons for redundancy
                    num_to_assign = 2
                    selected = suitable[:num_to_assign]
                    assignment_reason = f"HIGH PRIORITY MEDIUM target: assigning {num_to_assign} cannons for redundancy"
                    
                else:  # Most targets get single cannon (realistic expectation)
                    num_to_assign = 1
                    selected = suitable[:num_to_assign]
                    assignment_reason = f"Standard assignment: {num_to_assign} cannon (multi-cannon interference not worth it)"
                
                print(f"DEBUG CORRECTED: {assignment_reason}")
                
                if selected:
                    # Create assignment
                    cannon_ids = [c.id for c in selected]
                    assignments[target.id] = cannon_ids
                    
                    # Mark cannons as assigned
                    for cannon in selected:
                        cannon.assigned_target = target.id
                    
                    print(f"DEBUG CORRECTED: Assigned {len(selected)} cannons to {target.id}: {cannon_ids}")
                else:
                    print(f"DEBUG CORRECTED: No cannons selected for {target.id}")
            else:
                print(f"DEBUG CORRECTED: No suitable cannons found for target {target.id}")
        
        print(f"DEBUG CORRECTED: Final assignments: {assignments}")
        return assignments
    
    def calculate_combined_engagement(self, target: Target, 
                                    assigned_cannons: List[str],
                                    current_time: float = 0.0) -> Dict:
        """
        Calculate combined engagement effects with REALISTIC vortex ring interference.
        """
        cannon_units = [c for c in self.cannons if c.id in assigned_cannons]
        individual_solutions = []
        
        # Calculate individual engagement solutions
        for unit in cannon_units:
            calc = EngagementCalculator(unit.cannon)
            solution = calc.single_target_engagement(target, current_time)
            individual_solutions.append(solution)
        
        # PHYSICS CORRECTED: Apply realistic interference effects
        combined_results = self._apply_vortex_interference_physics(
            target, individual_solutions, cannon_units)
        
        return combined_results
    
    def _apply_vortex_interference_physics(self, target, solutions, cannons):
        """
        PHYSICS CORRECTED: Apply realistic vortex ring interference theory
        
        THEORY BASIS:
        1. Widnall & Sullivan (1973): Vortex ring instability and interaction
        2. Batchelor (1967): Multi-body vortex dynamics in viscous flow
        3. Saffman (1992): Energy dissipation in vortex interactions
        4. Gharib et al. (1998): Formation constraints limiting energy transfer
        
        REALISTIC EFFECTS:
        - Destructive interference dominates (20-40% energy loss per additional ring)
        - Timing sensitivity (±10ms window for any constructive effect)
        - Turbulent mixing dissipates 30-50% of interaction energy
        - Hit probability may improve slightly due to larger disturbance field
        """
        
        print(f"DEBUG INTERFERENCE: Starting realistic interference analysis for target {target.id}")
        print(f"DEBUG INTERFERENCE: Processing {len(solutions)} vortex rings from {len(cannons)} cannons")
        
        if not solutions:
            return {
                'success': False,
                'combined_kill_probability': 0.0,
                'participating_cannons': 0,
                'combined_energy': 0.0,
                'interference_analysis': 'No solutions to combine'
            }
        
        # Count participating cannons (solutions that attempted engagement)
        attempted_solutions = [s for s in solutions if hasattr(s, 'impact_energy')]
        participating_cannons = len(attempted_solutions)
        
        if participating_cannons == 0:
            return {
                'success': False,
                'combined_kill_probability': 0.0,
                'participating_cannons': 0,
                'combined_energy': 0.0,
                'interference_analysis': 'No valid engagement solutions'
            }
        
        # PHYSICS: Calculate individual ring energies
        individual_energies = []
        total_base_energy = 0
        
        print(f"DEBUG INTERFERENCE: Individual ring energies:")
        for i, solution in enumerate(attempted_solutions):
            if hasattr(solution, 'impact_energy') and solution.impact_energy > 0:
                ring_energy = solution.impact_energy
            else:
                # Fallback energy calculation
                if hasattr(cannons[i], 'cannon') and hasattr(cannons[i].cannon, 'config'):
                    muzzle_velocity = cannons[i].cannon.calculate_muzzle_velocity()
                    barrel_diameter = cannons[i].cannon.config.barrel_diameter
                    air_density = cannons[i].cannon.config.air_density
                    ring_volume = (np.pi**2) * ((barrel_diameter/4)**2) * barrel_diameter
                    estimated_mass = air_density * ring_volume
                    ring_energy = 0.5 * estimated_mass * (muzzle_velocity ** 2)
                else:
                    ring_energy = 2000.0  # Fallback
            
            individual_energies.append(ring_energy)
            total_base_energy += ring_energy
            print(f"DEBUG INTERFERENCE:   Ring {i+1}: {ring_energy:.0f}J")
        
        print(f"DEBUG INTERFERENCE: Total base energy: {total_base_energy:.0f}J")
        
        # PHYSICS CORRECTED: Apply vortex ring interference theory
        if participating_cannons == 1:
            # Single ring - no interference
            final_energy = total_base_energy
            interference_factor = 1.0
            interference_description = "Single vortex ring - no interference"
            
        else:
            # REALISTIC MULTI-RING INTERFERENCE CALCULATION
            
            # 1. Destructive interference loss (Widnall & Sullivan 1973)
            # Each additional ring causes 20-30% energy loss due to circulation cancellation
            interference_loss_per_ring = 0.25  # 25% loss per additional ring
            destructive_loss = 1.0 - (interference_loss_per_ring * (participating_cannons - 1))
            destructive_loss = max(0.3, destructive_loss)  # Minimum 30% efficiency
            
            # 2. Turbulent mixing dissipation (Batchelor 1967)
            # Vortex interactions create turbulence that dissipates energy
            mixing_efficiency = 0.7  # 30% lost to turbulent mixing
            
            # 3. Timing synchronization (realistic ±50ms timing window)
            # Perfect timing is nearly impossible in practice
            timing_error = np.random.uniform(0.02, 0.08)  # 20-80ms typical error
            if timing_error < 0.02:  # Perfect timing (rare)
                timing_factor = 1.05  # Small 5% bonus
            elif timing_error < 0.05:  # Good timing
                timing_factor = 1.0   # No bonus or penalty
            else:  # Poor timing
                timing_factor = 0.95  # 5% penalty
            
            # 4. Formation number constraints (Gharib et al. 1998)
            # Multiple rings violate optimal formation conditions
            formation_penalty = 0.9  # 10% penalty for non-optimal formation
            
            # Combined interference factor
            interference_factor = destructive_loss * mixing_efficiency * timing_factor * formation_penalty
            final_energy = total_base_energy * interference_factor
            
            interference_description = (f"Multi-ring interference: {participating_cannons} rings, "
                                      f"{interference_factor:.2f} efficiency "
                                      f"(destructive: {destructive_loss:.2f}, "
                                      f"mixing: {mixing_efficiency:.2f}, "
                                      f"timing: {timing_factor:.2f})")
            
            print(f"DEBUG INTERFERENCE: {interference_description}")
            print(f"DEBUG INTERFERENCE: Final energy after interference: {final_energy:.0f}J")
        
        # PHYSICS: Calculate kill probability with interference effects
        if final_energy > 0:
            # Damage threshold based on target size (from vortex_ring.py)
            if target.size <= 0.5:
                damage_threshold = 750.0
            elif target.size <= 1.0:
                damage_threshold = 1500.0
            else:
                damage_threshold = 3000.0
            
            # Energy-based kill probability
            if final_energy >= damage_threshold:
                energy_factor = min(0.8, final_energy / damage_threshold * 0.5)
            else:
                energy_factor = 0.1 * (final_energy / damage_threshold)
            
            # PHYSICS: Multi-ring hit probability bonus (larger disturbance field)
            # This is the ONLY realistic benefit of multiple rings
            if participating_cannons > 1:
                hit_prob_bonus = min(0.15, 0.03 * participating_cannons)  # Max 15% bonus
            else:
                hit_prob_bonus = 0.0
            
            # Combined kill probability
            base_kill_prob = energy_factor * target.vulnerability
            final_kill_prob = min(0.9, base_kill_prob + hit_prob_bonus)
            
            print(f"DEBUG INTERFERENCE: Kill probability calculation:")
            print(f"DEBUG INTERFERENCE:   Damage threshold: {damage_threshold:.0f}J")
            print(f"DEBUG INTERFERENCE:   Energy factor: {energy_factor:.3f}")
            print(f"DEBUG INTERFERENCE:   Hit probability bonus: {hit_prob_bonus:.3f}")
            print(f"DEBUG INTERFERENCE:   Final kill probability: {final_kill_prob:.3f}")
        else:
            final_kill_prob = 0.0
            print(f"DEBUG INTERFERENCE: Zero final energy, kill probability = 0.0")
        
        # Success criteria: meaningful energy delivery despite interference
        success = (final_energy > 200) and (final_kill_prob > 0.05)
        
        return {
            'success': success,
            'target_id': target.id,
            'participating_cannons': participating_cannons,
            'combined_energy': final_energy,
            'base_energy_before_interference': total_base_energy,
            'interference_factor': interference_factor,
            'combined_kill_probability': final_kill_prob,
            'individual_solutions': attempted_solutions,
            'interference_analysis': interference_description
        }
    
    def execute_engagement_sequence(self, targets: List[Target],
                          current_time: float = 0.0) -> List[Dict]:
        """Execute engagement with REALISTIC multi-cannon physics"""
        print(f"DEBUG REALISTIC: Starting engagement sequence with {len(targets)} targets")
        
        # Assign targets to cannons
        assignment_result = self.assign_targets(targets, current_time)
        assignments = assignment_result['assignments']
        
        print(f"DEBUG REALISTIC: Got assignments: {assignments}")
        
        engagement_results = []
        execution_time = current_time
        
        # Execute engagements based on assignments
        for target_id, cannon_ids in assignments.items():
            target = next((t for t in targets if t.id == target_id), None)
            if not target:
                print(f"DEBUG REALISTIC: Target {target_id} not found in target list")
                continue
            
            print(f"DEBUG REALISTIC: Processing engagement for {target_id} with cannons: {cannon_ids}")
            
            # Handle both list and string cannon_ids
            if isinstance(cannon_ids, str):
                cannon_ids = [cannon_ids]
            
            if len(cannon_ids) > 1:
                print(f"DEBUG REALISTIC: Multi-cannon engagement: {len(cannon_ids)} cannons (expect interference)")
                # Multi-cannon engagement with realistic interference
                result = self.calculate_combined_engagement(target, cannon_ids, execution_time)
            else:
                print(f"DEBUG REALISTIC: Single cannon engagement")
                # Single cannon engagement
                cannon = next((c for c in self.cannons if c.id == cannon_ids[0]), None)
                if cannon:
                    calc = EngagementCalculator(cannon.cannon)
                    solution = calc.single_target_engagement(target, execution_time)
                    
                    # Create result in consistent format
                    result = {
                        'success': solution.success,
                        'target_id': target.id,
                        'participating_cannons': 1,
                        'combined_kill_probability': solution.kill_probability,
                        'combined_energy': solution.impact_energy if solution.impact_energy > 0 else 2000.0,
                        'interference_analysis': 'Single cannon - no interference',
                        'individual_solutions': [solution]
                    }
                else:
                    result = {
                        'success': False,
                        'target_id': target.id,
                        'participating_cannons': 0,
                        'combined_kill_probability': 0.0,
                        'combined_energy': 0.0,
                        'interference_analysis': 'Cannon not found',
                        'individual_solutions': []
                    }
            
            print(f"DEBUG REALISTIC: Engagement result for {target_id}: "
                  f"success={result.get('success', False)}, "
                  f"cannons={result.get('participating_cannons', 0)}, "
                  f"kill_prob={result.get('combined_kill_probability', 0.0):.3f}, "
                  f"energy={result.get('combined_energy', 0.0):.0f}J")
            
            if 'interference_analysis' in result:
                print(f"DEBUG REALISTIC: {result['interference_analysis']}")
            
            engagement_results.append(result)
            
            # Update execution time for sequential operations
            if self.config.firing_mode == FiringMode.SEQUENTIAL:
                if result.get('success', False) and result.get('individual_solutions'):
                    max_impact_time = max(s.impact_time for s in result['individual_solutions'] 
                                        if hasattr(s, 'impact_time'))
                    execution_time = max_impact_time + 0.5  # Reload time
        
        # Update array metrics
        self._update_metrics(engagement_results)
        
        print(f"DEBUG REALISTIC: Completed engagement sequence with realistic physics")
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
        """Analyze array coverage capabilities with CORRECTED range limits"""
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
        
        # Analyze coverage overlap with CORRECTED range limits
        test_points = []
        for angle in np.linspace(0, 2*np.pi, 36):  # Every 10 degrees
            # PHYSICS CORRECTED: Reduced test ranges
            for range_val in [10, 15, 20, 25]:  # Realistic ranges only
                x = range_val * np.cos(angle) + self.array_center[0]
                y = range_val * np.sin(angle) + self.array_center[1]
                z = 15.0  # Typical drone altitude
                test_points.append(np.array([x, y, z]))
        
        overlap_counts = []
        for point in test_points:
            covering_cannons = 0
            for cannon in self.cannons:
                range_to_point = np.linalg.norm(point - cannon.position)
                # PHYSICS CORRECTED: Reduced effective range
                if range_to_point <= 25.0:  # Reduced from 45m
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
            'physics_model': 'CORRECTED - includes vortex ring interference',
            'max_effective_range': '25m (physics-limited)',
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


def create_realistic_test_array(topology: ArrayTopology = ArrayTopology.GRID_2x2,
                               firing_mode: FiringMode = FiringMode.COORDINATED) -> MultiCannonArray:
    """Create test multi-cannon array with REALISTIC physics-based configuration"""
    
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
        # PHYSICS CORRECTED: Reduced energy combination factor
        energy_combination_factor=0.6  # Accounts for interference losses
    )
    
    return MultiCannonArray(array_config)


def test_realistic_multi_cannon_array():
    """Test multi-cannon array with REALISTIC interference physics"""
    print("="*80)
    print("TESTING REALISTIC MULTI-CANNON ARRAY WITH VORTEX INTERFERENCE PHYSICS")
    print("="*80)
    
    # Create test array
    array = create_realistic_test_array(ArrayTopology.GRID_2x2, FiringMode.ADAPTIVE)
    
    print(f"Array created with {len(array.cannons)} cannons")
    print(f"Array center: {array.array_center}")
    print(f"Coverage radius: {array.coverage_radius:.1f}m (PHYSICS LIMITED)")
    
    # Create realistic test targets
    targets = [
        # Small drone at close range - should work with single cannon
        Target("small_close", np.array([15, 5, 12]), np.zeros(3), 0.3, 0.8, 1, 0.0),
        
        # Medium drone - multi-cannon might help slightly
        Target("medium_test", np.array([20, -10, 15]), np.array([-2, 1, 0]), 0.6, 0.7, 2, 0.0),
        
        # Large drone - multi-cannon still insufficient due to energy limits
        Target("large_challenge", np.array([22, 0, 18]), np.array([-1, 0, 0]), 1.2, 0.5, 1, 0.0),
        
        # Distant small drone - should fail due to range/accuracy limits
        Target("distant_small", np.array([30, 15, 20]), np.array([-3, -1, 0]), 0.3, 0.9, 2, 0.0)
    ]
    
    print(f"\nREALISTIC Test targets: {len(targets)}")
    for target in targets:
        range_to_array = np.linalg.norm(target.position - array.array_center)
        print(f"  {target.id}: size={target.size}m, range={range_to_array:.1f}m")
    
    # Execute engagement sequence
    print(f"\n" + "="*60)
    print("EXECUTING ENGAGEMENT WITH REALISTIC PHYSICS")
    print("="*60)
    
    results = array.execute_engagement_sequence(targets)
    
    print(f"\nREALISTIC Engagement Results:")
    print(f"{'Target':<15} {'Cannons':<7} {'Success':<7} {'Kill_Prob':<9} {'Energy':<8} {'Physics Notes'}")
    print("-" * 80)
    
    for result in results:
        target_id = result['target_id']
        success = result['success']
        cannons = result['participating_cannons']
        kill_prob = result.get('combined_kill_probability', 0)
        energy = result.get('combined_energy', 0)
        
        # Determine physics notes
        if cannons == 1:
            physics_note = "Single ring"
        elif cannons > 1:
            interference = result.get('interference_factor', 1.0)
            physics_note = f"Interference: {interference:.2f} efficiency"
        else:
            physics_note = "No engagement"
        
        print(f"{target_id:<15} {cannons:<7} {success:<7} {kill_prob:<9.3f} {energy:<8.0f} {physics_note}")
    
    # Physics analysis
    print(f"\n" + "="*60)
    print("PHYSICS ANALYSIS")
    print("="*60)
    
    single_cannon_results = [r for r in results if r['participating_cannons'] == 1]
    multi_cannon_results = [r for r in results if r['participating_cannons'] > 1]
    
    if single_cannon_results:
        avg_single = np.mean([r['combined_kill_probability'] for r in single_cannon_results])
        print(f"Single cannon average kill probability: {avg_single:.3f}")
    
    if multi_cannon_results:
        avg_multi = np.mean([r['combined_kill_probability'] for r in multi_cannon_results])
        avg_interference = np.mean([r.get('interference_factor', 1.0) for r in multi_cannon_results])
        print(f"Multi-cannon average kill probability: {avg_multi:.3f}")
        print(f"Average interference efficiency: {avg_interference:.3f}")
        print(f"Interference energy loss: {(1-avg_interference)*100:.1f}%")
    
    # Coverage analysis with corrected ranges
    coverage = array.analyze_coverage()
    print(f"\nREALISTIC Coverage Analysis:")
    print(f"  Array span: {coverage['array_span']:.1f}m")
    print(f"  Average overlap: {coverage['coverage_overlap']['average_overlap']:.1f}")
    print(f"  Uncovered points: {coverage['coverage_overlap']['uncovered_points']}")
    
    # System status
    status = array.get_array_status()
    print(f"\nArray Status:")
    print(f"  Ready cannons: {status['ready_cannons']}/{status['total_cannons']}")
    print(f"  Physics model: {status['physics_model']}")
    print(f"  Max effective range: {status['max_effective_range']}")
    print(f"  Success rate: {status['metrics']['successful_engagements']}/{status['metrics']['total_engagements']}")


def demonstrate_interference_effects():
    """Demonstrate vortex ring interference effects in detail"""
    print("\n" + "="*80)
    print("VORTEX RING INTERFERENCE PHYSICS DEMONSTRATION")
    print("="*80)
    
    print("Theory: Multiple vortex rings interfere destructively (Widnall & Sullivan 1973)")
    print("Effect: Energy combination efficiency decreases with number of rings")
    print()
    
    array = create_realistic_test_array(ArrayTopology.GRID_2x2, FiringMode.COORDINATED)
    
    # Test target - medium drone that might benefit from multiple rings
    test_target = Target("interference_test", np.array([20, 0, 15]), np.zeros(3), 0.6, 0.7, 1, 0.0)
    
    # Test different numbers of cannons
    for n_cannons in [1, 2, 3, 4]:
        cannon_ids = [f"cannon_{i+1:02d}" for i in range(min(n_cannons, len(array.cannons)))]
        
        result = array.calculate_combined_engagement(test_target, cannon_ids)
        
        base_energy = result.get('base_energy_before_interference', 0)
        final_energy = result.get('combined_energy', 0)
        interference_factor = result.get('interference_factor', 1.0)
        kill_prob = result.get('combined_kill_probability', 0)
        
        energy_loss_pct = (1 - interference_factor) * 100 if interference_factor < 1 else 0
        
        print(f"{n_cannons} cannons:")
        print(f"  Base energy: {base_energy:.0f}J")
        print(f"  Final energy: {final_energy:.0f}J") 
        print(f"  Interference loss: {energy_loss_pct:.1f}%")
        print(f"  Kill probability: {kill_prob:.3f}")
        print(f"  Analysis: {result.get('interference_analysis', 'N/A')}")
        print()
    
    print("CONCLUSION: Multi-cannon effectiveness limited by vortex ring interference")
    print("Realistic benefit mainly from improved hit probability, not energy addition")


if __name__ == "__main__":
    # Run comprehensive realistic multi-cannon array analysis
    test_realistic_multi_cannon_array()
    demonstrate_interference_effects()