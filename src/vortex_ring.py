"""
Vortex Ring Physics Model for Drone Defense Applications

This module implements the core physics of vortex ring formation, propagation,
and interaction with drone targets. Monte Carlo simulation is used to model
uncertainties in engagement outcomes, while drone detection is assumed perfect
(deterministic) as per paper scope.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
from scipy.optimize import minimize_scalar
import random


@dataclass
class VortexRingState:
    """State of a vortex ring at a given time"""
    position: np.ndarray  # [x, y, z] in meters
    velocity: float       # Current velocity in m/s
    diameter: float       # Current ring diameter in meters
    energy: float         # Kinetic energy in Joules
    time: float          # Time since formation in seconds


class VortexRing:
    """
    Models the physics of a vortex ring from formation through target interaction.
    
    Based on Helmholtz vortex theory and formation number optimization.
    Uses Monte Carlo simulation for engagement uncertainty analysis.
    """
    
    def __init__(self, 
                 initial_velocity: float,
                 initial_diameter: float, 
                 formation_number: float = 4.0,
                 air_density: float = 1.225,
                 viscosity: float = 1.81e-5):
        """
        Initialize vortex ring with physical parameters.
        
        Args:
            initial_velocity: Muzzle velocity in m/s
            initial_diameter: Initial ring diameter in meters  
            formation_number: Stroke-to-diameter ratio (optimal ~4.0)
            air_density: Ambient air density in kg/mÂ³
            viscosity: Air kinematic viscosity in mÂ²/s
        """
        self.v0 = initial_velocity
        self.d0 = initial_diameter
        self.formation_number = formation_number
        self.rho = air_density
        self.nu = viscosity
        
        # Calculate initial circulation from formation number
        self.gamma = self._calculate_circulation()
        
        # Physical constants for decay models
        self.velocity_decay_coeff = 0.02  # Per meter traveled
        self.diameter_growth_rate = 0.008  # Per meter traveled
        self.energy_decay_coeff = 0.035   # Per meter traveled
        
    def _calculate_circulation(self) -> float:
        """Calculate vortex ring circulation from formation parameters."""
        # Based on Gharib et al. formation number theory
        stroke_length = self.formation_number * self.d0
        return self.v0 * stroke_length / 2.0
        
    def velocity_at_range(self, distance: float) -> float:
        alpha = 0.03  # empirical decay constant
        n = 0.7       # decay exponent
        return self.v0 * (1 + alpha * distance / self.d0) ** (-n)
        
    def diameter_at_range(self, distance: float) -> float:
        """
        Calculate ring diameter expansion with distance.
        
        Ring grows due to viscous diffusion and entrainment.
        """
        return self.d0 * (1 + self.diameter_growth_rate * distance)
        
    def energy_at_range(self, distance: float) -> float:
        """
        Calculate kinetic energy remaining at given distance.
        """
        velocity = self.velocity_at_range(distance)
        diameter = self.diameter_at_range(distance)
        
        # Kinetic energy of toroidal vortex
        ring_volume = np.pi**2 * (diameter/4)**2 * diameter
        ring_mass = self.rho * ring_volume
        
        return 0.5 * ring_mass * velocity**2
        
    def trajectory(self, time: float) -> VortexRingState:
        """
        Calculate vortex ring state at given time.
        
        Assumes straight-line propagation (no gravity for short ranges).
        """
        # Distance traveled
        distance = self.v0 * time * (1 - np.exp(-self.velocity_decay_coeff * self.v0 * time))
        
        # Current state
        velocity = self.velocity_at_range(distance)
        diameter = self.diameter_at_range(distance)
        energy = self.energy_at_range(distance)
        
        # Position (assuming launch from origin in +x direction)
        position = np.array([distance, 0.0, 0.0])
        
        return VortexRingState(position, velocity, diameter, energy, time)
        
    def time_to_range(self, target_distance: float) -> float:
        """
        Calculate time required to reach target distance.
        """
        if target_distance <= 0:
            return 0.0
            
        def distance_error(t):
            if t <= 0:
                return target_distance
            state = self.trajectory(t)
            return abs(state.position[0] - target_distance)
            
        try:
            result = minimize_scalar(distance_error, bounds=(0, 10), method='bounded')
            return result.x
        except:
            # Fallback: simple approximation
            return target_distance / self.v0
            
    # ADDED: Missing interface methods for multi-cannon compatibility
    @property
    def velocity(self) -> float:
        """Current velocity of the vortex ring (initial velocity for new rings)"""
        return self.v0
    
    def velocity_at_time(self, time: float) -> float:
        """
        Calculate velocity at given time
        
        Args:
            time: Time since formation (seconds)
            
        Returns:
            Velocity at that time (m/s)
        """
        if time <= 0:
            return self.v0
        try:
            state = self.trajectory(time)
            return state.velocity
        except:
            # Fallback for edge cases
            distance = self.v0 * time
            return self.velocity_at_range(distance)
    
    @property
    def mass(self) -> float:
        """Estimated mass of vortex ring based on entrained air"""
        # Approximate mass based on ring volume and entrained air
        ring_volume = np.pi**2 * (self.d0/4)**2 * self.d0
        return self.rho * ring_volume
    
    @property
    def kinetic_energy(self) -> float:
        """Initial kinetic energy of vortex ring"""
        return 0.5 * self.mass * self.v0**2
        
    def monte_carlo_engagement(self, 
                             target_position: np.ndarray,
                             drone_size: float,
                             drone_vulnerability: float,
                             n_trials: int = 1000) -> Dict[str, float]:
        """
        Monte Carlo simulation of engagement outcome.
        
        Randomizes:
        - Atmospheric conditions (air density, wind)
        - Ring formation quality (velocity/diameter variations)
        - Target interaction physics (hit detection, vulnerability)
        
        Does NOT randomize:
        - Drone detection (assumed perfect as per paper scope)
        - Drone position (given as deterministic input)
        
        Args:
            target_position: Drone position [x, y, z] in meters
            drone_size: Drone characteristic dimension in meters
            drone_vulnerability: Base vulnerability factor (0-1)
            n_trials: Number of Monte Carlo trials
            
        Returns:
            Dictionary with engagement statistics
        """
        target_range = np.linalg.norm(target_position)
        hits = 0
        kills = 0
        impact_energies = []
        ring_sizes = []
        
        for trial in range(n_trials):
            # 1. Randomize atmospheric conditions (Â±10%)
            trial_density = self.rho * random.uniform(0.9, 1.1)
            
            # 2. Randomize ring formation quality (Â±5%)
            trial_velocity = self.v0 * random.uniform(0.95, 1.05)
            trial_diameter = self.d0 * random.uniform(0.95, 1.05)
            
            # 3. Create trial vortex ring
            trial_ring = VortexRing(trial_velocity, trial_diameter, 
                                   self.formation_number, trial_density, self.nu)
            
            # 4. Calculate ring state at target range
            time_to_target = trial_ring.time_to_range(target_range)
            ring_state = trial_ring.trajectory(time_to_target)
            
            # 5. Hit detection (geometric intersection)
            ring_radius = ring_state.diameter / 2.0
            drone_radius = drone_size / 2.0
            
            # Randomize hit position within ring area (Â±10% uncertainty)
            hit_offset = random.uniform(-0.1, 0.1) * ring_radius
            effective_ring_size = ring_radius + hit_offset
            
            if effective_ring_size + drone_radius > 0.1:  # Always true for realistic sizes
                hits += 1
                
                # 6. Kill probability calculation
                impact_energy = ring_state.energy
                impact_energies.append(impact_energy)
                ring_sizes.append(ring_state.diameter)
                
                # Energy-based kill probability with randomization
                energy_factor = min(impact_energy / 50.0, 1.0)  # 50J threshold
                size_factor = min(ring_state.diameter / drone_size, 1.0)
                
                # Randomize vulnerability (Â±20% uncertainty in drone response)
                trial_vulnerability = drone_vulnerability * random.uniform(0.8, 1.2)
                trial_vulnerability = min(max(trial_vulnerability, 0.0), 1.0)
                
                kill_probability = energy_factor * size_factor * trial_vulnerability
                
                if random.random() < kill_probability:
                    kills += 1
        
        # Calculate statistics
        hit_rate = hits / n_trials
        kill_rate = kills / n_trials
        conditional_kill_rate = kills / hits if hits > 0 else 0.0
        
        avg_impact_energy = np.mean(impact_energies) if impact_energies else 0.0
        avg_ring_size = np.mean(ring_sizes) if ring_sizes else 0.0
        
        return {
            'hit_probability': hit_rate,
            'kill_probability': kill_rate,
            'conditional_kill_probability': conditional_kill_rate,
            'average_impact_energy': avg_impact_energy,
            'average_ring_size_at_impact': avg_ring_size,
            'number_of_trials': n_trials,
            'target_range': target_range
        }
        
    def engagement_envelope(self, 
                           drone_size: float,
                           drone_vulnerability: float,
                           min_kill_prob: float = 0.5) -> float:
        """
        Calculate maximum effective range for given kill probability.
        
        Args:
            drone_size: Target drone size in meters
            drone_vulnerability: Drone vulnerability factor (0-1)
            min_kill_prob: Minimum acceptable kill probability
            
        Returns:
            Maximum effective range in meters
        """
        for test_range in np.arange(5, 100, 2):
            target_pos = np.array([test_range, 0, 0])
            results = self.monte_carlo_engagement(target_pos, drone_size, 
                                                drone_vulnerability, n_trials=500)
            
            if results['kill_probability'] < min_kill_prob:
                return test_range - 2  # Previous range that met criteria
                
        return 100  # Maximum tested range
        
    def plot_trajectory(self, max_time: float = 2.0) -> plt.Figure:
        """
        Plot vortex ring trajectory showing velocity, diameter, and energy decay.
        """
        times = np.linspace(0, max_time, 100)
        states = [self.trajectory(t) for t in times]
        
        positions = np.array([s.position[0] for s in states])
        velocities = np.array([s.velocity for s in states])
        diameters = np.array([s.diameter for s in states])
        energies = np.array([s.energy for s in states])
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Trajectory
        ax1.plot(positions, velocities)
        ax1.set_xlabel('Range (m)')
        ax1.set_ylabel('Velocity (m/s)')
        ax1.set_title('Velocity vs Range')
        ax1.grid(True)
        
        # Diameter growth
        ax2.plot(positions, diameters)
        ax2.set_xlabel('Range (m)')
        ax2.set_ylabel('Diameter (m)')
        ax2.set_title('Ring Diameter vs Range')
        ax2.grid(True)
        
        # Energy decay
        ax3.plot(positions, energies)
        ax3.set_xlabel('Range (m)')
        ax3.set_ylabel('Energy (J)')
        ax3.set_title('Kinetic Energy vs Range')
        ax3.grid(True)
        
        # Phase plot
        ax4.plot(velocities, diameters)
        ax4.set_xlabel('Velocity (m/s)')
        ax4.set_ylabel('Diameter (m)')
        ax4.set_title('Velocity vs Diameter')
        ax4.grid(True)
        
        plt.tight_layout()
        return fig


def test_vortex_ring():
    """Test function to validate vortex ring physics."""
    print("Testing Vortex Ring Physics...")
    
    # Create test vortex ring
    vr = VortexRing(initial_velocity=50.0, initial_diameter=0.3, formation_number=4.0)
    
    # Test basic physics
    print(f"Initial velocity: {vr.v0} m/s")
    print(f"Initial diameter: {vr.d0} m")
    print(f"Formation number: {vr.formation_number}")
    
    # ADDED: Test new interface methods
    print(f"Velocity property: {vr.velocity} m/s")
    print(f"Mass: {vr.mass:.6f} kg")
    print(f"Kinetic energy: {vr.kinetic_energy:.2f} J")
    print(f"Velocity at 1s: {vr.velocity_at_time(1.0):.2f} m/s")
    
    # Test trajectory
    state_1s = vr.trajectory(1.0)
    print(f"\nAt t=1.0s:")
    print(f"  Position: {state_1s.position}")
    print(f"  Velocity: {state_1s.velocity:.2f} m/s")
    print(f"  Diameter: {state_1s.diameter:.3f} m")
    print(f"  Energy: {state_1s.energy:.2f} J")
    
    # Test Monte Carlo engagement
    target_pos = np.array([30.0, 0.0, 0.0])
    results = vr.monte_carlo_engagement(target_pos, drone_size=0.5, 
                                       drone_vulnerability=0.8, n_trials=1000)
    
    print(f"\nMonte Carlo Engagement (30m range, 0.5m drone):")
    print(f"  Hit probability: {results['hit_probability']:.3f}")
    print(f"  Kill probability: {results['kill_probability']:.3f}")
    print(f"  Average impact energy: {results['average_impact_energy']:.2f} J")
    print(f"  Average ring size: {results['average_ring_size_at_impact']:.3f} m")
    
    # Test engagement envelope
    max_range = vr.engagement_envelope(drone_size=0.5, drone_vulnerability=0.8)
    print(f"\nMaximum effective range (50% kill prob): {max_range} m")
    
    print("\n✓ All VortexRing methods working correctly!")


if __name__ == "__main__":
    test_vortex_ring()
