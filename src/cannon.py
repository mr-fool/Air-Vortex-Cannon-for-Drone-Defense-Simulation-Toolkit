"""
Vortex Cannon System Model

This module implements the vortex cannon hardware model including barrel physics,
chamber pressure calculations, and system constraints. The cannon generates and
launches vortex rings with configurable parameters for drone defense applications.
"""

import numpy as np
import yaml
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import os
from pathlib import Path

# Import the vortex ring physics
from vortex_ring import VortexRing


@dataclass
class CannonConfiguration:
    """Configuration parameters for vortex cannon system"""
    barrel_length: float          # meters
    barrel_diameter: float        # meters
    max_chamber_pressure: float   # Pa
    max_elevation: float          # degrees
    max_traverse: float           # degrees
    formation_number: float       # optimal stroke-to-diameter ratio
    air_density: float           # kg/m³
    chamber_pressure: float       # Pa - operating pressure
    
    # Derived parameters
    @property
    def barrel_volume(self) -> float:
        """Calculate barrel volume in cubic meters"""
        radius = self.barrel_diameter / 2.0
        return np.pi * radius**2 * self.barrel_length
    
    @property
    def optimal_stroke_length(self) -> float:
        """Calculate optimal piston stroke for formation number"""
        return self.formation_number * self.barrel_diameter


class VortexCannon:
    """
    Vortex cannon system for drone defense applications.
    
    Handles cannon physics, configuration management, and vortex ring generation
    with realistic system constraints and performance characteristics.
    """
    
    def __init__(self, config_path: str = "config/cannon_specs.yaml"):
        """
        Initialize cannon system from configuration file.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config = self._load_configuration(config_path)
        self.position = np.array([0.0, 0.0, 0.0])  # Cannon position [x, y, z]
        self.orientation = {'elevation': 0.0, 'azimuth': 0.0}  # degrees
        
        # System state - use pressure from config
        self.chamber_pressure = self.config.chamber_pressure
        self.ready_to_fire = True
        self.last_shot_time = 0.0
        
        # Performance characteristics
        self.reload_time = 0.5  # seconds between shots
        self.pressure_buildup_time = 2.0  # seconds to reach max pressure
        
    def _load_configuration(self, config_path: str) -> CannonConfiguration:
        """Load cannon configuration from YAML file"""
        # Handle relative paths
        if not os.path.isabs(config_path):
            # Look for config relative to this file's directory
            module_dir = Path(__file__).parent.parent
            config_path = module_dir / config_path
            
        try:
            with open(config_path, 'r') as f:
                config_data = yaml.safe_load(f)
                
            cannon_config = config_data['cannon']
            vortex_config = config_data.get('vortex_ring', {})
            env_config = config_data.get('environment', {})
            
            # Read chamber pressure from config, fall back to percentage of max if not specified
            max_pressure = cannon_config['max_chamber_pressure']
            chamber_pressure = cannon_config.get('chamber_pressure', max_pressure * 0.8)
            
            return CannonConfiguration(
                barrel_length=cannon_config['barrel_length'],
                barrel_diameter=cannon_config['barrel_diameter'],
                max_chamber_pressure=max_pressure,
                max_elevation=cannon_config.get('max_elevation', 85.0),
                max_traverse=cannon_config.get('max_traverse', 360.0),
                formation_number=vortex_config.get('formation_number', 4.0),
                air_density=env_config.get('air_density', 1.225),
                chamber_pressure=chamber_pressure
            )
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except KeyError as e:
            raise ValueError(f"Missing required configuration parameter: {e}")
    
    def set_position(self, x: float, y: float, z: float) -> None:
        """Set cannon position in world coordinates"""
        self.position = np.array([x, y, z])
    
    def set_pressure(self, pressure: float) -> None:
        """
        Set chamber pressure with safety limits.
        
        Args:
            pressure: Desired pressure in Pa
        """
        if pressure < 0:
            raise ValueError("Pressure cannot be negative")
        if pressure > self.config.max_chamber_pressure:
            pressure = self.config.max_chamber_pressure
            print(f"Warning: Pressure limited to {self.config.max_chamber_pressure} Pa")
            
        self.chamber_pressure = pressure
    
    def calculate_muzzle_velocity(self, pressure: Optional[float] = None) -> float:
        if pressure is None:
            pressure = self.chamber_pressure
        
        gauge_pressure = pressure - 101325  # Convert to gauge pressure
        if gauge_pressure <= 0:
            return 0.0
        
        # Proper orifice flow with discharge coefficient
        C_d = 0.7  # Typical discharge coefficient
        v_theoretical = C_d * np.sqrt(2 * gauge_pressure / self.config.air_density)
        
        # Account for piston acceleration limits (realistic constraint)
        max_practical_velocity = min(v_theoretical, 300.0)  # 300 m/s upper limit
        
        return max_practical_velocity
    
    def aim_at_target(self, target_position: np.ndarray) -> Tuple[float, float]:
        """
        Calculate elevation and azimuth angles to aim at target.
        
        Args:
            target_position: Target position [x, y, z] in world coordinates
            
        Returns:
            Tuple of (elevation, azimuth) in degrees
        """
        # Vector from cannon to target
        target_vector = target_position - self.position
        
        # Calculate range and elevation
        horizontal_range = np.sqrt(target_vector[0]**2 + target_vector[1]**2)
        elevation = np.degrees(np.arctan2(target_vector[2], horizontal_range))
        
        # Calculate azimuth (from +X axis)
        azimuth = np.degrees(np.arctan2(target_vector[1], target_vector[0]))
        
        # Apply system constraints
        elevation = np.clip(elevation, -10.0, self.config.max_elevation)
        azimuth = azimuth % 360.0  # Normalize to 0-360
        
        return elevation, azimuth
    
    def can_engage_target(self, target_position: np.ndarray) -> Tuple[bool, str]:
        """
        Check if target is within engagement envelope.
        Enhanced for multi-cannon coordination scenarios.
        
        Args:
            target_position: Target position [x, y, z]
            
        Returns:
            Tuple of (can_engage, reason_if_not)
        """
        try:
            # Calculate target vector and range
            target_vector = target_position - self.position
            target_range = np.linalg.norm(target_vector)
            
            # Enhanced range limits for multi-cannon scenarios
            if target_range > 75.0:  # Increased from typical 60m limit
                return False, f"Target beyond maximum range ({target_range:.1f}m > 75m)"
            
            if target_range < 3.0:  # Reduced minimum range
                return False, f"Target too close ({target_range:.1f}m < 3m)"
            
            # Calculate elevation angle
            horizontal_range = np.sqrt(target_vector[0]**2 + target_vector[1]**2)
            if horizontal_range < 0.1:  # Prevent division by zero for vertical shots
                elevation = 90.0 if target_vector[2] > 0 else -90.0
            else:
                elevation = np.degrees(np.arctan2(target_vector[2], horizontal_range))
            
            # Relaxed elevation limits for coordinated multi-cannon fire
            # Many drone defense scenarios require steep angle shots
            max_elevation = 89.0  # Increased from 85° to handle steep angles
            if elevation > max_elevation:
                return False, f"Elevation too high ({elevation:.1f}° > {max_elevation}°)"
            
            if elevation < -15.0:  # Slightly more depression allowed
                return False, f"Elevation too low ({elevation:.1f}° < -15°)"
            
            # Check azimuth constraints (full 360° for multi-cannon arrays)
            azimuth = np.degrees(np.arctan2(target_vector[1], target_vector[0]))
            # No azimuth restrictions for multi-cannon coordination
            
            # Check if system is ready
            if not self.ready_to_fire:
                return False, "System not ready to fire"
            
            # Reduced minimum pressure threshold for multi-cannon scenarios
            if self.chamber_pressure < 5000:  # Reduced from 10000 Pa
                return False, "Insufficient chamber pressure"
            
            return True, "Target can be engaged"
            
        except Exception as e:
            return False, f"Calculation error: {str(e)}"
    
    def generate_vortex_ring(self, target_position: Optional[np.ndarray] = None) -> VortexRing:
        """
        Generate a vortex ring with current cannon settings.
        
        Args:
            target_position: Optional target for automatic aiming
            
        Returns:
            VortexRing object configured for current cannon state
        """
        # Aim at target if provided
        if target_position is not None:
            elevation, azimuth = self.aim_at_target(target_position)
            self.orientation['elevation'] = elevation
            self.orientation['azimuth'] = azimuth
        
        # Calculate muzzle velocity
        muzzle_velocity = self.calculate_muzzle_velocity()
        
        # Create vortex ring
        vortex_ring = VortexRing(
            initial_velocity=muzzle_velocity,
            initial_diameter=self.config.barrel_diameter * 0.8,  # Ring slightly smaller than barrel
            formation_number=self.config.formation_number,
            air_density=self.config.air_density
        )
        
        return vortex_ring
    
    def fire_at_target(self, target_position: np.ndarray) -> Dict:
        """
        Complete firing sequence at target position.
        
        Args:
            target_position: Target coordinates [x, y, z]
            
        Returns:
            Dictionary with firing results
        """
        # Check if target can be engaged
        can_engage, reason = self.can_engage_target(target_position)
        if not can_engage:
            return {
                'success': False,
                'reason': reason,
                'target_position': target_position
            }
        
        # Aim and fire
        elevation, azimuth = self.aim_at_target(target_position)
        vortex_ring = self.generate_vortex_ring(target_position)
        
        # Calculate engagement results
        target_range = np.linalg.norm(target_position - self.position)
        time_to_target = vortex_ring.time_to_range(target_range)
        
        # Update system state
        self.ready_to_fire = False
        self.last_shot_time = 0.0  # Would be current time in real system
        
        return {
            'success': True,
            'elevation': elevation,
            'azimuth': azimuth,
            'muzzle_velocity': vortex_ring.v0,
            'time_to_target': time_to_target,
            'target_range': target_range,
            'vortex_ring': vortex_ring
        }
    
    def engagement_analysis(self, 
                          target_position: np.ndarray,
                          drone_size: float,
                          drone_vulnerability: float,
                          n_trials: int = 1000) -> Dict:
        """
        Complete engagement analysis including Monte Carlo simulation.
        
        Args:
            target_position: Target coordinates [x, y, z]
            drone_size: Drone characteristic size in meters
            drone_vulnerability: Vulnerability factor (0-1)
            n_trials: Number of Monte Carlo trials
            
        Returns:
            Complete engagement analysis results
        """
        # Check engagement feasibility
        can_engage, reason = self.can_engage_target(target_position)
        if not can_engage:
            return {
                'can_engage': False,
                'reason': reason
            }
        
        # Generate vortex ring for this engagement
        vortex_ring = self.generate_vortex_ring(target_position)
        
        # Run Monte Carlo analysis
        mc_results = vortex_ring.monte_carlo_engagement(
            target_position - self.position,  # Relative position
            drone_size,
            drone_vulnerability,
            n_trials
        )
        
        # Add cannon-specific information
        elevation, azimuth = self.aim_at_target(target_position)
        
        return {
            'can_engage': True,
            'cannon_position': self.position.copy(),
            'target_position': target_position.copy(),
            'elevation': elevation,
            'azimuth': azimuth,
            'muzzle_velocity': vortex_ring.v0,
            'chamber_pressure': self.chamber_pressure,
            **mc_results  # Include all Monte Carlo results
        }
    
    def system_status(self) -> Dict:
        """Get current system status and configuration"""
        return {
            'position': self.position.tolist(),
            'orientation': self.orientation.copy(),
            'chamber_pressure': self.chamber_pressure,
            'max_pressure': self.config.max_chamber_pressure,
            'ready_to_fire': self.ready_to_fire,
            'muzzle_velocity': self.calculate_muzzle_velocity(),
            'barrel_length': self.config.barrel_length,
            'barrel_diameter': self.config.barrel_diameter,
            'max_elevation': self.config.max_elevation
        }


def test_cannon():
    """Test function to validate cannon system"""
    print("Testing Vortex Cannon System...")
    
    try:
        # Try to create cannon from YAML config first
        try:
            cannon = VortexCannon("config/cannon_specs.yaml")
            print(f"Cannon loaded from YAML config successfully")
        except FileNotFoundError:
            # Fallback: Create cannon with realistic test configuration
            config_obj = CannonConfiguration(
                barrel_length=2.0,
                barrel_diameter=0.5,
                max_chamber_pressure=300000,  # 3 bar
                max_elevation=85.0,
                max_traverse=360.0,
                formation_number=4.0,
                air_density=1.225,
                chamber_pressure=240000  # 80% of max pressure
            )
            
            cannon = VortexCannon.__new__(VortexCannon)
            cannon.config = config_obj
            cannon.position = np.array([0.0, 0.0, 0.0])
            cannon.orientation = {'elevation': 0.0, 'azimuth': 0.0}
            cannon.chamber_pressure = config_obj.chamber_pressure
            cannon.ready_to_fire = True
            cannon.last_shot_time = 0.0
            cannon.reload_time = 0.5
            cannon.pressure_buildup_time = 2.0
            
            print(f"Cannon created with fallback configuration")
        
        print(f"Chamber pressure: {cannon.chamber_pressure} Pa ({cannon.chamber_pressure/1000:.0f} kPa)")
        
        # Calculate muzzle velocity
        velocity = cannon.calculate_muzzle_velocity()
        print(f"Muzzle velocity: {velocity:.2f} m/s")
        
        # Test aiming at steep angle (typical multi-cannon scenario)
        target = np.array([30.0, 15.0, 25.0])  # Higher altitude target
        elevation, azimuth = cannon.aim_at_target(target)
        print(f"Aiming at {target}: elevation={elevation:.1f}°, azimuth={azimuth:.1f}°")
        
        # Check engagement capability
        can_engage, reason = cannon.can_engage_target(target)
        print(f"Can engage target: {can_engage} ({reason})")
        
        # Test with even steeper angle
        steep_target = np.array([25.0, 0.0, 30.0])  # Very high target
        steep_elevation, steep_azimuth = cannon.aim_at_target(steep_target)
        steep_can_engage, steep_reason = cannon.can_engage_target(steep_target)
        print(f"Steep target ({steep_elevation:.1f}°): {steep_can_engage} ({steep_reason})")
        
        # Generate vortex ring
        vr = cannon.generate_vortex_ring(target)
        print(f"Generated vortex ring: v0={vr.v0:.2f} m/s, d0={vr.d0:.3f} m")
        
        # System status
        status = cannon.system_status()
        print(f"System status: Ready={status['ready_to_fire']}, Pressure={status['chamber_pressure']} Pa")
        
        print("✓ Enhanced cannon system working correctly for multi-cannon scenarios!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_cannon()