#!/usr/bin/env python3
"""
Complete Vortex Cannon Analysis Runner

This script runs the complete analysis suite for the vortex cannon research paper.
Executes all tests, generates all data files, and creates all visualizations
in the correct order with proper error handling and progress reporting.

Usage:
    python run_complete_analysis.py [--quick] [--skip-viz] [--verbose]
"""

import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime
import argparse


class AnalysisRunner:
    """Manages the complete analysis workflow"""
    
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.start_time = time.time()
        self.completed_steps = 0
        self.total_steps = 0
        self.results = []
        
    def log(self, message, level="INFO"):
        """Log messages with timestamps"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}] {level}:"
        print(f"{prefix} {message}")
        
    def run_command(self, cmd, description, required=True, timeout=300):
        """Execute a command with error handling and progress tracking"""
        self.log(f"Running: {description}")
        if self.verbose:
            self.log(f"Command: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                timeout=timeout,
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                self.log(f"SUCCESS: {description}")
                self.completed_steps += 1
                self.results.append({
                    'step': description,
                    'status': 'SUCCESS',
                    'duration': time.time() - self.start_time
                })
                if self.verbose and result.stdout:
                    print(result.stdout)
                return True
            else:
                error_msg = f"FAILED: {description}"
                self.log(error_msg, "ERROR")
                if result.stderr:
                    self.log(f"Error output: {result.stderr}", "ERROR")
                if result.stdout:
                    self.log(f"Standard output: {result.stdout}", "DEBUG")
                
                self.results.append({
                    'step': description,
                    'status': 'FAILED',
                    'error': result.stderr,
                    'duration': time.time() - self.start_time
                })
                
                if required:
                    self.log(f"Required step failed. Stopping execution.", "ERROR")
                    return False
                return True
                
        except subprocess.TimeoutExpired:
            self.log(f"TIMEOUT: {description} (>{timeout}s)", "ERROR")
            return False
        except Exception as e:
            self.log(f"EXCEPTION: {description} - {str(e)}", "ERROR")
            return False
    
    def setup_directories(self):
        """Create necessary directories"""
        directories = ['results', 'figs']
        for dir_name in directories:
            Path(dir_name).mkdir(exist_ok=True)
            self.log(f"Ensured directory exists: {dir_name}")
    
    def run_setup_phase(self):
        """Phase 1: Setup and basic testing"""
        self.log("=== PHASE 1: SETUP AND TESTING ===")
        
        # Step 1: Fix imports
        if not self.run_command(
            [sys.executable, "import_fix.py"],
            "Fix Python imports",
            required=True
        ):
            return False
        
        # Step 2: Basic functionality test
        if not self.run_command(
            [sys.executable, "scripts/engage.py", "--target-x", "30", "--target-y", "0", "--target-z", "15", "--drone-size", "small"],
            "Test basic functionality",
            required=True
        ):
            return False
            
        return True
    
    def run_core_analysis_phase(self):
        """Phase 2: Core analysis generation"""
        self.log("=== PHASE 2: CORE ANALYSIS GENERATION ===")
        
        analyses = [
            ([sys.executable, "examples/single_drone.py"], "Generate single drone analysis", 180),
            ([sys.executable, "examples/multiple_targets.py"], "Generate multi-target analysis", 300),
            ([sys.executable, "examples/parametric_study.py"], "Generate parametric study", 600)
        ]
        
        for cmd, desc, timeout in analyses:
            if not self.run_command(cmd, desc, required=False, timeout=timeout):
                self.log(f"Non-critical analysis failed: {desc}", "WARNING")
        
        return True
    
    def run_validation_phase(self):
        """Phase 3: Paper-specific validation tests"""
        self.log("=== PHASE 3: PAPER VALIDATION TESTS ===")
        
        tests = [
            (["--target-x", "15", "--target-y", "0", "--target-z", "10", "--drone-size", "small"], 
             "Close range effectiveness test"),
            (["--target-x", "50", "--target-y", "0", "--target-z", "25", "--drone-size", "small"], 
             "Maximum range analysis"),
            (["--target-x", "35", "--target-y", "0", "--target-z", "18", "--drone-size", "medium", "--velocity-x", "-8"], 
             "Moving target interception test"),
            (["--envelope-analysis", "--drone-type", "small"], 
             "Engagement envelope analysis")
        ]
        
        for args, desc in tests:
            cmd = [sys.executable, "scripts/engage.py"] + args
            self.run_command(cmd, desc, required=False, timeout=120)
        
        return True
    
    def run_visualization_phase(self):
        """Phase 4: Generate visualizations"""
        self.log("=== PHASE 4: VISUALIZATION GENERATION ===")
        
        visualizations = [
            (["--envelope-plot", "--drone-type", "small", "--output", "figs/envelope_small.png"], 
             "Generate engagement envelope plot"),
            (["--trajectory-analysis", "--output", "figs/trajectory.png"], 
             "Generate trajectory analysis"),
            (["--target-x", "30", "--target-y", "10", "--target-z", "15", "--drone-size", "small", "--output", "figs/engagement_3d.png"], 
             "Generate 3D engagement visualization")
        ]
        
        for args, desc in visualizations:
            cmd = [sys.executable, "scripts/visualize.py"] + args
            self.run_command(cmd, desc, required=False, timeout=60)
        
        return True
    
    def run_additional_tests_phase(self):
        """Phase 5: Additional validation tests"""
        self.log("=== PHASE 5: ADDITIONAL VALIDATION TESTS ===")
        
        # Target size comparison
        sizes = ["small", "medium", "large"]
        for size in sizes:
            cmd = [sys.executable, "scripts/engage.py", "--target-x", "30", "--target-y", "0", "--target-z", "15", "--drone-size", size]
            self.run_command(cmd, f"Target size test: {size} drone", required=False)
        
        # Elevation angle optimization
        elevations = [("5", "low"), ("15", "medium"), ("25", "high")]
        for z_pos, desc in elevations:
            cmd = [sys.executable, "scripts/engage.py", "--target-x", "25", "--target-y", "0", "--target-z", z_pos, "--drone-size", "small"]
            self.run_command(cmd, f"Elevation optimization: {desc} angle", required=False)
        
        return True
    
    def print_summary(self):
        """Print execution summary"""
        duration = time.time() - self.start_time
        success_count = sum(1 for r in self.results if r['status'] == 'SUCCESS')
        
        self.log("=== EXECUTION SUMMARY ===")
        self.log(f"Total runtime: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        self.log(f"Completed steps: {success_count}/{len(self.results)}")
        
        # Show failed steps
        failed_steps = [r for r in self.results if r['status'] == 'FAILED']
        if failed_steps:
            self.log(f"Failed steps ({len(failed_steps)}):", "WARNING")
            for step in failed_steps:
                self.log(f"  - {step['step']}", "WARNING")
        
        # Check for generated files
        self.check_generated_files()
    
    def check_generated_files(self):
        """Check which output files were generated"""
        expected_files = [
            "results/single_drone_analysis.txt",
            "results/multiple_targets_analysis.txt", 
            "results/parametric_analysis.txt",
            "figs/envelope_small.png",
            "figs/trajectory.png",
            "figs/engagement_3d.png"
        ]
        
        self.log("Generated files check:")
        for file_path in expected_files:
            if Path(file_path).exists():
                size = Path(file_path).stat().st_size
                self.log(f"  [OK] {file_path} ({size:,} bytes)")
            else:
                self.log(f"  [FAIL] {file_path} (missing)", "WARNING")
    
    def run_complete_analysis(self, quick_mode=False, skip_visualizations=False):
        """Run the complete analysis workflow"""
        self.log("Starting complete vortex cannon analysis")
        self.setup_directories()
        
        # Phase 1: Setup (always required)
        if not self.run_setup_phase():
            self.log("Setup phase failed. Cannot continue.", "ERROR")
            return False
        
        # Phase 2: Core analysis
        if not quick_mode:
            self.run_core_analysis_phase()
        else:
            self.log("Skipping core analysis (quick mode)")
        
        # Phase 3: Validation tests
        self.run_validation_phase()
        
        # Phase 4: Visualizations
        if not skip_visualizations:
            self.run_visualization_phase()
        else:
            self.log("Skipping visualizations")
        
        # Phase 5: Additional tests
        if not quick_mode:
            self.run_additional_tests_phase()
        else:
            self.log("Skipping additional tests (quick mode)")
        
        self.print_summary()
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run complete vortex cannon analysis suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_complete_analysis.py                    # Full analysis
  python run_complete_analysis.py --quick            # Skip time-consuming steps
  python run_complete_analysis.py --skip-viz         # Skip visualizations
  python run_complete_analysis.py --verbose          # Detailed output
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: skip time-consuming analyses')
    parser.add_argument('--skip-viz', action='store_true',
                       help='Skip visualization generation')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with command details')
    
    args = parser.parse_args()
    
    # Create runner and execute
    runner = AnalysisRunner(verbose=args.verbose)
    
    try:
        success = runner.run_complete_analysis(
            quick_mode=args.quick,
            skip_visualizations=args.skip_viz
        )
        
        if success:
            runner.log("Analysis completed successfully!")
            return 0
        else:
            runner.log("Analysis completed with errors.", "WARNING")
            return 1
            
    except KeyboardInterrupt:
        runner.log("Analysis interrupted by user.", "WARNING")
        return 1
    except Exception as e:
        runner.log(f"Unexpected error: {str(e)}", "ERROR")
        return 1


if __name__ == "__main__":
    sys.exit(main())
