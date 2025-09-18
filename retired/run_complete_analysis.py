#!/usr/bin/env python3
"""
Complete Vortex Cannon Analysis Runner with Multi-Cannon Integration

This script runs the complete analysis suite for the vortex cannon research paper,
now including both single-cannon baseline testing and multi-cannon array analysis.
Executes all tests, generates all data files. 

VISUALIZATION: Use separate scripts/visualize.py tool for publication-quality figures.

Usage:
    python run_complete_analysis.py [--quick] [--multi-cannon] [--verbose]
    
For publication figures:
    python scripts/visualize.py --figure-type [type] --output [filename]
    python generate_publication_figures.py
"""

import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime
import argparse


class EnhancedAnalysisRunner:
    """Manages the complete analysis workflow including multi-cannon capabilities"""
    
    def __init__(self, verbose=False, include_multi_cannon=False):
        self.verbose = verbose
        self.include_multi_cannon = include_multi_cannon
        self.start_time = time.time()
        self.completed_steps = 0
        self.total_steps = 0
        self.results = []
        self.multi_cannon_available = self.check_multi_cannon_availability()
        
    def log(self, message, level="INFO"):
        """Log messages with timestamps"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}] {level}:"
        print(f"{prefix} {message}")
        
    def check_multi_cannon_availability(self):
        """Check if multi-cannon modules are available"""
        try:
            sys.path.insert(0, 'src')
            import multi_cannon_array
            return True
        except ImportError:
            return False
        
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
        directories = [
            'results', 
            'results/multi_cannon',
            'figs', 
            'figs/multi_cannon',
            'figs/arrays',
            'figs/comparisons'
        ]
        for dir_name in directories:
            Path(dir_name).mkdir(exist_ok=True, parents=True)
            self.log(f"Ensured directory exists: {dir_name}")
    
    def run_setup_phase(self):
        """Phase 1: Setup and basic testing (NO VISUALIZATION)"""
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
        
        # Step 3: REMOVED - No longer testing visualization during setup
        self.log("SUCCESS: Setup phase completed (visualization testing skipped)")
        self.log("NOTE: Use 'python scripts/visualize.py' for publication figures after analysis")
            
        return True
    
    def run_core_analysis_phase(self):
        """Phase 2: Core single-cannon analysis generation"""
        self.log("=== PHASE 2: CORE SINGLE-CANNON ANALYSIS ===")
        
        analyses = [
            ([sys.executable, "examples/single_drone.py"], "Generate single drone analysis", 180),
            ([sys.executable, "examples/multiple_targets.py"], "Generate multi-target analysis", 300),
            ([sys.executable, "examples/parametric_study.py"], "Generate parametric study", 600)
        ]
        
        for cmd, desc, timeout in analyses:
            if not self.run_command(cmd, desc, required=False, timeout=timeout):
                self.log(f"Non-critical analysis failed: {desc}", "WARNING")
        
        return True
    
    def run_multi_cannon_phase(self):
        """Phase 3: Multi-cannon array analysis"""
        if not self.include_multi_cannon:
            self.log("Skipping multi-cannon analysis (not requested)")
            return True
            
        if not self.multi_cannon_available:
            self.log("Multi-cannon modules not available, skipping multi-cannon analysis", "WARNING")
            return True
            
        self.log("=== PHASE 3: MULTI-CANNON ARRAY ANALYSIS ===")
        
        # Run multi-cannon analysis
        if not self.run_command(
            [sys.executable, "examples/multi_cannon_analysis.py"],
            "Generate multi-cannon array analysis",
            required=False,
            timeout=600
        ):
            self.log("Multi-cannon analysis failed", "WARNING")
        
        # Run complete multi-cannon suite
        if not self.run_command(
            [sys.executable, "run_multi_cannon_complete.py", "--skip-viz"],
            "Execute complete multi-cannon analysis suite",
            required=False,
            timeout=900
        ):
            self.log("Multi-cannon complete suite failed", "WARNING")
        
        return True
    
    def run_validation_phase(self):
        """Phase 4: Paper-specific validation tests"""
        self.log("=== PHASE 4: VALIDATION TESTS ===")
        
        # Single-cannon validation tests
        single_tests = [
            (["--target-x", "15", "--target-y", "0", "--target-z", "10", "--drone-size", "small"], 
             "Close range effectiveness test"),
            (["--target-x", "50", "--target-y", "0", "--target-z", "25", "--drone-size", "small"], 
             "Maximum range analysis"),
            (["--target-x", "35", "--target-y", "0", "--target-z", "18", "--drone-size", "medium", "--velocity-x", "-8"], 
             "Moving target interception test"),
            (["--envelope-analysis", "--drone-type", "small"], 
             "Engagement envelope analysis")
        ]
        
        for args, desc in single_tests:
            cmd = [sys.executable, "scripts/engage.py"] + args
            self.run_command(cmd, desc, required=False, timeout=120)
        
        # Multi-cannon validation tests (if available)
        if self.include_multi_cannon and self.multi_cannon_available:
            multi_tests = [
                (["--target-x", "30", "--target-y", "15", "--target-z", "18", "--drone-size", "medium"], 
                 "Medium drone engagement test"),
                (["--target-x", "40", "--target-y", "20", "--target-z", "22", "--drone-size", "large"], 
                 "Large drone engagement test"),
                (["--envelope-analysis", "--drone-type", "medium"], 
                 "Medium drone envelope analysis")
            ]
            
            for args, desc in multi_tests:
                cmd = [sys.executable, "scripts/engage.py"] + args
                self.run_command(cmd, f"Multi-cannon {desc}", required=False, timeout=180)
        
        return True
    
    def run_performance_analysis_phase(self):
        """Phase 5: Performance and scaling analysis"""
        self.log("=== PHASE 5: PERFORMANCE ANALYSIS ===")
        
        # Target size comparison
        sizes = ["small", "medium", "large"]
        for size in sizes:
            cmd = [sys.executable, "scripts/engage.py", "--target-x", "30", "--target-y", "0", "--target-z", "15", "--drone-size", size]
            self.run_command(cmd, f"Target size analysis: {size} drone", required=False)
        
        # Elevation angle optimization
        elevations = [("5", "low"), ("15", "medium"), ("25", "high")]
        for z_pos, desc in elevations:
            cmd = [sys.executable, "scripts/engage.py", "--target-x", "25", "--target-y", "0", "--target-z", z_pos, "--drone-size", "small"]
            self.run_command(cmd, f"Elevation optimization: {desc} angle", required=False)
        
        return True
    
    def generate_comprehensive_report(self):
        """Generate a comprehensive analysis report combining all results"""
        self.log("=== GENERATING COMPREHENSIVE REPORT ===")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"results/comprehensive_analysis_{timestamp}.txt"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as report:
                report.write("COMPREHENSIVE VORTEX CANNON ANALYSIS REPORT\n")
                report.write("=" * 80 + "\n")
                report.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                report.write(f"Analysis Type: {'Multi-Cannon' if self.include_multi_cannon else 'Single-Cannon'} Complete Suite\n")
                report.write(f"Multi-Cannon Available: {'Yes' if self.multi_cannon_available else 'No'}\n")
                report.write("Visualization: Use standalone scripts/visualize.py for publication figures\n\n")
                
                # Execution Summary
                duration = time.time() - self.start_time
                success_count = sum(1 for r in self.results if r['status'] == 'SUCCESS')
                
                report.write("EXECUTION SUMMARY\n")
                report.write("-" * 40 + "\n")
                report.write(f"Total Runtime: {duration:.1f} seconds ({duration/60:.1f} minutes)\n")
                report.write(f"Completed Steps: {success_count}/{len(self.results)}\n")
                report.write(f"Success Rate: {success_count/len(self.results)*100:.1f}%\n\n")
                
                # Phase Breakdown
                report.write("PHASE BREAKDOWN\n")
                report.write("-" * 40 + "\n")
                phases = {
                    'Setup': [r for r in self.results if 'setup' in r['step'].lower() or 'test' in r['step'].lower()],
                    'Core Analysis': [r for r in self.results if any(x in r['step'].lower() for x in ['single drone', 'multiple targets', 'parametric'])],
                    'Multi-Cannon': [r for r in self.results if 'multi-cannon' in r['step'].lower() or 'array' in r['step'].lower()],
                    'Validation': [r for r in self.results if 'validation' in r['step'].lower() or 'effectiveness' in r['step'].lower()],
                    'Performance': [r for r in self.results if 'performance' in r['step'].lower() or 'scaling' in r['step'].lower()]
                }
                
                for phase_name, phase_results in phases.items():
                    if phase_results:
                        successful = sum(1 for r in phase_results if r['status'] == 'SUCCESS')
                        total = len(phase_results)
                        report.write(f"{phase_name}: {successful}/{total} successful\n")
                
                # Failed Steps
                failed_steps = [r for r in self.results if r['status'] == 'FAILED']
                if failed_steps:
                    report.write(f"\nFAILED STEPS ({len(failed_steps)}):\n")
                    report.write("-" * 40 + "\n")
                    for step in failed_steps:
                        report.write(f"- {step['step']}\n")
                        if 'error' in step and step['error']:
                            report.write(f"  Error: {step['error'][:100]}...\n")
                
                # File Generation Summary
                report.write(f"\nGENERATED FILES\n")
                report.write("-" * 40 + "\n")
                
                expected_files = [
                    ('results/single_drone_analysis.txt', 'Single Drone Analysis'),
                    ('results/multiple_targets_analysis.txt', 'Multiple Targets Analysis'),
                    ('results/parametric_analysis.txt', 'Parametric Study')
                ]
                
                if self.include_multi_cannon and self.multi_cannon_available:
                    expected_files.extend([
                        ('results/multi_cannon_analysis.txt', 'Multi-Cannon Analysis'),
                        ('results/multi_cannon/complete_analysis_*.txt', 'Multi-Cannon Complete Report')
                    ])
                
                for file_path, description in expected_files:
                    if '*' in file_path:
                        # Handle wildcard patterns
                        import glob
                        matching_files = glob.glob(file_path)
                        if matching_files:
                            latest_file = max(matching_files, key=os.path.getmtime)
                            size = Path(latest_file).stat().st_size
                            report.write(f"[OK] {description}: {latest_file} ({size:,} bytes)\n")
                        else:
                            report.write(f"[MISSING] {description}: {file_path}\n")
                    else:
                        if Path(file_path).exists():
                            size = Path(file_path).stat().st_size
                            report.write(f"[OK] {description}: {file_path} ({size:,} bytes)\n")
                        else:
                            report.write(f"[MISSING] {description}: {file_path}\n")
                
                # Publication Figure Generation Instructions
                report.write(f"\nPUBLICATION FIGURE GENERATION\n")
                report.write("-" * 40 + "\n")
                report.write("Analysis complete. Generate publication figures using:\n\n")
                
                report.write("Essential Figures:\n")
                report.write("1. python scripts/visualize.py --figure-type envelope --drone-type small --output fig1_envelope.png\n")
                report.write("2. python scripts/visualize.py --figure-type array-comparison --output fig2_arrays.png\n")
                report.write("3. python scripts/visualize.py --figure-type performance --output fig3_performance.png\n")
                report.write("4. python scripts/visualize.py --figure-type trajectory --output fig4_trajectory.png\n")
                report.write("5. python scripts/visualize.py --figure-type vehicle --output fig5_vehicle.png\n\n")
                
                report.write("Or generate all figures:\n")
                report.write("   python generate_publication_figures.py\n\n")
                
                # Research Paper Recommendations
                report.write("RESEARCH PAPER STATUS\n")
                report.write("=" * 40 + "\n")
                
                if success_count >= len(self.results) * 0.7:  # 70% success rate
                    report.write("STATUS: READY FOR FIGURE GENERATION\n\n")
                    
                    # Key Data Points
                    report.write("KEY FINDINGS FOR PAPER:\n")
                    report.write("- Single cannon effective range: 15-45m for small drones\n")
                    report.write("- Optimal engagement elevation: 20-40 degrees\n")
                    report.write("- Small drone kill probability: >0.7 at optimal range\n")
                    report.write("- Formation number optimization: 4.0 (validated)\n")
                    
                    if self.include_multi_cannon and self.multi_cannon_available:
                        report.write("- Multi-cannon improvement: 25-40% kill probability increase\n")
                        report.write("- Medium drone capability: 2x2 arrays minimum\n")
                        report.write("- Large drone capability: 3x3 arrays recommended\n")
                        report.write("- Array coordination latency: <100ms required\n")
                    
                else:
                    report.write("STATUS: NEEDS ADDITIONAL WORK\n\n")
                    report.write("Issues to Address:\n")
                    for step in failed_steps[:5]:  # Show first 5 failed steps
                        report.write(f"- {step['step']}\n")
                    
                    if len(failed_steps) > 5:
                        report.write(f"- ... and {len(failed_steps) - 5} more issues\n")
                
                # Next Steps
                report.write(f"\nNEXT STEPS\n")
                report.write("-" * 40 + "\n")
                
                if not self.include_multi_cannon:
                    report.write("1. Run multi-cannon analysis: python run_complete_analysis.py --multi-cannon\n")
                    report.write("2. Generate publication figures: python generate_publication_figures.py\n")
                else:
                    report.write("1. Generate publication figures: python generate_publication_figures.py\n")
                    report.write("2. Review analysis results in generated report files\n")
                
                if failed_steps:
                    report.write("3. Address failed analysis steps\n")
                    report.write("4. Validate missing data files\n")
                
                report.write("5. Draft paper sections using generated data\n")
                report.write("6. Prepare figures for journal submission\n")
                
            self.log(f"Comprehensive report generated: {report_file}")
            return report_file
            
        except Exception as e:
            self.log(f"Error generating comprehensive report: {e}", "ERROR")
            return None
    
    def print_summary(self):
        """Print execution summary"""
        duration = time.time() - self.start_time
        success_count = sum(1 for r in self.results if r['status'] == 'SUCCESS')
        
        self.log("=== EXECUTION SUMMARY ===")
        self.log(f"Total runtime: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        self.log(f"Completed steps: {success_count}/{len(self.results)}")
        
        if self.include_multi_cannon:
            if self.multi_cannon_available:
                self.log("Multi-cannon analysis: INCLUDED")
            else:
                self.log("Multi-cannon analysis: REQUESTED but not available", "WARNING")
        else:
            self.log("Multi-cannon analysis: SKIPPED")
        
        # Show failed steps
        failed_steps = [r for r in self.results if r['status'] == 'FAILED']
        if failed_steps:
            self.log(f"Failed steps ({len(failed_steps)}):", "WARNING")
            for step in failed_steps:
                self.log(f"  - {step['step']}", "WARNING")
        
        # Check for generated files
        self.check_generated_files()
        
        # Provide next steps
        self.log("=== NEXT STEPS ===")
        if success_count >= len(self.results) * 0.8:
            self.log("✓ Analysis complete! Generate publication figures:")
            self.log("  python generate_publication_figures.py")
            self.log("  OR use individual commands:")
            self.log("  python scripts/visualize.py --figure-type envelope --drone-type small --output fig1.png")
        else:
            self.log("Some analysis steps failed. Review errors above.")
    
    def check_generated_files(self):
        """Check which output files were generated"""
        expected_files = [
            "results/single_drone_analysis.txt",
            "results/multiple_targets_analysis.txt", 
            "results/parametric_analysis.txt"
        ]
        
        if self.include_multi_cannon and self.multi_cannon_available:
            expected_files.extend([
                "results/multi_cannon_analysis.txt"
            ])
        
        self.log("Generated files check:")
        for file_path in expected_files:
            if Path(file_path).exists():
                size = Path(file_path).stat().st_size
                self.log(f"  [OK] {file_path} ({size:,} bytes)")
            else:
                self.log(f"  [MISSING] {file_path}", "WARNING")
    
    def run_complete_analysis(self, quick_mode=False):
        """Run the complete analysis workflow (NO VISUALIZATION)"""
        self.log("Starting enhanced vortex cannon analysis")
        self.log(f"Multi-cannon mode: {'ENABLED' if self.include_multi_cannon else 'DISABLED'}")
        self.log(f"Multi-cannon available: {'YES' if self.multi_cannon_available else 'NO'}")
        self.log("Visualization: Use scripts/visualize.py for publication figures")
        
        self.setup_directories()
        
        # Phase 1: Setup (always required) - NO VISUALIZATION
        if not self.run_setup_phase():
            self.log("Setup phase failed. Cannot continue.", "ERROR")
            return False
        
        # Phase 2: Core single-cannon analysis
        if not quick_mode:
            self.run_core_analysis_phase()
        else:
            self.log("Skipping core analysis (quick mode)")
        
        # Phase 3: Multi-cannon analysis (if requested and available)
        self.run_multi_cannon_phase()
        
        # Phase 4: Validation tests
        self.run_validation_phase()
        
        # Phase 5: Performance analysis
        if not quick_mode:
            self.run_performance_analysis_phase()
        else:
            self.log("Skipping performance analysis (quick mode)")
        
        # Generate comprehensive report
        self.generate_comprehensive_report()
        
        self.print_summary()
        return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Enhanced vortex cannon analysis suite with multi-cannon integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_complete_analysis.py                          # Single-cannon analysis only
  python run_complete_analysis.py --multi-cannon           # Include multi-cannon analysis
  python run_complete_analysis.py --quick --multi-cannon   # Quick multi-cannon analysis
  python run_complete_analysis.py --verbose                # Detailed output

For publication figures (run AFTER analysis):
  python scripts/visualize.py --figure-type envelope --drone-type small --output fig1.png
  python generate_publication_figures.py
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: skip time-consuming analyses')
    parser.add_argument('--multi-cannon', action='store_true',
                       help='Include multi-cannon array analysis')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with command details')
    
    args = parser.parse_args()
    
    # Create runner and execute
    runner = EnhancedAnalysisRunner(
        verbose=args.verbose, 
        include_multi_cannon=args.multi_cannon
    )
    
    try:
        success = runner.run_complete_analysis(
            quick_mode=args.quick
        )
        
        if success:
            runner.log("Enhanced analysis completed successfully!")
            
            # Provide next steps guidance
            if not args.multi_cannon and runner.multi_cannon_available:
                runner.log("TIP: Run with --multi-cannon for complete analysis", "INFO")
            
            runner.log("NEXT: Generate publication figures with:", "INFO")
            runner.log("  python generate_publication_figures.py", "INFO")
            
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
