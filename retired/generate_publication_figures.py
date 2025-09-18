#!/usr/bin/env python3
"""
Publication Figure Generator for Vortex Cannon Research

Generates all publication-quality figures for the vortex cannon research paper
using the standalone scripts/visualize.py tool. Creates journal-ready figures
with proper naming and organization.

Usage:
    python generate_publication_figures.py [--quick] [--format png] [--dpi 300]

Output: All figures saved to figs/ directory with systematic naming
"""

import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime
import argparse


class PublicationFigureGenerator:
    """Manages batch generation of all publication figures"""
    
    def __init__(self, format='png', dpi=300, quick_mode=False):
        self.format = format
        self.dpi = dpi
        self.quick_mode = quick_mode
        self.start_time = time.time()
        self.successful_figures = 0
        self.failed_figures = 0
        self.figure_list = []
        
    def log(self, message, level="INFO"):
        """Log messages with timestamps"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}] {level}:"
        print(f"{prefix} {message}")
        
    def generate_figure(self, figure_type, output_name, drone_type=None, timeout=120):
        """Generate a single figure using the visualize.py tool"""
        
        # Build command
        cmd = [
            sys.executable, 
            "scripts/visualize.py",
            "--figure-type", figure_type,
            "--output", output_name,
            "--dpi", str(self.dpi)
        ]
        
        if drone_type:
            cmd.extend(["--drone-type", drone_type])
        
        description = f"{figure_type}"
        if drone_type:
            description += f" ({drone_type})"
            
        self.log(f"Generating {description} figure...")
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=os.getcwd()
            )
            
            if result.returncode == 0:
                self.log(f"SUCCESS: {description} -> {output_name}")
                self.successful_figures += 1
                self.figure_list.append({
                    'name': output_name,
                    'type': figure_type,
                    'drone_type': drone_type,
                    'status': 'SUCCESS'
                })
                return True
            else:
                self.log(f"FAILED: {description}", "ERROR")
                if result.stderr:
                    self.log(f"Error: {result.stderr[:200]}...", "ERROR")
                self.failed_figures += 1
                self.figure_list.append({
                    'name': output_name,
                    'type': figure_type,
                    'drone_type': drone_type,
                    'status': 'FAILED',
                    'error': result.stderr
                })
                return False
                
        except subprocess.TimeoutExpired:
            self.log(f"TIMEOUT: {description} (>{timeout}s)", "ERROR")
            self.failed_figures += 1
            return False
        except Exception as e:
            self.log(f"EXCEPTION: {description} - {str(e)}", "ERROR")
            self.failed_figures += 1
            return False
    
    def generate_essential_figures(self):
        """Generate the core figures needed for publication"""
        self.log("=== GENERATING ESSENTIAL PUBLICATION FIGURES ===")
        
        essential_figures = [
            # Figure 1: Engagement envelopes for different drone types
            ("envelope", "fig1_envelope_small.png", "small"),
            ("envelope", "fig1_envelope_medium.png", "medium"),
            
            # Figure 2: Array configurations
            ("array-comparison", "fig2_array_comparison.png", None),
            
            # Figure 3: Performance comparison
            ("performance", "fig3_performance_analysis.png", None),
            
            # Figure 4: Trajectory physics
            ("trajectory", "fig4_trajectory_analysis.png", None),
            
            # Figure 5: Vehicle integration
            ("vehicle", "fig5_vehicle_integration.png", None)
        ]
        
        for figure_type, output_name, drone_type in essential_figures:
            self.generate_figure(figure_type, output_name, drone_type)
            
        return True
    
    def generate_supplementary_figures(self):
        """Generate additional figures for comprehensive analysis"""
        if self.quick_mode:
            self.log("Skipping supplementary figures (quick mode)")
            return True
            
        self.log("=== GENERATING SUPPLEMENTARY FIGURES ===")
        
        supplementary_figures = [
            # Additional envelope analysis
            ("envelope", "figS1_envelope_large.png", "large"),
            
            # Additional performance views
            ("performance", "figS2_detailed_performance.png", None),
            
            # Additional trajectory analysis
            ("trajectory", "figS3_detailed_trajectory.png", None),
        ]
        
        for figure_type, output_name, drone_type in supplementary_figures:
            self.generate_figure(figure_type, output_name, drone_type)
            
        return True
    
    def check_prerequisites(self):
        """Check if visualize.py and required modules are available"""
        self.log("Checking prerequisites...")
        
        # Check if visualize.py exists
        if not Path("scripts/visualize.py").exists():
            self.log("MISSING: scripts/visualize.py not found", "ERROR")
            return False
        
        # Check if figs directory exists (will be created by visualize.py)
        figs_dir = Path("figs")
        if not figs_dir.exists():
            self.log("Creating figs directory...")
            figs_dir.mkdir(exist_ok=True)
        
        # Test basic functionality
        try:
            result = subprocess.run(
                [sys.executable, "scripts/visualize.py", "--help"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode != 0:
                self.log("FAILED: visualize.py --help failed", "ERROR")
                return False
        except Exception as e:
            self.log(f"FAILED: Cannot run visualize.py: {e}", "ERROR")
            return False
        
        self.log("Prerequisites check passed")
        return True
    
    def generate_figure_index(self):
        """Generate an index file listing all created figures"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        index_file = f"figs/figure_index_{timestamp}.txt"
        
        try:
            with open(index_file, 'w') as f:
                f.write("PUBLICATION FIGURE INDEX\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Format: {self.format.upper()}, DPI: {self.dpi}\n")
                f.write(f"Success Rate: {self.successful_figures}/{self.successful_figures + self.failed_figures}\n\n")
                
                # Essential figures
                f.write("ESSENTIAL FIGURES:\n")
                f.write("-" * 30 + "\n")
                essential_types = ["envelope", "array-comparison", "performance", "trajectory", "vehicle"]
                for fig in self.figure_list:
                    if fig['type'] in essential_types and fig['status'] == 'SUCCESS':
                        desc = fig['type'].replace('-', ' ').title()
                        if fig['drone_type']:
                            desc += f" ({fig['drone_type'].title()})"
                        f.write(f"{fig['name']} - {desc}\n")
                
                # Failed figures
                failed_figs = [fig for fig in self.figure_list if fig['status'] == 'FAILED']
                if failed_figs:
                    f.write(f"\nFAILED FIGURES ({len(failed_figs)}):\n")
                    f.write("-" * 30 + "\n")
                    for fig in failed_figs:
                        f.write(f"{fig['name']} - {fig['type']}\n")
                
                # Usage instructions
                f.write(f"\nUSAGE INSTRUCTIONS:\n")
                f.write("-" * 30 + "\n")
                f.write("For journal submission:\n")
                f.write("1. Use fig1-fig5 as main figures\n")
                f.write("2. Include figS* as supplementary material\n")
                f.write("3. All figures are 300 DPI and print-ready\n")
                f.write("4. Figures use grayscale-compatible color scheme\n\n")
                
                f.write("For presentations:\n")
                f.write("1. Generate color versions if needed:\n")
                f.write("   python scripts/visualize.py --figure-type [type] --output [name]\n")
                f.write("2. Consider PDF format for vector graphics:\n")
                f.write("   python generate_publication_figures.py --format pdf\n")
            
            self.log(f"Figure index created: {index_file}")
            return index_file
            
        except Exception as e:
            self.log(f"Failed to create figure index: {e}", "ERROR")
            return None
    
    def print_summary(self):
        """Print generation summary"""
        duration = time.time() - self.start_time
        total_figures = self.successful_figures + self.failed_figures
        
        self.log("=== FIGURE GENERATION SUMMARY ===")
        self.log(f"Total runtime: {duration:.1f} seconds")
        self.log(f"Successful figures: {self.successful_figures}/{total_figures}")
        
        if self.successful_figures > 0:
            self.log("Generated figures:")
            for fig in self.figure_list:
                if fig['status'] == 'SUCCESS':
                    desc = fig['type'].replace('-', ' ').title()
                    if fig['drone_type']:
                        desc += f" ({fig['drone_type'].title()})"
                    self.log(f"  ✓ {fig['name']} - {desc}")
        
        if self.failed_figures > 0:
            self.log("Failed figures:", "WARNING")
            for fig in self.figure_list:
                if fig['status'] == 'FAILED':
                    self.log(f"  ✗ {fig['name']} - {fig['type']}", "WARNING")
        
        # Check file sizes
        self.log("Checking generated files...")
        figs_dir = Path("figs")
        if figs_dir.exists():
            fig_files = list(figs_dir.glob(f"fig*.{self.format}"))
            total_size = sum(f.stat().st_size for f in fig_files)
            self.log(f"Total size: {total_size/1024/1024:.1f} MB ({len(fig_files)} files)")
        
        # Next steps
        if self.successful_figures >= total_figures * 0.8:
            self.log("✓ Figure generation complete! Ready for publication submission.")
        else:
            self.log("⚠ Some figures failed. Review errors and retry if needed.", "WARNING")
    
    def run_complete_generation(self):
        """Run the complete figure generation workflow"""
        self.log("Starting publication figure generation")
        self.log(f"Output format: {self.format.upper()}, DPI: {self.dpi}")
        self.log(f"Quick mode: {'ENABLED' if self.quick_mode else 'DISABLED'}")
        
        # Check prerequisites
        if not self.check_prerequisites():
            self.log("Prerequisites check failed. Cannot continue.", "ERROR")
            return False
        
        # Generate essential figures
        if not self.generate_essential_figures():
            self.log("Essential figure generation failed.", "ERROR")
            return False
        
        # Generate supplementary figures
        if not self.generate_supplementary_figures():
            self.log("Supplementary figure generation failed.", "WARNING")
        
        # Generate figure index
        self.generate_figure_index()
        
        # Print summary
        self.print_summary()
        
        return self.failed_figures == 0


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Publication Figure Generator for Vortex Cannon Research",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Generated Figures:
  fig1_envelope_small.png     - Small drone engagement envelope
  fig1_envelope_medium.png    - Medium drone engagement envelope  
  fig2_array_comparison.png   - Vehicle-mounted array configurations
  fig3_performance_analysis.png - Single vs multi-cannon performance
  fig4_trajectory_analysis.png - Vortex ring trajectory physics
  fig5_vehicle_integration.png - Vehicle integration analysis

Examples:
  python generate_publication_figures.py                    # Generate all PNG figures
  python generate_publication_figures.py --format pdf       # Generate PDF figures
  python generate_publication_figures.py --quick            # Essential figures only
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: generate essential figures only')
    parser.add_argument('--format', choices=['png'],
                       default='png', help='Output format (png only - visualize.py limitation)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Output resolution (default: 300)')
    
    args = parser.parse_args()
    
    # Create generator and run
    generator = PublicationFigureGenerator(
        format=args.format,
        dpi=args.dpi,
        quick_mode=args.quick
    )
    
    try:
        success = generator.run_complete_generation()
        
        if success:
            print(f"\n✓ All figures generated successfully!")
            print(f"📁 Check the 'figs/' directory for your publication-ready figures")
            return 0
        else:
            print(f"\n⚠ Figure generation completed with some errors")
            print(f"📁 Check the 'figs/' directory for successfully generated figures")
            return 1
            
    except KeyboardInterrupt:
        generator.log("Figure generation interrupted by user.", "WARNING")
        return 1
    except Exception as e:
        generator.log(f"Unexpected error: {str(e)}", "ERROR")
        return 1


if __name__ == "__main__":
    sys.exit(main())
