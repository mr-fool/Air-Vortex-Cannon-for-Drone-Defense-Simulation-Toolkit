#!/usr/bin/env python3
"""
Quick script to fix relative imports in src/ directory
Run this from your project root directory
"""

import os
import re

def fix_file(filepath):
    """Fix relative imports in a single file"""
    print(f"Checking {filepath}...")
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # Fix relative imports
        content = re.sub(r'from \.(\w+) import', r'from \1 import', content)
        content = re.sub(r'from \. import (\w+)', r'import \1', content)
        
        if content != original_content:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"[FIXED] Fixed imports in {filepath}")
            return True
        else:
            print(f"[OK] No changes needed in {filepath}")
            return False
            
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        return False

def main():
    src_files = [
        'src/cannon.py',
        'src/engagement.py', 
        'src/vortex_ring.py'
    ]
    
    print("Fixing relative imports...")
    print("=" * 40)
    
    fixed = 0
    for filepath in src_files:
        if os.path.exists(filepath):
            if fix_file(filepath):
                fixed += 1
        else:
            print(f"[NOT FOUND] File not found: {filepath}")
    
    print("=" * 40)
    print(f"Fixed {fixed} files. You can now run:")
    print("python scripts/engage.py --target-x 30 --target-y 0 --target-z 15 --drone-size small")

if __name__ == "__main__":
    main()
