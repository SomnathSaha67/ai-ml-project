"""Weka CLI bridge for feature extraction (optional)."""

import sys
from pathlib import Path
import subprocess
from typing import Optional
import os

# Add paths
shared_path = Path(__file__).parent.parent.parent.parent / "shared" / "src"
if str(shared_path) not in sys.path:
    sys.path.insert(0, str(shared_path))

from shared.config import get_weka_jar_path


def check_weka_available() -> bool:
    """Check if Weka is available."""
    weka_path = get_weka_jar_path()
    if weka_path and Path(weka_path).exists():
        return True
    return False


def run_weka_filter(
    input_file: str,
    output_file: str,
    filter: str = "weka.filters.unsupervised.attribute.Remove",
    filter_options: str = "-R 1",
) -> bool:
    """
    Run Weka filter via CLI.
    
    Args:
        input_file: Input ARFF file.
        output_file: Output ARFF file.
        filter: Weka filter class.
        filter_options: Filter options.
        
    Returns:
        True if successful, False otherwise.
    """
    weka_path = get_weka_jar_path()
    
    if not weka_path or not Path(weka_path).exists():
        print(f"Weka JAR not found at {weka_path}")
        print(f"To use Weka features, set WEKA_JAR_PATH in .env")
        return False
    
    try:
        cmd = [
            "java", "-cp", weka_path,
            "weka.filters.AllFilter",
            "-i", input_file,
            "-o", output_file,
            "-F", f'"{filter} {filter_options}"',
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode == 0:
            print(f"Weka filter successful: {input_file} -> {output_file}")
            return True
        else:
            print(f"Weka filter failed: {result.stderr}")
            return False
    
    except Exception as e:
        print(f"Error running Weka: {e}")
        return False


def demonstrate_weka_bridge():
    """Demonstrate Weka bridge (optional execution)."""
    print("\n=== Weka Bridge Demonstration ===")
    
    if check_weka_available():
        print("Weka is available!")
        print("In a real scenario, could run feature extraction filters via Weka CLI")
    else:
        print("Weka is not installed or JAR path not configured")
        print("Set WEKA_JAR_PATH environment variable to use Weka features")
    
    print("Skipping Weka execution (optional)")
    print("=== Weka Bridge Demonstration Complete ===")
