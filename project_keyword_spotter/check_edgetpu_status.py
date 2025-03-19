#!/usr/bin/env python3
"""
Check the status of the Edge TPU configuration.

This script verifies the EdgeTPU installation, TensorFlow Lite runtime version, 
and hardware detection for the Coral USB Accelerator.
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if the Python version is compatible."""
    print(f"Python version: {platform.python_version()}")
    if sys.version_info.major < 3 or (sys.version_info.major == 3 and sys.version_info.minor < 9):
        print("Warning: Python 3.9+ is recommended for Edge TPU compatibility")
    else:
        print("Python version is compatible")

def check_tflite_runtime():
    """Check if TensorFlow Lite runtime is installed and its version."""
    try:
        import tflite_runtime
        print(f"TensorFlow Lite Runtime version: {tflite_runtime.__version__}")
        
        # Check if version matches recommended for Edge TPU
        if tflite_runtime.__version__ != "2.5.0.post1":
            print("Warning: Recommended TFLite runtime version for Edge TPU is 2.5.0.post1")
            print("To install the recommended version, run:")
            print("pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl")
    except ImportError:
        print("Error: TensorFlow Lite Runtime is not installed")
        print("To install for Edge TPU, run:")
        print("pip install https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_aarch64.whl")
        return False
    return True

def check_libedgetpu():
    """Check if the Edge TPU runtime library is installed."""
    system = platform.system()
    if system == "Linux":
        # Check common library locations
        lib_paths = [
            "/usr/lib/libedgetpu.so.1",
            "/usr/lib/aarch64-linux-gnu/libedgetpu.so.1",
            "/usr/local/lib/libedgetpu.so.1"
        ]
        
        found = False
        for path in lib_paths:
            if os.path.exists(path):
                print(f"Edge TPU library found at: {path}")
                found = True
                break
        
        if not found:
            # Try to find it elsewhere
            try:
                result = subprocess.run(["find", "/usr", "-name", "libedgetpu.so*"], 
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if result.stdout:
                    print(f"Edge TPU library found at: {result.stdout.strip()}")
                    found = True
            except Exception as e:
                print(f"Error searching for Edge TPU library: {e}")
        
        if not found:
            print("Error: Edge TPU library not found")
            print("To install:")
            print("  sudo apt-get update")
            print("  sudo apt-get install libedgetpu1-std")
            return False
    elif system == "Darwin":  # macOS
        lib_path = "/usr/local/lib/libedgetpu.1.dylib"
        if os.path.exists(lib_path):
            print(f"Edge TPU library found at: {lib_path}")
        else:
            print("Error: Edge TPU library not found for macOS")
            print("Follow installation instructions at https://coral.ai/docs/accelerator/get-started/")
            return False
    else:
        print(f"Unsupported platform for Edge TPU library check: {system}")
        return False
    
    return True

def check_coral_device():
    """Check if a Coral device is connected."""
    try:
        result = subprocess.run(["lsusb"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        output = result.stdout
        if "Coral" in output:
            print("Coral USB device detected!")
            for line in output.split('\n'):
                if "Coral" in line:
                    print(f"  {line.strip()}")
            return True
        else:
            print("No Coral USB device detected")
            return False
    except Exception as e:
        print(f"Error checking for Coral device: {e}")
        return False

def check_project_structure():
    """Check if the project has the necessary directory structure for Poetry."""
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    package_dir = os.path.join(project_dir, "project_keyword_spotter")
    
    print(f"Project directory: {project_dir}")
    print(f"Package directory: {package_dir}")
    
    if os.path.exists(os.path.join(project_dir, "pyproject.toml")):
        print("pyproject.toml found")
    else:
        print("Warning: pyproject.toml not found")
    
    if os.path.exists(package_dir):
        print("Package directory exists")
        if os.path.exists(os.path.join(package_dir, "__init__.py")):
            print("__init__.py found in package directory")
        else:
            print("Warning: __init__.py missing from package directory")
    else:
        print("Warning: Package directory does not exist")

def main():
    """Run all checks and print status."""
    print("=== Edge TPU Environment Check ===\n")
    
    print("System Information:")
    print(f"  Platform: {platform.platform()}")
    print(f"  Architecture: {platform.machine()}")
    print()
    
    print("Python Environment:")
    check_python_version()
    print()
    
    print("TensorFlow Lite Runtime:")
    tflite_ok = check_tflite_runtime()
    print()
    
    print("Edge TPU Library:")
    library_ok = check_libedgetpu()
    print()
    
    print("Coral Device:")
    device_ok = check_coral_device()
    print()
    
    print("Project Structure:")
    check_project_structure()
    print()
    
    print("=== Summary ===")
    if tflite_ok and library_ok and device_ok:
        print("✅ Environment is properly configured for Edge TPU")
    else:
        print("❌ Some issues were detected with the Edge TPU configuration")

if __name__ == "__main__":
    main()
