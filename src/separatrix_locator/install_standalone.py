#!/usr/bin/env python3
"""
Standalone installation script for the separatrix_locator package.

This script can be used to install the package independently of the original codebase.
"""

import os
import sys
import subprocess
from pathlib import Path

def install_package():
    """Install the separatrix_locator package in development mode."""
    print("🚀 Installing separatrix_locator package...")
    
    # Get the package directory
    package_dir = Path(__file__).parent
    setup_file = package_dir / "setup.py"
    
    if not setup_file.exists():
        print("❌ setup.py not found!")
        return False
    
    try:
        # Install in development mode
        cmd = [sys.executable, "-m", "pip", "install", "-e", str(package_dir)]
        print(f"Running: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Package installed successfully!")
            print("📦 You can now import separatrix_locator from anywhere")
            return True
        else:
            print("❌ Installation failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

def install_dependencies():
    """Install required dependencies."""
    print("📦 Installing dependencies...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("⚠️ requirements.txt not found, installing basic dependencies...")
        basic_deps = ["torch", "numpy", "scikit-learn", "matplotlib", "torchdiffeq"]
        
        for dep in basic_deps:
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                             check=True, capture_output=True)
                print(f"✅ Installed {dep}")
            except subprocess.CalledProcessError:
                print(f"⚠️ Failed to install {dep}")
    else:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)], 
                         check=True, capture_output=True)
            print("✅ Dependencies installed from requirements.txt")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install dependencies: {e}")

def verify_installation():
    """Verify that the installation works."""
    print("🔍 Verifying installation...")
    
    try:
        # Test import
        import separatrix_locator
        print("✅ Package import successful")
        
        # Test core functionality
        from separatrix_locator import SeparatrixLocator
        from separatrix_locator.dynamics import Bistable1D
        from separatrix_locator.core import LinearModel
        
        # Quick test
        dynamics = Bistable1D()
        locator = SeparatrixLocator(num_models=1, dynamics_dim=1, model_class=LinearModel)
        print("✅ Core functionality verified")
        
        return True
        
    except Exception as e:
        print(f"❌ Verification failed: {e}")
        return False

def main():
    """Main installation function."""
    print("🎯 Separatrix Locator Standalone Installation")
    print("=" * 50)
    
    # Install dependencies
    install_dependencies()
    
    # Install package
    if install_package():
        # Verify installation
        if verify_installation():
            print("\n🎉 Installation completed successfully!")
            print("\nYou can now use the package from anywhere:")
            print("  from separatrix_locator import SeparatrixLocator")
            print("  from separatrix_locator.dynamics import Bistable1D")
            return True
    
    print("\n❌ Installation failed!")
    return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
