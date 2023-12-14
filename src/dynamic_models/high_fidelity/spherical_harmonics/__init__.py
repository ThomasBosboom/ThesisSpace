import os
import glob

# Get the package directory
package_dir = os.path.dirname(__file__)

# Get a list of all Python files in the package directory
python_files = glob.glob(os.path.join(package_dir, '*.py'))

# Extract module names from file names
module_names = [os.path.splitext(os.path.basename(file))[0] for file in python_files]

# Exclude "__init__" itself from __all__
__all__ = [module for module in module_names if module != '__init__']