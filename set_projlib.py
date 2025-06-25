# set_projlib.py
import os
import sys

# Set the correct PROJ_LIB path at runtime
proj_path = os.path.join(sys._MEIPASS, "pyproj_data")  # This matches 'pyproj_data' in datas
os.environ["PROJ_LIB"] = proj_path