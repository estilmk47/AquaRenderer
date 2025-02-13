BLENDER_AVAILABLE = False
try:
    import bpy
    import bmesh
    BLENDER_AVAILABLE = hasattr(bpy, "context")  # Check if bpy has actual Blender functionality
except ImportError:
    BLENDER_AVAILABLE = False

import sys
import os
import numpy as np
import math

root = '.'
if BLENDER_AVAILABLE:
    root = bpy.data.filepath        # The filepath of the .blend file you run the script in
    root = os.path.dirname(root)    # The repo you are working in [most likely]
    venv = 'venv'
    pyhton_env_path = f'{root}\{venv}\Lib\site-packages'
    sys.path.append(pyhton_env_path)
    sys.path.append(root)

def print_progress(iteration, end_iteration, bar_length=50):
    """
    Prints a progress bar to the terminal.
    
    Args:
        iteration (int): Current iteration index (0-based).
        end_iteration (int): Total number of iterations.
        bar_length (int): Length of the progress bar in characters.
    """
    progress = (iteration + 1) / end_iteration  # Normalize progress (0 to 1)
    filled_length = int(bar_length * progress)  # Calculate filled bar length
    bar = "█" * filled_length + "-" * (bar_length - filled_length)  # Construct bar
    
    # Print progress bar
    sys.stdout.write(f"\rProgress: |{bar}| {iteration + 1}/{end_iteration} frames")
    
    # Flush and reset cursor unless it's the last iteration
    if iteration + 1 < end_iteration:
        sys.stdout.flush()
    else:
        print()  # Ensure the final line is printed properly


def delete_from_collection(collection_name):
    if not BLENDER_AVAILABLE: return
    """Deletes all objects in the specified collection except those named 'cube' or 'Cube'."""
    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]
        for obj in list(collection.objects):
            if obj.name.lower() != 'cube':
                bpy.data.objects.remove(obj, do_unlink=True)


def bmesh_find_vertecies_in_plane(bmesh: bmesh, direction='x', shift=0, threshold=0.001):
    verts = []
    for v in bmesh.verts:
        if direction == 'x':
            if math.fabs(v.co.x - shift) < threshold:
                verts.append(v)
        elif direction == 'y':
            if math.fabs(v.co.y - shift) < threshold:
                verts.append(v)
        elif direction == 'z':
            if math.fabs(v.co.z - shift) < threshold:
                verts.append(v)
    
    return tuple(verts)