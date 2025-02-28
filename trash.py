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

root = '.'
if BLENDER_AVAILABLE:
    root = bpy.data.filepath        # The filepath of the .blend file you run the script in
    root = os.path.dirname(root)    # The repo you are working in [most likely]
    venv = 'venv'
    pyhton_env_path = f'{root}\{venv}\Lib\site-packages'
    sys.path.append(pyhton_env_path)
    sys.path.append(root)

## PROJECT_LIBS
from custom_blender_core import *
from divemesh_reef3d import Context

import vtkmodules.all as vtk

# PYTHON_CORE_LIBS
import math

def extract_NHFLOW_mesh_from_pvtu(filepath: str, context: Context, surface_only=False, shell_only=False, with_attributes=True, return_mesh=True):
    """
    Reads a VTU/PVTU file and extracts the mesh as a BMesh (if in Blender)
    or a NumPy array of vertices (if outside Blender).
    """
    
    # Check if the file is .pvtu (parallel) or .vtu (single file)
    if filepath.endswith(".pvtu"):
        reader = vtk.vtkXMLPUnstructuredGridReader()  # Use parallel reader
    else:
        reader = vtk.vtkXMLUnstructuredGridReader()  # Use single-file reader
    reader.SetFileName(filepath)
    reader.Update()

    # Get the mesh output from the reader
    mesh = reader.GetOutput()
    points = mesh.GetPoints()

    if points:
        start_range = points.GetNumberOfPoints() - 1
        end_range = 0
        if surface_only or shell_only:
            resolution_x = context.grid_resolution[0]
            resolution_y = context.grid_resolution[1]
            resolution_z = context.grid_resolution[2]
            # end_range = points.GetNumberOfPoints() - 1 - resolution_x*resolution_y
            end_range = (resolution_z)*(resolution_x+1)*(resolution_y+1) - 1
    
    # Create a new BMesh object
    if BLENDER_AVAILABLE:
        bm = bmesh.new()
        # Add custom layers for attributes
        if with_attributes:
            data_arrays = {
                "velocity": mesh.GetPointData().GetArray(0),
                "pressure": mesh.GetPointData().GetArray(1),
                "omega_sig": mesh.GetPointData().GetArray(2),
                "elevation": mesh.GetPointData().GetArray(3)
            }
            layer_velocity = bm.verts.layers.float_vector.new("velocity")
            layer_pressure = bm.verts.layers.float.new("pressure")
            layer_omega_sig = bm.verts.layers.float.new("omega_sig")
            layer_elevation = bm.verts.layers.float.new("elevation")
    else:
        num_points = int(math.fabs(end_range-start_range))
        np_verts = np.zeros((num_points, 3), dtype=np.float32)  # Allocate space

    # Add vertices to BMesh
    if points:
        ## TODO: ensure that the verts are added from entire surface scanning from x_end, y_end in the reverse x direction no matter the partition
        number_of_cores = context.mpi_cores
        skip_size = math.floor((points.GetNumberOfPoints() - end_range)/number_of_cores)
        skip_index = math.floor(end_range/number_of_cores)
        for i in range(start_range, end_range, -1):
            # skip = math.floor(i/skip_index)
            x = i # - skip*int(skip_size)
            a = [0, 0, 0]
            points.GetPoint(x, a)

            if BLENDER_AVAILABLE:
                v = bm.verts.new(a)
                if with_attributes:
                    v[layer_velocity] = data_arrays["velocity"].GetTuple3(x)
                    v[layer_pressure] = data_arrays["pressure"].GetValue(x)
                    v[layer_omega_sig] = data_arrays["omega_sig"].GetValue(x)
                    v[layer_elevation] = data_arrays["elevation"].GetValue(x)
            else:
                np_verts[(int(math.fabs(start_range-x))+4), :] = np.array(a, dtype=np.float32)  # Assign to NumPy array

        if shell_only:
            bottom_left     = [context.domain_dimentions[0], context.domain_dimentions[2], 0]
            bottom_right    = [context.domain_dimentions[1], context.domain_dimentions[2], 0]
            top_right       = [context.domain_dimentions[1], context.domain_dimentions[3], 0]
            top_left        = [context.domain_dimentions[0], context.domain_dimentions[3], 0]
            if BLENDER_AVAILABLE:                
                floor_verts = []
                floor_verts.append(bm.verts.new(bottom_left))
                floor_verts.append(bm.verts.new(bottom_right))
                floor_verts.append(bm.verts.new(top_right))
                floor_verts.append(bm.verts.new(top_left))
                if with_attributes:
                    for v in floor_verts:
                        v[layer_velocity] = (0, 0, 0)
                        v[layer_pressure] = 0
                        v[layer_omega_sig] = 0
                        v[layer_elevation] = 0
            else:
                np_verts[3, :] = bottom_left
                np_verts[2, :] = bottom_right
                np_verts[1, :] = top_right
                np_verts[0, :] = top_left


    # Only return the vertecies if the mesh is not required
    if not return_mesh and BLENDER_AVAILABLE:
        return bm      
    if not BLENDER_AVAILABLE:
        return np_verts 
    
    bm.verts.ensure_lookup_table()
    # Naive mesh generation approach for the surface:
    num_x = context.grid_resolution[0] + 1
    num_y = context.grid_resolution[1] + 1
    for y in range(context.grid_resolution[1]):
        for x in range(context.grid_resolution[0]):
            v1 = bm.verts[y * num_x + x]
            v2 = bm.verts[y * num_x + (x + 1)]
            v3 = bm.verts[(y + 1) * num_x + (x + 1)]
            v4 = bm.verts[(y + 1) * num_x + x]
            
            # Create the face from four vertices
            bm.faces.new((v1, v2, v3, v4))

    if not shell_only:
        return bm
    
    # Naive floor
    v1 = bm.verts[-4] # Bottom Left
    v2 = bm.verts[-3] # Bottom Right
    v3 = bm.verts[-2] # Top Right
    v4 = bm.verts[-1] # Top Left
    bm.faces.new((v4, v3, v2, v1))
    # Naive left
    verts = []
    for y in range(num_y):
        verts.append(bm.verts[(num_y-y)*num_x - 1])
    verts.append(v4)
    verts.append(v1)
    bm.faces.new(tuple(verts))
    # Naive right
    verts = []
    for y in range(num_y):
        verts.append(bm.verts[y*num_x])
    verts.append(v2)
    verts.append(v3)
    bm.faces.new(tuple(verts))
    # Naive back
    verts = []
    for x in range(num_x):
        verts.append(bm.verts[num_x-x-1])
    verts.append(v3)
    verts.append(v4)
    bm.faces.new(tuple(verts))
    # Naive front
    verts = []
    for x in range(num_x):
        verts.append(bm.verts[(num_y-1)*num_x+x])
    verts.append(v1)
    verts.append(v2)
    bm.faces.new(tuple(verts))

    return bm