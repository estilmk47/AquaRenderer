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

# PYTHON_CORE_LIBS
import math

# PYTHON SITE-PACKAGES
import vtkmodules.all as vtk
# print(f"vtk module version {vtk.VTK_VERSION}")

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
        for x in range(start_range, end_range, -1):
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

def fill_zeros(frame_count, length=8):
    """Formats the frame count with leading zeros."""
    return str(frame_count).zfill(length)

# def fill_zeros(frame_count, length=8):
#     zeros = '0' * (length - len(str(frame_count)))
#     return f'{zeros}{frame_count}'

def create_water_object(mesh_data: bmesh, object_name: str, collection: str, visible_keyframe=-1, as_pointcloud=False, subdevide=False):
    if not BLENDER_AVAILABLE: return
    """Creates a Blender object with the given mesh data and assigns the 'Water' material."""
    # Create a new mesh and object
    
    mesh = bpy.data.meshes.new(object_name)
    obj = bpy.data.objects.new(object_name, mesh)
    

    # Link the object to the collection
    if collection not in bpy.data.collections:
        collection_data = bpy.data.collections.new(collection)
        bpy.context.scene.collection.children.link(collection_data)
    bpy.data.collections[collection].objects.link(obj)

    # Assign 'water' material if it exists
    if 'water' in bpy.data.materials:
        obj.data.materials.append(bpy.data.materials['water'])

    #
    if 'mesh_2_pointcloud' in bpy.data.node_groups and as_pointcloud:
        modifier = obj.modifiers.new(name='Mesh to PointCloud', type='NODES')
        modifier.node_group = bpy.data.node_groups['mesh_2_pointcloud']

    # Smooth the surface
    if subdevide:
        # for face in bm.faces[-5:]:
        #     for edge in face.edges:
        #         edge_crease = bm.edges.get(edge.index)
        #         if edge_crease:
        #             edge_crease.smooth = False
        #             edge_crease.crease = 1.0

        subdiv = obj.modifiers.new(name="Subdivision", type='SUBSURF')
        subdiv.levels = 2
        subdiv.use_limit_surface = True
    
    # Handle visibility keyframes
    if visible_keyframe > -1:
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_render", frame=-1)
        obj.hide_render = False
        obj.keyframe_insert(data_path="hide_render", frame=visible_keyframe)
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_render", frame=visible_keyframe + 1)

    bm = mesh_data.copy()
    bm.to_mesh(mesh)
    bm.free()

if __name__ == "__main__":
    context = Context()
    context.read_control(f'{root}/data/examples/single_split_experiment/control.txt')
    context.read_ctrl(f'{root}/data/examples/single_split_experiment/ctrl.txt')

    fps = 24

    start_frame = 0
    end_frame = 1 #fps*1  # TODO: Set to 1 -> then setup blender to nice render settings on that one sample -> then change this to fit your simulation 721
    batch_size = 24  # Number of frames loaded at a time

    filepath = "data/examples/single_split_experiment/REEF3D_NHFLOW_VTU" # TODO: change this
    file_name_skeleton_0 = "REEF3D-NHFLOW-"
    file_name_skeleton_1 = ".pvtu"

    collection = 'Waves'
    frames_loaded = 0
    for batch_start in range(start_frame, end_frame, batch_size):
        batch_end = min(batch_start + batch_size, end_frame)  # Ensure last batch is handled properly

        # Clear previous objects before loading new batch
        delete_from_collection(collection_name=collection)

        # Load the batch of frames
        for frame in range(batch_start, batch_end):
            filepath_pvtu = f"{root}/{filepath}/{file_name_skeleton_0}{fill_zeros(frame)}{file_name_skeleton_1}"
            mesh = extract_NHFLOW_mesh_from_pvtu(
                filepath_pvtu, 
                context, 
                surface_only=True,
                with_attributes=True,
                return_mesh=True,
            )
            create_water_object(mesh, f'wave_{fill_zeros(frame, len(str(end_frame)))}', collection, frame, subdevide=True)
            frames_loaded += 1
        # Update and move the render window to avoid overwriting render settings
        bpy.context.scene.frame_start = batch_start
        bpy.context.scene.frame_end = batch_end - 1
        bpy.context.scene.frame_set(batch_start)

        # Trigger render
        print_progress(frames_loaded-int(math.round(batch_size/2)), end_frame)
        # bpy.ops.render.render(animation=True)
        print_progress(frames_loaded, frame)
