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
import pandas as pd
# print(f"vtk module version {vtk.VTK_VERSION}")

def exctract_NHFLOW_pointcloud_from_pvtu(filepath: str, context: Context, with_attributes=True):
    if not BLENDER_AVAILABLE:
        raise ValueError(f"Cannot use the function (exctract_NHFLOW_pointcloud_from_pvtu) outside of blender")
    
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

    # Create Blender return mesh
    bm = bmesh.new()
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

    for x in range(points.GetNumberOfPoints()):
        a = [0, 0, 0]
        points.GetPoint(x, a)
        v = bm.verts.new(a)
       
        if with_attributes:
            v[layer_velocity] = data_arrays["velocity"].GetTuple3(x)
            v[layer_pressure] = data_arrays["pressure"].GetValue(x)
            v[layer_omega_sig] = data_arrays["omega_sig"].GetValue(x)
            v[layer_elevation] = data_arrays["elevation"].GetValue(x)
    return bm

def extract_NHFLOW_mesh_from_pvtu(filepath: str, context: Context, surface_only=False, with_attributes=True, with_faces=True):
    """
    Reads a VTU/PVTU file and extracts the mesh as a BMesh (if in Blender)
    or a NumPy array of vertices (if outside Blender).
    """
    
    # Check if the file is .pvtu (parallel) or .vtu (single file)
    if filepath.endswith(".pvtu"):
        reader = vtk.vtkXMLPUnstructuredGridReader()    # Use parallel reader
    else:
        reader = vtk.vtkXMLUnstructuredGridReader()     # Use single-file reader
    reader.SetFileName(filepath)
    reader.Update()

    # Get the mesh output from the reader
    mesh = reader.GetOutput()
    points = mesh.GetPoints()

    resolution_x = context.grid_resolution[0]
    resolution_y = context.grid_resolution[1]
    points_x = resolution_x+1
    points_y = resolution_y+1
    surface_end_index = (points_x*points_y)
    start_x = context.domain_dimentions[0]
    end_x   = context.domain_dimentions[1]
    start_y = context.domain_dimentions[2]
    end_y   = context.domain_dimentions[3]

    # Create a new BMesh object
    if BLENDER_AVAILABLE:
        bm = bmesh.new()
        method_turn_into_structured_grid(
            bmesh=bm,
            x_start = start_x,
            x_end   = end_x,
            y_start = start_y,
            y_end   = end_y,
            x_resolution=resolution_x,
            y_resolution=resolution_y
        )

        # print(f"Number of verts in the surface: {len(bm.verts)}")
        # print(f"surface_end_index: {surface_end_index}")


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
        num_points = int(((points_x)*(points_y) + 2*(resolution_x + resolution_y)))
        np_verts = np.zeros((num_points, 3), dtype=np.float32)  # Allocate space

    if points:
        for x in range(points.GetNumberOfPoints()):
            a = [0, 0, 0]
            points.GetPoint(x, a)
            x_value = a[0]
            y_value = a[1]
            z_value = a[2]

            index = structured_grid_vert_index(
                x_value = x_value,
                y_value = y_value,
                x_start = start_x,
                x_end   = end_x,
                y_start = start_y,
                y_end   = end_y,
                x_resolution=resolution_x,
                y_resolution=resolution_y,
                bmesh=bm
            )

            if index == None:
                raise ValueError(f"Something went wrong reading the .pvtu. \nUnable to find index of point {a} in the allocated blender mesh")

            if BLENDER_AVAILABLE:
                v = bm.verts[index]
                if v.co.z < z_value:
                    v.co.z = z_value ## TODO: double check that v is passed by value and not by copy. If it is passed by copy you are in deep shit!
                    if with_attributes:
                        v[layer_velocity] = data_arrays["velocity"].GetTuple3(x)
                        v[layer_pressure] = data_arrays["pressure"].GetValue(x)
                        v[layer_omega_sig] = data_arrays["omega_sig"].GetValue(x)
                        v[layer_elevation] = data_arrays["elevation"].GetValue(x)
            else:
                if np_verts[index, 2] < z_value:
                    np_verts[index, 2] = z_value


        if not surface_only:
            start_index = surface_end_index
            end_index = start_index + 2*(resolution_x + resolution_y)
            dx = (end_x-start_x)/(resolution_x)
            dy = (end_y-start_y)/(resolution_y)

            # front edge 
            for i in range(resolution_x):
                x = start_x + i*dx
                y = start_y
                if BLENDER_AVAILABLE:
                    bm.verts.new((x,y,0))
                else:
                    np_verts[start_index+i,:] =  np.array([x,y,0], dtype=np.float32)

            # right edge
            for i in range(resolution_y):
                x = end_x
                y = start_y + i*dy
                if BLENDER_AVAILABLE:
                    bm.verts.new((x,y,0))
                else:
                    np_verts[start_index+resolution_x+i,:] =  np.array([x,y,0], dtype=np.float32)

            # back edge 
            for i in range(resolution_x):
                x = end_x - i*dx
                y = end_y
                if BLENDER_AVAILABLE:
                    bm.verts.new((x,y,0))
                else:
                    np_verts[start_index+resolution_x+resolution_y+i,:] =  np.array([x,y,0], dtype=np.float32)

            # left edge 
            for i in range(resolution_y):
                x = start_x
                y = end_y - i*dy
                if BLENDER_AVAILABLE:
                    bm.verts.new((x,y,0))
                else:
                    np_verts[start_index+2*resolution_x+resolution_y+i,:] =  np.array([x,y,0], dtype=np.float32)

            bm.verts.ensure_lookup_table()

            # print(f"Number of verts in the surface + bed edges: {len(bm.verts)}")
            # print(f"surface_end_index: {surface_end_index}")

            if BLENDER_AVAILABLE:              
                if with_attributes:
                    for i in range(start_index, end_index):
                        v = bm.verts[i]
                        v[layer_velocity] = (0, 0, 0)
                        v[layer_pressure] = 0
                        v[layer_omega_sig] = 0
                        v[layer_elevation] = 0


    # Only return the vertecies if the mesh is not required
    if not with_faces and BLENDER_AVAILABLE:
        return bm      
    if not BLENDER_AVAILABLE:
        return np_verts

    # Optimized mesh generation approach for the surface (NHFLOW assumptions):
    for y in range(resolution_y):
        for x in range(resolution_x):
            v1 = bm.verts[y * points_x + x]
            v2 = bm.verts[y * points_x + (x + 1)]
            v3 = bm.verts[(y + 1) * points_x + (x + 1)]
            v4 = bm.verts[(y + 1) * points_x + x]
            
            # Create the face from four vertices
            bm.faces.new((v1, v2, v3, v4))

    if not surface_only:
        # front faces:
        for x in range(resolution_x):
            v1 = bm.verts[x]
            v2 = bm.verts[surface_end_index + x]
            v3 = bm.verts[surface_end_index + (x+1)]
            v4 = bm.verts[(x+1)]  
            bm.faces.new((v1, v2, v3, v4))

        # right faces:
        for y in range(resolution_y):
            v1 = bm.verts[resolution_x + y*points_x]
            v2 = bm.verts[surface_end_index + resolution_x + y]
            v3 = bm.verts[surface_end_index + resolution_x + (y+1)]
            v4 = bm.verts[resolution_x + (y+1)*points_x]
            bm.faces.new((v1, v2, v3, v4))

        # back faces:
        for x in range(resolution_x):
            v1 = bm.verts[surface_end_index - 1 - x]
            v2 = bm.verts[surface_end_index + resolution_x + resolution_y + x]
            v3 = bm.verts[surface_end_index + resolution_x + resolution_y + (x+1)]
            v4 = bm.verts[surface_end_index - 1 - (x+1)]
            bm.faces.new((v1, v2, v3, v4))

        # left faces:
        for y in range(resolution_y):
            v1 = bm.verts[surface_end_index - points_x - y*points_x]
            v2 = bm.verts[surface_end_index + 2*resolution_x + resolution_y + y]
            if y != resolution_y-1: v3 = bm.verts[surface_end_index + 2*resolution_x + resolution_y + y + 1]
            else: v3 = bm.verts[surface_end_index]
            v4 = bm.verts[surface_end_index - points_x - (y+1)*points_x]
            bm.faces.new((v1, v2, v3, v4))

        # bottom face: NB: This makes the water surface "unconnected" but that is miles better than creating ngons. DO NOT MAKE N-GONS!!!
        v1 = bm.verts[surface_end_index]
        v2 = bm.verts[surface_end_index + 2*resolution_x + resolution_y]
        v3 = bm.verts[surface_end_index + resolution_x + resolution_y]
        v4 = bm.verts[surface_end_index + resolution_x]
        bm.faces.new((v1, v2, v3, v4))

        bm.faces.ensure_lookup_table()
        bm.edges.ensure_lookup_table()

    return bm

def extract_NHFLOW_6DOF(filepath, frame):
    df = pd.read_csv(filepath, sep='\s+')
    if frame >= len(df):
        print("Frame out of range")
        return None, None
    
    row = df.iloc[frame]
    position = (row['XG'], row['YG'], row['ZG'])
    rotation_zyx = tuple(math.radians(row[angle]) for angle in ['Psi', 'Theta', 'Phi']) # Convert from degrees to radians
    
    return position, rotation_zyx

def fill_zeros(frame_count, length=8):
    """Formats the frame count with leading zeros."""
    return str(frame_count).zfill(length)

# def fill_zeros(frame_count, length=8):
#     zeros = '0' * (length - len(str(frame_count)))
#     return f'{zeros}{frame_count}'

def create_water_object(mesh_data: bmesh, object_name: str, collection: str, visible_keyframe=-1, as_pointcloud=False, subdevide_and_edge_crease=False):
    if not BLENDER_AVAILABLE: return
    """Creates a Blender object with the given mesh data and assigns the 'Water' material."""
    # Create a new mesh and object
    bm = mesh_data.copy()
    bm.verts.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
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

    # Add the nodegroup that converts the mesh to a pointcloud with the water material in blender
    if 'mesh_2_pointcloud' in bpy.data.node_groups and as_pointcloud:
        modifier = obj.modifiers.new(name='Mesh to PointCloud', type='NODES')
        modifier.node_group = bpy.data.node_groups['mesh_2_pointcloud']

    # Smooth the surface
    if subdevide_and_edge_crease:
        rx = context.grid_resolution[0]
        ry = context.grid_resolution[1]
        non_surface_faces = 2*(rx+ry)+1

        crease_layer = bm.edges.layers.float.new("crease_edge")  ## From blender 4.2 and newer

        for face_index, face in enumerate(bm.faces[-non_surface_faces:-1]):
            corner_face = False
            if face_index in {0, rx, rx+ry, 2*rx+ry}:
                corner_face = True

            for edge_index, edge in enumerate(face.edges):
                if edge_index == 0 and not corner_face:
                    continue
                if edge_index in {1, 2}: 
                    continue
                edge[crease_layer] = 1.0

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

    
    bm.to_mesh(mesh)
    bm.free()

def pose_floating(position, rotation_zyx, keyframe = -1):
    obj = bpy.data.objects.get("floating")
    wrf = bpy.data.objects.get("floating_wireframe")
    if obj is None and wrf is None:
        print("Object 'floating' not found in 'Floating' collection")
        return
    
    obj.location = position
    obj.rotation_mode = 'ZYX'
    obj.rotation_euler = rotation_zyx

    wrf.location = position
    wrf.rotation_mode = 'ZYX'
    wrf.rotation_euler = rotation_zyx
    
    if keyframe > -1:
        obj.keyframe_insert(data_path="location", frame=keyframe)
        obj.keyframe_insert(data_path="rotation_euler", frame=keyframe)
        wrf.keyframe_insert(data_path="location", frame=keyframe)
        wrf.keyframe_insert(data_path="rotation_euler", frame=keyframe)

if __name__ == "__main__":
    experiment = 'examples/floating'

    context = Context()
    context.read_control(f'{root}/data/{experiment}/control.txt')
    context.read_ctrl(f'{root}/data/{experiment}/ctrl.txt')

    # fps = 24

    start_frame = 0
    end_frame = 721  # TODO: Set to 1 -> then setup blender to nice render settings on that one sample -> then change this to fit your simulation 721
    batch_size = 4  # Number of frames loaded at a time

    filepath = f"data/{experiment}/REEF3D_NHFLOW_VTU" # TODO: change this
    file_name_skeleton_0 = "REEF3D-NHFLOW-"
    file_name_skeleton_1 = ".pvtu"

    wave_collection = 'Waves'
    frames_loaded = start_frame-1
    batch = 0
    for batch_start in range(start_frame, end_frame, batch_size):
        batch_end = min(batch_start + batch_size, end_frame)  # Ensure last batch is handled properly
        batch += 1

        # Clear previous objects before loading new batch
        delete_from_collection(collection_name=wave_collection)

        # Load the batch of frames
        print(f"\n\nBatch {batch}")
        for frame in range(batch_start, batch_end):
            filepath_pvtu = f"{root}/{filepath}/{file_name_skeleton_0}{fill_zeros(frame)}{file_name_skeleton_1}"
            mesh = extract_NHFLOW_mesh_from_pvtu(
                filepath=filepath_pvtu, 
                context=context, 
                surface_only=False,
                with_attributes=False,
                with_faces=True,
            )
            create_water_object(
                mesh_data=mesh, 
                object_name=f'wave_{fill_zeros(frame, len(str(end_frame)))}', 
                collection=wave_collection,
                visible_keyframe=frame,
                as_pointcloud=False,
                subdevide_and_edge_crease=True
            )
            frames_loaded += 1

            if context.floating:
                filepath_dat = f"{root}/data/{experiment}/REEF3D_NHFLOW_6DOF/REEF3D_6DOF_position_0.dat"
                position, rotation_zyx = extract_NHFLOW_6DOF(filepath_dat, frame)
                pose_floating(position=position, rotation_zyx=rotation_zyx, keyframe=frame)

            print_progress(frame-batch_start, (batch_end-batch_start))
            
            

        # Update and move the render window to avoid overwriting render settings
        bpy.context.scene.frame_start = batch_start
        bpy.context.scene.frame_end = batch_end - 1
        bpy.context.scene.frame_set(batch_start)

        # Trigger render
        bpy.ops.render.render(animation=True)
        print_progress(frames_loaded-start_frame, end_frame-start_frame)
