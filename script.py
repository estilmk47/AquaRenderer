## BLENDER LIBS
BLENDER_AVAILABLE = False
try:
    import bpy
    import bmesh
    BLENDER_AVAILABLE = hasattr(bpy, "context")  # Check if bpy has actual Blender functionality
except ImportError:
    BLENDER_AVAILABLE = False

import numpy as np

# PYTHON CORE_LIBS
import sys
import os
import xml.etree.ElementTree as ET

root = '.'

if BLENDER_AVAILABLE:
    root = bpy.data.filepath        # The filepath of the .blend file you run the script in
    root = os.path.dirname(root)    # The repo you are working in [most likely]
    venv = 'venv'
    pyhton_env_path = f'{root}\{venv}\Lib\site-packages'
    sys.path.append(pyhton_env_path)

# PYTHON SITE-PACKAGES
import vtkmodules.all as vtk
print(f"vtk module version {vtk.VTK_VERSION}")

def delete_from_collection(collection_name):
    if not BLENDER_AVAILABLE: return
    """Deletes all objects in the specified collection except those named 'cube' or 'Cube'."""
    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]
        for obj in list(collection.objects):
            if obj.name.lower() != 'cube':
                bpy.data.objects.remove(obj, do_unlink=True)

def extract_NHFLOW_mesh_from_pvtu(filepath, with_attributes=True, return_mesh=True):
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
        num_points = points.GetNumberOfPoints()  # Get the number of vertices
        np_verts = np.zeros((num_points, 3), dtype=np.float32)  # Allocate space

    # Add vertices to BMesh
    if points:
        for x in range(points.GetNumberOfPoints()):
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
                np_verts[x, :] = np.array(a, dtype=np.float32)  # Assign to NumPy array

    # Only return the vertecies if the mesh is not required
    if not return_mesh and BLENDER_AVAILABLE:
        return bm      
    if not BLENDER_AVAILABLE:
        return np_verts 
    
    bm.verts.ensure_lookup_table()
    # Add faces to BMesh
    for i in range(mesh.GetNumberOfCells()):
        face = mesh.GetCell(i)
        num_points = face.GetNumberOfPoints()

        if num_points < 3:
            continue  # Skip invalid face (e.g., if it's not a polygon)

        # Get the point indices for this face
        face_points_ids = []
        for p in range(num_points):
            face_points_ids.append(face.GetPointId(p))

        # Create the face in BMesh using the corresponding vertices
        face_verts = [bm.verts[vid] for vid in face_points_ids]
        bm.faces.new(face_verts)
    return bm

def fill_zeros(frame_count, length=8):
    """Formats the frame count with leading zeros."""
    return str(frame_count).zfill(length)

# def fill_zeros(frame_count, length=8):
#     zeros = '0' * (length - len(str(frame_count)))
#     return f'{zeros}{frame_count}'

def create_water_object(mesh_data: bmesh, object_name: str, collection: str, visible_keyframe=-1):
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
    
    # Convert bmesh to mesh data
    bm = mesh_data.copy()
    bm.to_mesh(mesh)
    bm.free()
    
    # Assign 'water' material if it exists
    if 'water' in bpy.data.materials:
        obj.data.materials.append(bpy.data.materials['water'])
    
    # Handle visibility keyframes
    if visible_keyframe > -1:
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_render", frame=-1)
        obj.hide_render = False
        obj.keyframe_insert(data_path="hide_render", frame=visible_keyframe)
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_render", frame=visible_keyframe + 1)

if __name__ == "__main__":
    start_frame = 0
    end_frame = 2  # TODO: change this to fit your simulation 721

    fps = 24
    # M10 = 1  # FROM THE REEF3D and DiveMESH Documentation

    filepath = "data/examples/single_split_experiment/REEF3D_NHFLOW_VTU" # TODO: change this
    file_name_skeleton_0 = "REEF3D-NHFLOW-"
    file_name_skeleton_1 = ".pvtu"

    collection = 'Waves'
    delete_from_collection(collection_name=collection)
    
    for frame in range(start_frame, end_frame):
        filepath_pvtu = f"{root}/{filepath}/{file_name_skeleton_0}{fill_zeros(frame)}{file_name_skeleton_1}"
        mesh = extract_NHFLOW_mesh_from_pvtu(filepath_pvtu, with_attributes=False, return_mesh=False)
        create_water_object(mesh, f'wave_{fill_zeros(frame, 3)}', collection, frame)

        if not BLENDER_AVAILABLE:
            print(mesh)
