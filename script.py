## BLENDER LIBS
import bpy
import bmesh
import numpy as np

# PYTHON CORE_LIBS
import sys
import os
import xml.etree.ElementTree as ET

root = bpy.data.filepath        # The filepath of the .blend file you run the script in
root = os.path.dirname(root)    # The repo you are working in [most likely]
venv = '.venv'

pyhton_env_path = f'{root}\{venv}\Lib\site-packages'
sys.path.append(pyhton_env_path)

# PYTHON SITE-PACKAGES
import vtkmodules.all as vtk
print(vtk.VTK_VERSION)


def delete_from_collection(collection_name):
    """Deletes all objects in the specified collection except those named 'cube' or 'Cube'."""
    if collection_name in bpy.data.collections:
        collection = bpy.data.collections[collection_name]
        for obj in list(collection.objects):
            if obj.name.lower() != 'cube':
                bpy.data.objects.remove(obj, do_unlink=True)

def extract_mesh_from_vtu(filepath, return_mesh=True):
    # Create the VTK reader
    reader = vtk.vtkXMLUnstructuredGridReader()
    reader.SetFileName(filepath)
    reader.Update()

    # Get the mesh output from the reader
    mesh = reader.GetOutput()
    points = mesh.GetPoints()

    # Create a new BMesh object
    bm = bmesh.new()

    # Add vertices to BMesh
    if points:
        for x in range(points.GetNumberOfPoints()):
            a = [0, 0, 0]
            points.GetPoint(x, a)
            bm.verts.new(a)

    # Only return the vertecies if the mesh is not required
    if not return_mesh:
        return bm

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
    end_frame = 2  # 721

    fps = 24
    M10 = 1  # FROM THE REEF3D and DiveMESH Documentation

    file_name_skeleton_0 = "REEF3D-NHFLOW-"
    file_name_skeleton_1 = "-00000001.vtu"
    filepath_vtu = f"{root}/data/RUN1/REEF3D_NHFLOW_VTU/{file_name_skeleton_0}{fill_zeros(start_frame)}{file_name_skeleton_1}"

    collection = 'Waves'
    print("MAIN-111111111111111111111")
    delete_from_collection(collection_name=collection)
    
    print("MAIN-222222222222222222222")
    for frame in range(start_frame, end_frame):
        print("MAIN-LOOOOOOOOOOOOOOOP")
        filepath_vtu = f"{root}/data/RUN1/REEF3D_NHFLOW_VTU/{file_name_skeleton_0}{fill_zeros(frame)}{file_name_skeleton_1}"
        mesh = extract_mesh_from_vtu(filepath_vtu)
        create_water_object(mesh, f'wave_{fill_zeros(frame, 3)}', collection, frame)
