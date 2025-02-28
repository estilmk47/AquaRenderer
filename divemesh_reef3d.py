from custom_blender_core import *

class Context:
    def __init__(self):
        self._DIVEMESH_read = False
        self._REEF3D_read = False
        self._primitive_S_cmds = [10, 11, 12, 32, 33] # TODO
        self._grid_resolution = [] # B 2
        self._domain_dimentions = [] # B 10
        self._static_primitives_cmds = [] # S 10, 11, 12, 32, 33, 
        self._mpi_cores = 1
        self._floating = False

    @property
    def grid_resolution(self):
        return self._grid_resolution
    
    @property
    def domain_dimentions(self):
        return self._domain_dimentions
    
    @property
    def mpi_cores(self):
        return self._mpi_cores
    
    @property
    def floating(self):
        return self._floating

    def read_ctrl(self, filepath: str):
        if not filepath.endswith("ctrl.txt"):
            raise ValueError("File must be named ctrl.txt")
        
        with open(filepath, 'r') as file:
            for line in file:
                parts = line.split()
                if parts:
                    try:
                        cmd_class = parts[0]
                        cmd_type = int(parts[1])
                        cmd_variables = [parts[i] for i in range(2, len(parts))]

                        if cmd_class == "M": # MPI
                            if cmd_type == 10:
                                self._mpi_cores = int(cmd_variables[0])
                            # elif cmd_type == 10:
                            #     pass
                            else:
                                raise
                        elif cmd_class == "X": # 6DOF
                            if cmd_type == 180:
                                self._floating = True
                    except:
                        print(f"Unable to parse line: {line}")

    def read_control(self, filepath: str):
        if not filepath.endswith("control.txt"):
            raise ValueError("File must be named control.txt")
        
        with open(filepath, 'r') as file:
            for line in file:
                parts = line.split()
                if parts:
                    try:
                        cmd_class = parts[0]
                        cmd_type = int(parts[1])
                        cmd_variables = [parts[i] for i in range(2, len(parts))]

                        if cmd_class == "B": # Boundary
                            if cmd_type == 2:
                                self._grid_resolution = [int(cmd_variables[i]) for i in range(3)]
                            elif cmd_type == 10:
                                self._domain_dimentions = [float(cmd_variables[i]) for i in range(6)]
                            else:
                                raise
                        if cmd_class == "C": # Channels
                            pass
                        if cmd_class == "C": # Channels
                            pass
                        if cmd_class == "D": # Data Interpolation
                            pass
                        if cmd_class == "G": # Geodat
                            pass
                        if cmd_class == "H": # Hydrodynamic Coupling
                            pass
                        if cmd_class == "S": # Solid
                            if cmd_type in self._primitive_S_cmds:
                                self._static_primitives_cmds.append(parts)
                            #  elif cmd_type == 
                        if cmd_class == "T": # Topo
                            pass
                    except:
                        print(f"Unable to parse line: {line}")
        self._DIVEMESH_read = True

    # def get_sigma_grid_shell(self):
    #     if not self._DIVEMESH_read or not BLENDER_AVAILABLE: 
    #         return None

    #     bm = bmesh.new()
    #     x_min, x_max, y_min, y_max, z_min, z_max = self._domain_dimentions
    #     x_res, y_res, z_res = self._grid_resolution
    #     bmesh.ops.create_cube(bm, size=1.0)
    #     for vert in bm.verts:
    #         vert.co.x = x_min + (x_max - x_min) * (vert.co.x + 0.5)
    #         vert.co.y = y_min + (y_max - y_min) * (vert.co.y + 0.5)
    #         vert.co.z = z_min + (z_max - z_min) * (vert.co.z + 0.5)
    #     bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=z_res, use_grid_fill=True)
    #     bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=x_res, use_grid_fill=True)
    #     bmesh.ops.subdivide_edges(bm, edges=bm.edges, cuts=y_res, use_grid_fill=True)

    #     return bm
    

    def get_static_primitives(self):
        bmesh_stack = []
        for i in range(len(self._static_primitives_cmds)):
            if self._static_primitives_cmds[i][0] != "S": continue

            # What does the numbers mean?? -> look at DiveMESH documentatoion [DIVEMesh-UserGuide.pdf]
            if self._static_primitives_cmds[i][1] == 10: # RECTANGLE (x_start, x_end, y_start, y_end, z_start, z_end)
                pass
            elif self._static_primitives_cmds[i][1] == 11: # RECTANGLE ARRAY (x_origin, y_origin, z_origin, length_ni, gap_ni, length_nj, gap_nj, length_nk, gap_nk)
                pass
            elif self._static_primitives_cmds[i][1] == 12: # BEAM
                pass

            elif self._static_primitives_cmds[i][1] == 32: # CYLINDER IN X DIRECTION ()
                pass
            elif self._static_primitives_cmds[i][1] == 33: # CYLINDER IN Y DIRECTION ()
                pass
            elif self._static_primitives_cmds[i][1] == 11: # CYLINDER [FLEXIBLE] ()
                pass
        return bmesh_stack



if __name__ == '__main__':
    context = Context()
    context.read_control(f'./data/examples/single_split_experiment/control.txt')
    context.read_ctrl(f'./data/examples/single_split_experiment/ctrl.txt')

    print(context.domain_dimentions)
    print(context.grid_resolution)
    print(context.get_sigma_grid_shell())