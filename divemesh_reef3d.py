from custom_blender_core import *

def prime_factorize(number):
    primes = []
    number = int(number)

    factor = 2
    while number > 1:
        # Finding a prime factor
        if number%factor == 0:
            number /= factor
            primes.append(factor)
            factor = 2
            continue

        # The number remaining is a prime
        if factor > math.sqrt(number):
            primes.append(number)
            break
        factor += 1

    return primes
        

class Context:
    def __init__(self):
        self.__DIVEMESH_read = False
        self.__REEF3D_read = False
        self.__primitive_S_cmds = [10, 11, 12, 32, 33] # TODO
        self.__grid_resolution = [] # B 2
        self.__domain_dimentions = [] # B 10
        self.__static_primitives_cmds = [] # S 10, 11, 12, 32, 33, 
        self.__mpi_cores = 1
        self.__floating = False

    @property
    def grid_resolution(self):
        return self.__grid_resolution
    
    @property
    def domain_dimentions(self):
        return self.__domain_dimentions
    
    @property
    def mpi_cores(self):
        return self.__mpi_cores
    
    @property
    def partition_dimentions(self):
        dimentions = [1, 1]
        factors = prime_factorize(self.__mpi_cores)
        for i, fac in enumerate(factors[::-1]):
            dimentions[i%2] *= fac
        return dimentions
    
    @property
    def floating(self):
        return self.__floating

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
                                self.__mpi_cores = int(cmd_variables[0])
                            # elif cmd_type == 10:
                            #     pass
                            else:
                                raise
                        elif cmd_class == "X": # 6DOF
                            if cmd_type == 180:
                                self.__floating = True
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
                                self.__grid_resolution = [int(cmd_variables[i]) for i in range(3)]
                            elif cmd_type == 10:
                                self.__domain_dimentions = [float(cmd_variables[i]) for i in range(6)]
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
                            if cmd_type in self.__primitive_S_cmds:
                                self.__static_primitives_cmds.append(parts)
                            #  elif cmd_type == 
                        if cmd_class == "T": # Topo
                            pass
                    except:
                        print(f"Unable to parse line: {line}")
        self.__DIVEMESH_read = True

    # def get_sigma_grid_shell(self):
    #     if not self.__DIVEMESH_read or not BLENDER_AVAILABLE: 
    #         return None

    #     bm = bmesh.new()
    #     x_min, x_max, y_min, y_max, z_min, z_max = self.__domain_dimentions
    #     x_res, y_res, z_res = self.__grid_resolution
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
        for i in range(len(self.__static_primitives_cmds)):
            if self.__static_primitives_cmds[i][0] != "S": continue

            # What does the numbers mean?? -> look at DiveMESH documentatoion [DIVEMesh-UserGuide.pdf]
            if self.__static_primitives_cmds[i][1] == 10: # RECTANGLE (x_start, x_end, y_start, y_end, z_start, z_end)
                pass
            elif self.__static_primitives_cmds[i][1] == 11: # RECTANGLE ARRAY (x_origin, y_origin, z_origin, length_ni, gap_ni, length_nj, gap_nj, length_nk, gap_nk)
                pass
            elif self.__static_primitives_cmds[i][1] == 12: # BEAM
                pass

            elif self.__static_primitives_cmds[i][1] == 32: # CYLINDER IN X DIRECTION ()
                pass
            elif self.__static_primitives_cmds[i][1] == 33: # CYLINDER IN Y DIRECTION ()
                pass
            elif self.__static_primitives_cmds[i][1] == 11: # CYLINDER [FLEXIBLE] ()
                pass
        return bmesh_stack



if __name__ == '__main__':
    context = Context()
    context.read_control(f'./data/examples/single_split_experiment/control.txt')
    context.read_ctrl(f'./data/examples/single_split_experiment/ctrl.txt')

    print(context.domain_dimentions)
    print(context.grid_resolution)
    print(context.get_sigma_grid_shell())