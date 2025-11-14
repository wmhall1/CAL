"""
Tools for reading and storing Athena data from vtk and 1d files. Also contains
routines for mapping between periodic points, computing turbulent velocity,
power spectrums and autocorrelation functions (ACF).

Author: David Rea
mapping routines adapted from Jake Simon's IDL code
Debanjan Sengupta told me how to compute power spectra
DataX methods inspired by pyridoxine by Rixin Li

class DataHST
class Data1D
class DataVTK
class DataPHST
class DataLIS
class SpaceTimeData
class TimeAverageVTK

def magnitude
def gmean
def save
def load
def turbulent
def vorticity

def shear_map
def shear_map_2nd_order
def map_k
def power_spectrum
def shell_average3d
def autocorr
"""

import sys
import os
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.ndimage import rotate
from glob import glob
import matplotlib.pyplot as plt
from time import time

try:
    import cPickle as pickle
except ModuleNotFoundError:
    import pickle

#------------------------------------------------------------------------------#
# Objects for containing Athena data
#------------------------------------------------------------------------------#

class DataHST:
    """
    contains data from Athena history (hst) file
    volume averaged quantities over time
    """
# consider using open() instead of pandas.read_csv()
    
    version = '2.1'
    
    def __init__(self, filename, silent=False):
        
        if filename.split('.')[-1] != 'hst':
            raise NameError("file must be an Athena history dump (.hst)")
                
        with open(filename, 'r') as f:
    
            f.readline()
            line = f.readline()
            self.names = [s.strip() for s in line.replace('=','  ').split('  ') if s][2::2]

            f.close()
        
        df = pd.read_csv(filename,
                         header = None,
                         skiprows = 3,
                         delim_whitespace = True
                         )
        df.set_axis(self.names, axis='columns')
        
        # organize the data
        
        self.data = {}
        for i, name in enumerate(self.names):
            
            self.data[name] = np.array(df.get(name), float)
            
        if not silent:
            print("time averaged quantities:", self.names)
        

class DataPHST:
    """
    contains data from Athena particle history (phst) file
    volume averaged quantities over time
    """
# consider using open() instead of pandas.read_csv()
    
    version = '1.1'
    
    def __init__(self, filename, silent=False):
        
        assert filename.split('.')[-1] == 'phst', "file must be an Athena particle history dump (.phst)"
        
        with open(filename, 'r') as f:
            
            # Global scalars
            f.readline()
            line = f.readline()
            glo_names = [s.strip() for s in line.replace('=','  ').split('  ') if s][2::2]
            
            # Particle dependent scalars
            f.readline()
            line = f.readline()
            par_names = [s.strip() for s in line.replace('=','  ').split('  ') if s][2::2]
            
            self.names = glo_names + par_names

            f.close()
        
        df = pd.read_csv(filename,
                         header = None,
                         skiprows = 5,
                         )
        
        # need some extra work due to the horrendous nature of phst files
        
        df = df[0].str.split(expand=True)
        glo = df[0::2].dropna(axis='columns')
        par = df[1::2].dropna(axis='columns')
        glo.set_axis(glo_names, axis='columns')
        par.set_axis(par_names, axis='columns')
        
        # organize the data
        
        self.data = {}

        for i, name in enumerate(glo_names):
            self.data[name] = np.array(glo.get(name), float)
            
        for i, name in enumerate(par_names):
            self.data[name] = np.array(par.get(name), float)
            
        if not silent:
            print("  Global quantities:", glo_names)
            print("Particle quantities:", par_names)


class Data1D:
    """
    contains data from Athena 1d files
    horizontally (xy) averaged quantities
    """
# consider using open() instead of pandas.read_csv()

    version = '2.3'
        
    def __init__(self, filename, silent=False):
        
        if '1d' not in filename.split('.')[-1]:
            raise NameError("file must contain horizontally averaged variables (.1d)")
                
        with open(filename, 'r') as f:
    
            self.names = f.readline().split()[1:]
        
        df = pd.read_csv(filename,
                         header = None,
                         skiprows = 1,
                         delim_whitespace = True
                         )
        df.set_axis(self.names, axis='columns')
        
        # organize the data
        
        self.data = {}
        for i, name in enumerate(self.names):
            
            self.data[name] = np.array(df.get(name), float)
            
        if not silent:
            print("horizontally averaged quantities:\n", self.names)
            
            
class SpaceTimeData:
    """
    creates spacetime data for quantities from a series of Athena 1d files
    
    path : specify the directory containing the series of 1d files
    dt   : timestep between 1d file outputs
    """
    
    version = '1.1'
        
    def __init__(self, path, dt=0.2*np.pi, silent=False):
        
        if path[-1] not in ['\\', '/']:
            path += '/'
        
        if not os.path.exists(path):
            raise NameError("path does not exist!")
            
        self.data = {}
        
        self.file_count = len([name for name in os.listdir(path) if os.path.isfile(path+name) and '.1d' in name])
        if not self.file_count > 0:
            raise ValueError(f"No 1d files were found in {path}")
            
        print(path)
        
        for i, f in enumerate(sorted(glob(path+"*.1d"))):
            
            if i == 0: # first loop
                
                do = Data1D(f)
                
                self.progress(0, self.file_count)
                
                self.names = do.names
                
                for name in self.names:
                    self.data[name] = do.data[name]
                    
                self.z = do.data['x3']
                self.t = np.arange(self.file_count)*dt
                self.t_orbit = self.t/(2*np.pi)
                
                self.Z, self.T = np.meshgrid(self.z, self.t)
                self.T_orbit = self.T/(2*np.pi)
                    
                continue # end first loop
                    
            self.progress(i+1, self.file_count)
            
            do = Data1D(f, silent=True)

            for name in self.names:
                self.data[name] = np.append(self.data[name], do.data[name])
                
        # reshape data
        for name in self.names:
            self.data[name] = self.data[name].reshape(self.Z.shape)
            
    def progress(self, it, total):
    
        fill = "█"
        length = 50

        fraction = it/total
        filledLength = int(length*fraction)
        
        bar = fill*filledLength + "-"*(length - filledLength)    

        print(f"\rProgress |{bar}| {100*fraction:.1f}% Complete", end='\r')
        

class DataVTK:
    """
    Contains data and metadata from Athena VTK files. Borrows heavily from Rixin Li's pyridoxine module
    """
    
    version = '3.3'
    
    def __init__(self, filename, silent=True):
        """
        : filename : the VTK file
        : silent   : print basic info or not
        """
        
        if not os.path.exists(filename):
            raise NameError(f"File {filename} does not exist")
        
        with open(filename, 'rb') as f:
            
            eof = f.seek(0,2) # mark the end of the file
            f.seek(0,0)       # and go back to the beginning
                        
            # read the header metadata
            
            self._metadata(f)
            
            self.box_size = self.dx * self.Nx
            self._set_cell_centers()
            
            # now handle the scalar and vector data
            
            self.data    = {} # data to be in {'name': np array} dictionary
            self.svtypes = [] # scalar or vector
            self.names   = [] # name of quantity
            self.dtypes  = [] # data type
            
            shape3d = np.hstack([np.flipud(self.Nx[:self.dim]), 3])
            
            while f.tell() != eof:
                
                line = f.readline().decode('utf-8').split()
                if line == []:
                    line = f.readline().decode('utf-8').split()
                
                self.svtypes.append(line[0])
                self.names.append(line[1])
                self.dtypes.append(line[2])
                
                if line[0] == "SCALARS":
                    f.readline() # skip line
                    
                    data = np.fromfile(f, np.dtype('f'), self.size)
                    self.data[line[1]] = data.byteswap().reshape(np.flipud(self.Nx[:self.dim]))
                
                if line[0] == "VECTORS":
                    
                    data = np.fromfile(f, np.dtype('f'), 3*self.size) # 3D vector fields even for 2D simulations
                    self.data[line[1]] = data.byteswap().reshape(shape3d)
            
            f.close()
                        
        if not silent:
            print("Scalars:", [name for i, name in enumerate(self.names) if self.svtypes[i]=="SCALARS"])
            print("Vectors:", [name for i, name in enumerate(self.names) if self.svtypes[i]=="VECTORS"])
    
    @property
    def tn(self, omega=1.0):
        """
        Computes the nearest periodic point to current time 't'
        tn = n*Ly/(q*omega*Lx)
        assumes q = 1.5 (Keplerian disk)
        """
        
        Lx, Ly, _ = self.box_size

        def func(n, t):
            tn = n*Ly/(1.5*omega*Lx)
            return np.abs(tn - t)

        sol = minimize(func, 0, args=self.t)

        n = round(*sol.x)

        return func(n, 0)
    
    def _metadata(self, f):
        
        f.readline() # skip line
            
        line = f.readline().decode('utf-8')
        self.t      = float(line[line.find("time")+6:line.find(", level")])
        self.level  = int(line[line.find("level")+7:line.find(", domain")])
        self.domain = int(line[line.find("domain")+8:])

        line = f.readline().decode('utf-8')[:-1]
        assert line == "BINARY", f"VTK file does not contain binary data, contains {line}"

        line = f.readline().decode('utf-8')[:-1]
        assert line == "DATASET STRUCTURED_POINTS", f"{line} is not supported"

        line = f.readline().decode('utf-8')
        self.Nx = np.array(line.split()[1:], int) - 1
        if self.Nx[2] == 0:
            self.dim = 2
        else:
            self.dim = 3

        line = f.readline().decode('utf-8')
        assert line[:6] == "ORIGIN", f"no ORIGIN, {line}"
        self.origin = np.array(line.split()[1:], float)

        line = f.readline().decode('utf-8')
        assert line[:7] == "SPACING", f"no SPACING, {line}"
        self.dx = np.array(line.split()[1:], float)

        line = f.readline().decode('utf-8')
        assert line[:9] == "CELL_DATA", f"no CELL_DATA, {line}"
        self.size = int(line[10:])
            
    def _set_cell_centers(self):
        # need to handle cases when 2D data is xz or yz

        self.ccx = np.linspace(self.origin[0] + 0.5*self.dx[0],
                               self.origin[0] + self.box_size[0] - 0.5*self.dx[0],
                               self.Nx[0])
        self.ccy = np.linspace(self.origin[1] + 0.5*self.dx[1],
                               self.origin[1] + self.box_size[1] - 0.5*self.dx[1],
                               self.Nx[1])
        self.ccz = np.linspace(self.origin[2] + 0.5*self.dx[2],
                               self.origin[2] + self.box_size[2] - 0.5*self.dx[2],
                               self.Nx[2])


class TimeAverageVTK:
    
    version = '0.1'
    
    notes = 'currently a bug in the progress bar when step != 1. Consider deleting tempvtk attribute after use'
    
    def __init__(self, path, **kwargs):
        
        if path[-1] not in ['\\', '/']:
            path += '/'
            
        if not os.path.exists(path):
            raise NameError("path does not exist!")
            
        self.tstart = 0.0
        self.tend   = None
        
        self.start_int = 0
        self.step = 1
            
        self.file_count = len([name for name in os.listdir(path) if os.path.isfile(path+name) and '.vtk' in name][self.start_int::self.step])
        if not self.file_count > 0:
            raise ValueError(f"No vtk files were found in {path}")
                        
        for param, value in kwargs.items():
            
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{param}'")
                
        # maybe a way to make this cleaner
        if type(self.start_int) != int:
            raise ValueError(f"start_int must be type 'int', but was instead type '{type(self.start_int)}'")
        if type(self.step) != int:
            raise ValueError(f"start_int must be type 'int', but was instead type '{type(self.step)}'")
                
        print(path)
        
        count = 0
        self.deltav = 0
        
        for i, f in enumerate(sorted(glob(path+"*.vtk"))[self.start_int::self.step]):
            
            if i == 0: # first loop
                
                self.tempvtk = DataVTK(f, silent=False)
                
                self.x = self.tempvtk.ccx
                self.y = self.tempvtk.ccy
                self.z = self.tempvtk.ccz

                if self.tempvtk.t >= self.tstart:
                    
                    count += 1
                    self.deltav += magnitude(turbulent(self.tempvtk.data['velocity'])).mean(axis=(1,2))
                    
                self.progress(0, self.file_count)
                
                continue
                
            self.tempvtk = DataVTK(f, silent=True)
                        
            if self.tempvtk.t >= self.tstart:
                    
                count += 1
                self.deltav += magnitude(turbulent(self.tempvtk.data['velocity'])).mean(axis=(1,2))
            
            self.progress(i+1, self.file_count)
                        
        self.deltav /= count
        
        # delattr(self, 'tempvtk')
    
    def progress(self, it, total):
    
        fill = "█"
        length = 50

        fraction = it/total
        filledLength = int(length*fraction)
        
        bar = fill*filledLength + "-"*(length - filledLength)    

        print(f"\rProgress |{bar}| {100*fraction:.1f}% Complete", end='\r')
        
        
class DataLIS:
    """
    contains data from Athena LIS binary files
    position, velocity, radius, mass for Np number of particles
    
    no metadata in LIS files - must verify what header information there is
    """
    
    version = '1.1'
    
    def __init__(self, filename, silent=True):
        
        assert os.path.isfile(filename)
        
        with open(filename, 'rb') as f:
    
            # read the "header" info (there is no header)
    
            self.coord_lim = np.fromfile(f, np.float32, 12)     # need to verify each component
            mystery = np.fromfile(f, np.float32, 2)             # unsure of what these values are
            self.time, self.dt  = np.fromfile(f, np.float32, 2) # time and dt do not seem correct
            self.Np,        = np.fromfile(f, np.int64, 1)
            
            # read the particle info
            
            f.seek(72,0) # ensures that the entire header is skipped for redundancy

            par_info = np.fromfile(f, np.float32)

            self.x1     = par_info[0::11] # x position
            self.x2     = par_info[1::11] # y position
            self.x3     = par_info[2::11] # z position
            self.v1     = par_info[3::11] # x velocity
            self.v2     = par_info[4::11] # x velocity
            self.v3     = par_info[5::11] # x velocity
            self.radius = par_info[6::11] # particle radius
            self.mass   = par_info[7::11] # particle mass

            # don't care about these
            # self.pid    = par_info[8::10] # particle id ??
            # self.cpuid  = par_info[9::10] # cpu id ??

            f.close()
            
        if not silent:
            print(coord_lim, time, dt, Np)
            

#------------------------------------------------------------------------------#
# Functions to return list or generator object of data
#------------------------------------------------------------------------------#

def read_hst(path=os.getcwd(), silent=False):
    # not strictly need, as there will only ever be one hst dump, but will
    # automatically search in the id0/ directory for convenience
    
    try: # prefer phst file in id0/ if exists
        filename, = glob(path+"/id0/*.hst")
        
    except ValueError:
        try: # then try phst file in path
            filename, = glob(path+"/*.hst")
            
        except ValueError as e: # if file is still not found
            print(e)
    
    return DataHST(filename, silent=silent)

def read_phst(path=os.getcwd(), silent=False):
    # not strictly need, as there will only ever be one phst dump, but will
    # automatically search in the id0/ directory for convenience
    #
    # mostly identical to read_hst, but seperate functions are retained in
    # case DataHST and DataPHST diverge in the future
    
    try: # prefer phst file in id0/ if exists
        filename, = glob(path+"/id0/*.phst")
        
    except ValueError:
        try: # then try phst file in path
            filename, = glob(path+"/*.phst")
            
        except ValueError as e: # if file is still not found
            print(e)
    
    return DataPHST(filename, silent=silent)

def read_1d(path, start=0, stop=None, thin=1, generator=False, silent=True):
    
    files = sorted(glob(path + "*.1d"))
    files = files[start:stop:thin]
    
    assert len(files) != 0, "no files being read"
    
    if not silent: # print the list of files being read
        print_list = [f.split(".")[-2] for f in files]
        print(*print_list)
        
    #if generator: # yield a generator
    #    for f in files:
    #        yield Data1D(f)
            
    else: # return a numpy array, default
        obj_list = np.empty_like(files, dtype=object)
        
        for i, f in enumerate(files):
            obj_list[i] = Data1D(f)
        
        return obj_list

def read_vtk(path, start=0, stop=None, thin=1, generator=True, silent=True):

    files = sorted(glob(path + "*.vtk"))
    files = files[start:stop:thin]
    
    assert len(files) != 0, "no files being read"
    
    if not silent: # print the list of files being read
        print_list = [f.split(".")[-2] for f in files]
        print(*print_list)
        
    if generator: # yield a generator, default
        for f in files:
            yield DataVTK(f, wanted=None, xyz_order=None, silent=True, pad=True)
    
    else: # return a numpy array
        obj_list = np.empty_like(files, dtype=object)
        
        for i, f in enumerate(files):
            obj_list[i] = DataVTK(f, wanted=None, xyz_order=None, silent=True, pad=True)
            
        return obj_list
    
def read_lis(path, start=0, stop=None, thin=1, generator=True, silent=True):
    
    pass

#------------------------------------------------------------------------------#
# Compute various statistics from a series of vtk files
# and save to a npy binary file
#------------------------------------------------------------------------------#

def compute_turbulent_velocity(path, savename, component=None, axis='z', silent=False):
    """
    computes the turbulent velocity spacetime matrix from a series of VTK files
    located in 'path'.
    
    param component: 'x', 'y', 'z' component of turbulent velocity, or None
                     (default) for the magnitude
    param axis: spatial axis to remain after averaging, e.g., 'z' => horizontal
                average over x and y
    """
    
    comp = {'x': 2,     'y': 1,     'z': 0}
    ax   = {'x': (0,1), 'y': (0,2), 'z': (1,2)}
    
    files = sorted(glob(f"{path}/*.vtk"))
        
    N = DataVTK(files[0]).Nx[abs(comp[component]-2)]
    
    turbvel = np.zeros( (len(files), N) )
    
    for i, f in enumerate(files):
        
        vtk = DataVTK(f)
        
        vt = turbulent(vtk.data['velocity'])
        
        if component is None:
            turbvel[i,:] = magnitude(vt).mean(axis=ax[axis])
            
        else:
            turbvel[i,:] = np.abs(vt[...,comp[component]]).mean(axis=ax[axis])
                       
    np.save(turbvel, f"./{savename}_vt{component}_{axis}.npy")

def compute_energy_spectrum():
    
    pass

def compute_autocorr():
    
    pass
    
#------------------------------------------------------------------------------#
# Module-level functions below here
#------------------------------------------------------------------------------#

def magnitude(x, axis=None):
    'computes the magnitude of x'

    if axis==None:
        x2 = sum( x[...,i]**2 for i in range(x.shape[-1]))
    else:
        x2 = sum( x[...,i]**2 for i in axis )

    return np.sqrt(x2)
    
def gmean(x, axis=None):
    'computes the geometric mean of |x|'

    return np.exp(np.log(np.abs(x)).mean(axis=axis))

def movingaverage(interval, window_size):
    'computes the moving average over some interval, given some window size'
    
    window = np.ones(int(window_size)) / float(window_size)
    
    return np.convolve(interval, window, 'same')
    
def turbulent(velocity):
    """
    computes the gas turbulent velocity.
    See Simon et al. (2013;2015) for more details
    """

    turbvel = velocity[:]
    
    # remove any net radial outflow
    # subtract xy-average of velocity at each z

    turbvel -= turbvel.mean(axis=(1,2))[:,None,None,:]
    
    # remove influence of zonal flows
    # subtract y-average of y-component at each x, z
    # from y-component of vel_red

    turbvel[...,1].mean(axis=1)[:,None,:]

    return turbvel
    
def vorticity(velocity, dx):
    'computes the vorticity of the input velocity field'
    
    u = velocity[...,0]
    v = velocity[...,1]
    w = velocity[...,2]

    Nz, Ny, Nx = u.shape
    dx, dy, dz = dx

    du_dy = ((u[1:-1,2:,1:-1] - u[1:-1,:-2,1:-1])/(2*dy))
    du_dz = ((u[2:,1:-1,1:-1] - u[:-2,1:-1,1:-1])/(2*dz))
    dv_dx = ((v[1:-1,1:-1,2:] - v[1:-1,1:-1,:-2])/(2*dx))
    dv_dz = ((v[2:,1:-1,1:-1] - v[:-2,1:-1,1:-1])/(2*dz))
    dw_dx = ((w[1:-1,1:-1,2:] - w[1:-1,1:-1,:-2])/(2*dx))
    dw_dy = ((w[1:-1,2:,1:-1] - w[1:-1,:-2,1:-1])/(2*dy))

    vorticity = np.zeros((Nz-2, Ny-2, Nx-2, 3), float)
    vorticity[...,0] = dw_dy - dv_dz
    vorticity[...,1] = du_dz - dw_dx
    vorticity[...,2] = dv_dx - du_dy

    return vorticity

def images_to_gif(files, name):
    "converts images matching 'files' into 'name' gif"
    
    frames = [Image.open(image) for image in sorted(glob(f"{files}"))]
    
    name = name.lower()
    if name[-4:] != ".gif": name.append(".gif")
    
    params = {
        'append_images': frames,
        'save_all': True,
        'duration': 100,         # display time for each frame, in ms
        'loop': 1,               # number of times for gif to loop
        'subsampling': 1,        # subsampling (1) reduces the quality?
        'quality': 95,
    }
    
    print("saving animation...")
    tic = time()
    
    anim = frames[0]
    anim.save(f"{name}", format="GIF", **params)
    
    toc=time()
    print(f"saving animation took {int((toc-tic)/60)}m {(toc-tic)%60:.0f}s")

#------------------------------------------------------------------------------#
# Save and Load objects with pickle files
#
# these functions are useful in general for saving objects, but may not be
# more convenient than simply reading the large Athena files
# consider reworking or removing in a future version
#------------------------------------------------------------------------------#

def save(obj, filename):
    
    with open(filename, 'wb') as f:
        
        pickle.dump(obj, f)
        f.close()
        
def load(filename):
    
    with open(filename, 'rb') as f:
        
        obj = pickle.load(f)
        f.close()
        
    return obj

def save_all(lst, filename):

    with open(str(filename), 'wb') as f:

        for obj in lst:
            pickle.dump(obj, f)
        f.close()

def load_all(filename):

    with open(filename, 'rb') as f:

        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break
        f.close()


#------------------------------------------------------------------------------#
# Below this point are routines that are deprecated or no longer used, but are
# kept just in case
#------------------------------------------------------------------------------#


################################################################################
# Power Spectrum mapping and averaging routines
# see Hawley et al. (1995) sec 2.4 for details
################################################################################

def shear_map_2nd_order(obj, data, r=1):
    # map from y --> y' - q \Omega x (t - t_n) (r=1)
    # or reverse (r=-1)
    
    assert r==1 or r==-1, "r must equal 1 or -1"
    
    x = obj.ccx
    dy = obj.dx[1]
    deltat = obj.t - obj.tn
    nx, ny, nz = obj.Nx
    
    # Calculate integer and fractional number of grid zones to shift by
    deltay = obj.q*obj.omega*x*deltat
    shift_int = np.round(deltay/dy).astype(int)
    shift_frac = deltay/dy - shift_int
    
    data_shift = np.zeros_like(data)
    
    # check which index correspond to x1
    ind = np.where(data.shape == nx)[0][0]
    
    if ind == 2:
        for i in range(nx):
            data_shift[...,i] = (1.0-shift_frac[i]) * np.roll(data[...,i], r*shift_int[i],     axis=1) + \
                                     shift_frac[i]  * np.roll(data[...,i], r*(shift_int[i]+1), axis=1)
    elif ind == 0:
        for i in range(nx):
            data_shift[i,...] = (1.0-shift_frac[i]) * np.roll(data[i,...], r*shift_int[i],     axis=1) + \
                                     shift_frac[i]  * np.roll(data[i,...], r*(shift_int[i]+1), axis=1)
        
    else:
        print("None of data.shape == nx, did not map")

    return data_shift

################################################################################
# Auto-correlation routines
################################################################################

def autocorr(obj, x):
    'computes the spatial autocorrelation function of x'
        
    nx, ny, nz = obj.Nx
    N = nz//2
    
    #x_map = shear_map_2nd_order(obj, x)
    x_map = shear_map(obj, x)
    
    fftx = np.fft.fftn(x_map)
    
    ret = np.fft.ifftn( fftx * np.conjugate(fftx) ).real
    
    ret = np.fft.fftshift(ret)
    
    #ret = shear_map_2nd_order(obj, ret, r=-1)
    ret = shear_map(obj, ret, r=-1)
        
    return ret[N,...] / np.sum(x*x)

def line_profile(acf, theta, delta=4/128, show=False):
    
    ny, nx = acf.shape
    x = np.arange(-2, 2, delta)
    y = np.arange(-4, 4, delta)
    X, Y = np.meshgrid(x, y)
    rot = rotate(acf, theta, reshape=False)
    
    rot_major = rot[ny//2:,nx//2 ]
    rot_minor = rot[ny//2 ,nx//2:]
    
    if show:
        plt.figure(figsize=(8,8))
        plt.pcolormesh(X, Y, rot, shading='auto', vmin=np.min(acf))
        plt.colorbar(pad=-0.35)
        plt.vlines(0, -4, 4, color='r', linewidth=0.5)
        plt.hlines(0, -2, 2, color='b', linewidth=0.5)
        plt.text(-1.8, 3.6, "major", c='r', fontsize=14)
        plt.text(-1.8, 3.3, "minor", c='b', fontsize=14)
        plt.axis('scaled')
        plt.show()
        plt.close()
    
    return rot_major, rot_minor
