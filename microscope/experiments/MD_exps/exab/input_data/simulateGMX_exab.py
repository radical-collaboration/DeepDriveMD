#!/usr/bin/env python 

# OpenMM imports
from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
#from simtk.openmm.app.gromacstopfile import _defaultGromacsIncludeDir

# ParmEd imports
from parmed import load_file

from sys import stdout, argv

#sys.exit(1)
# Load .gro and .top file from gromacs input. 
#gro = GromacsGroFile('exab_sol_ion.gro')
#top = GromacsTopFile('exab.top', periodicBoxVectors=gro.getPeriodicBoxVectors())
        #includeDir='/home/hm0/Research/FF_study/1bdo/')
print 'Loading GMX files.'
top = load_file('exab.top', xyz='exab.gro')


# Create system with 1.2 nm LJ and Coul, 1.0 nm LJ switch. 
print 'Creating OpenMM system.'
system = top.createSystem(nonbondedMethod=PME, nonbondedCutoff=1.2*nanometer,
 	switchDistance=1.0*nanometer,
	constraints=HBonds, verbose=0)

# Adding pressure coupling, NPT system
system.addForce(MonteCarloAnisotropicBarostat((1, 1, 1)*bar, 300*kelvin, False, False, True))

# Langevin integrator at 300 K, 1 ps^{-1} collision frequency, time step at 2 fs. 
integrator = LangevinIntegrator(300*kelvin, 1/picosecond, 0.002*picoseconds) 
# using CUDA platform, fastest of all 
platform = Platform.getPlatformByName('CUDA')

# pass the simulation parameters (topology, integrators, system) to Simulation function.  
if ("GPU" in argv and argv.index('GPU')+1 < len(argv)): 
    print 'using GPU', argv[argv.index('GPU')+1]
    properties = {'DeviceIndex': argv[argv.index('GPU')+1]}
    simulation = Simulation(top.topology, system, integrator, platform, properties)
else: 
    simulation = Simulation(top.topology, system, integrator, platform)

# initial position of simulation at t=0
simulation.context.setPositions(top.positions)

# energy minimization relaxing the configuration
print 'Minimizing energy.'
simulation.minimizeEnergy()

# simulation outputs pdb and log every 50000 steps and checkpoint every 50000 steps
simulation.reporters.append(DCDReporter('exab.dcd', 50000))
simulation.reporters.append(StateDataReporter('exab.log', 
	50000, step=True, time=True, speed=True, 
	potentialEnergy=True, temperature=True, totalEnergy=True))
simulation.reporters.append(StateDataReporter(stdout,
        50000, step=True, time=True, speed=True,
        potentialEnergy=True, temperature=True, totalEnergy=True))
simulation.reporters.append(CheckpointReporter('checkpnt.chk', 50000))

# necessary to load checkpoint for continuing an interupted simulation
#simulation.loadCheckpoint('state.chk')

# run the simulation for 500,000,000 steps
print 'Hitting the road!'
simulation.step(500000000)
