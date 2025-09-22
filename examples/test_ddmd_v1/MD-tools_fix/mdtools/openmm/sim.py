import random
from pathlib import Path
from typing import Optional, Tuple, Union
import openmm
import openmm.unit as u
import openmm.app as app
import sys

PathLike = Union[str, Path]


def _configure_amber_implicit(
    pdb_file: PathLike,
    top_file: Optional[PathLike],
    dt_ps: float,
    temperature_kelvin: float,
    heat_bath_friction_coef: float,
    platform: "openmm.Platform",
    platform_properties: dict,
) -> Tuple["app.Simulation", Optional["app.PDBFile"]]:

    # Configure system
    if top_file is not None:
        pdb = None
        top = app.AmberPrmtopFile(str(top_file))
        system = top.createSystem(
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * u.nanometer,
            constraints=app.HBonds,
            implicitSolvent=app.OBC1,
        )
    else:
        pdb = app.PDBFile(str(pdb_file))
        top = pdb.topology
        forcefield = app.ForceField("amber99sbildn.xml", "amber99_obc.xml")
        system = forcefield.createSystem(
            top,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=1.0 * u.nanometer,
            constraints=app.HBonds,
        )

    # Configure integrator
    integrator = openmm.LangevinIntegrator(
        temperature_kelvin * u.kelvin,
        heat_bath_friction_coef / u.picosecond,
        dt_ps * u.picosecond,
    )
    integrator.setConstraintTolerance(0.00001)

    sim = app.Simulation(top, system, integrator, platform, platform_properties)

    # Returning the pdb file object for later use to reduce I/O.
    # If a topology file is passed, the pdb variable is None.
    return sim, pdb


def _configure_amber_explicit(
    top_file: PathLike,
    dt_ps: float,
    temperature_kelvin: float,
    heat_bath_friction_coef: float,
    platform: "openmm.Platform",
    platform_properties: dict,
    explicit_barostat: str,
) -> "app.Simulation":

    top = app.AmberPrmtopFile(str(top_file))
    system = top.createSystem(
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * u.nanometer,
        constraints=app.HBonds,
    )

    # Congfigure integrator
    integrator = openmm.LangevinIntegrator(
        temperature_kelvin * u.kelvin,
        heat_bath_friction_coef / u.picosecond,
        dt_ps * u.picosecond,
    )

    if explicit_barostat == "MonteCarloBarostat":
        system.addForce(
            openmm.MonteCarloBarostat(1 * u.bar, temperature_kelvin * u.kelvin)
        )
    elif explicit_barostat == "MonteCarloAnisotropicBarostat":
        system.addForce(
            openmm.MonteCarloAnisotropicBarostat(
                (1, 1, 1) * u.bar, temperature_kelvin * u.kelvin, False, False, True
            )
        )
    else:
        raise ValueError(f"Invalid explicit_barostat option: {explicit_barostat}")

    sim = app.Simulation(
        top.topology, system, integrator, platform, platform_properties
    )

    return sim


def configure_simulation(
    pdb_file: PathLike,
    top_file: Optional[PathLike],
    solvent_type: str,
    gpu_index: int,
    dt_ps: float,
    temperature_kelvin: float,
    heat_bath_friction_coef: float,
    explicit_barostat: str = "MonteCarloBarostat",
    run_minimization: bool = True,
    set_positions: bool = True,
    set_velocities: bool = True,
) -> "app.Simulation":
    """Configure an OpenMM amber simulation.

    Parameters
    ----------
    pdb_file : PathLike
        The PDB file to initialize the positions (and topology if
        `top_file` is not present and the `solvent_type` is `implicit`).
    top_file : Optional[PathLike]
        The topology file to initialize the systems topology.
    solvent_type : str
        Solvent type can be either `implicit` or `explicit`, if `explicit`
        then `top_file` must be present.
    gpu_index : int
        The GPU index to use for the simulation.
    dt_ps : float
        The timestep to use for the simulation.
    temperature_kelvin : float
        The temperature to use for the simulation.
    heat_bath_friction_coef : float
        The heat bath friction coefficient to use for the simulation.
    explicit_barostat : str, optional
        The barostat used for an `explicit` solvent simulation can be either
        "MonteCarloBarostat" by deafult, or "MonteCarloAnisotropicBarostat".
    run_minimization : bool, optional
        Whether or not to run energy minimization, by default True.
    set_positions : bool, optional
        Whether or not to set positions (Loads the PDB file), by default True.
    set_velocities : bool, optional
        Whether or not to set velocities to temperature, by default True.

    Returns
    -------
    app.Simulation
        Configured OpenMM Simulation object.
    """
    # Configure hardware
    try:
        platform = openmm.Platform.getPlatformByName("CUDA")
        platform_properties = {"DeviceIndex": str(gpu_index), "CudaPrecision": "mixed"}
    except Exception:
        try:
            platform = openmm.Platform.getPlatformByName("OpenCL")
            platform_properties = {"DeviceIndex": str(gpu_index)}
        except Exception:
            platform = openmm.Platform.getPlatformByName("CPU")
            platform_properties = {}

    # Select implicit or explicit solvent configuration
    if solvent_type == "implicit":
        sim, pdb = _configure_amber_implicit(
            pdb_file,
            top_file,
            dt_ps,
            temperature_kelvin,
            heat_bath_friction_coef,
            platform,
            platform_properties,
        )
    else:
        assert solvent_type == "explicit"
        assert top_file is not None
        pdb = None
        sim = _configure_amber_explicit(
            top_file,
            dt_ps,
            temperature_kelvin,
            heat_bath_friction_coef,
            platform,
            platform_properties,
            explicit_barostat,
        )

    # Set the positions
    if set_positions:
        if pdb is None:
            pdb = app.PDBFile(str(pdb_file))
        sim.context.setPositions(pdb.getPositions())

    # Minimize energy and equilibrate
    if run_minimization:
        sim.minimizeEnergy()

    # Set velocities to temperature
    if set_velocities:
        sim.context.setVelocitiesToTemperature(
            temperature_kelvin * u.kelvin, random.randint(1, 10000)
        )

    return sim
