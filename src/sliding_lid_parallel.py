import numpy as np
import argparse
import lbm

from mpi4py import MPI
from typing import List
from utils import theoretical_viscosity


def save_mpiio(comm, fn, g_kl):
    """
    Method taken from: https://ilias.uni-freiburg.de/data/unifreiburg/lm_data/lm_2480685/save_mpiio.html
    Write a global two-dimensional array to a single file in the npy format
    using MPI I/O: https://docs.scipy.org/doc/numpy/neps/npy-format.html

    Arrays written with this function can be read with numpy.load.

    Parameters
    ----------
    comm (MPI.Comm):  MPI communicator.
    fn (str): File name
    g_kl (ndarray): Portion of the array on this MPI processes.
        This needs to be a two-dimensional array.
    """
    from numpy.lib.format import dtype_to_descr, magic
    magic_str = magic(1, 0)

    local_nx, local_ny = g_kl.shape
    nx = np.empty_like(local_nx)
    ny = np.empty_like(local_ny)

    commx = comm.Sub((True, False))
    commy = comm.Sub((False, True))
    commx.Allreduce(np.asarray(local_nx), nx)
    commy.Allreduce(np.asarray(local_ny), ny)

    arr_dict_str = str({'descr': dtype_to_descr(g_kl.dtype),
                        'fortran_order': False,
                        'shape': (nx.item(), ny.item())})
    while (len(arr_dict_str) + len(magic_str) + 2) % 16 != 15:
        arr_dict_str += ' '
    arr_dict_str += '\n'
    header_len = len(arr_dict_str) + len(magic_str) + 2

    offsetx = np.zeros_like(local_nx)
    commx.Exscan(np.asarray(ny*local_nx), offsetx)
    offsety = np.zeros_like(local_ny)
    commy.Exscan(np.asarray(local_ny), offsety)

    file = MPI.File.Open(comm, fn, MPI.MODE_CREATE | MPI.MODE_WRONLY)
    if comm.Get_rank() == 0:
        file.Write(magic_str)
        file.Write(np.int16(len(arr_dict_str)))
        file.Write(arr_dict_str.encode('latin-1'))
    mpitype = MPI._typedict[g_kl.dtype.char]
    filetype = mpitype.Create_vector(g_kl.shape[0], g_kl.shape[1], ny)
    filetype.Commit()
    file.Set_view(header_len + (offsety+offsetx)*mpitype.Get_size(),
                  filetype=filetype)
    file.Write_all(g_kl.copy())
    filetype.Free()
    file.Close()


def communicate(proba_density: np.array, cartcomm: MPI.Cartcomm, all_source_dest: List) -> None:
    """
    Communicate ghost cells.

    Args:
        proba_density (np.ndarray): Probability density function of Lattice.
        cartcomm (MPI.Cartcomm): Cartesian communicator
        all_source_dest (list): list containing all source and destinations for a process.

    Returns:
        None.
    """

    # Unpack destinations and sources
    r_source, r_dest, l_source, l_dest, u_source, u_dest, d_source, d_dest = all_source_dest

    # send to up, receive from down
    # intermediate buffer otherwise the value of proba density is not populated correctly
    # this is expected to increase the memory footprint
    buffer = np.ascontiguousarray(proba_density[:, :, 0])
    comm.Sendrecv(np.ascontiguousarray(proba_density[:, :, -2]),
                  u_dest,
                  recvbuf=buffer,
                  source=u_source)
    # Populate proba density based on buffer
    proba_density[:, :, 0] = buffer

    # send to down, receive from up
    buffer = np.ascontiguousarray(proba_density[:, :, -1])
    comm.Sendrecv(np.ascontiguousarray(proba_density[:, :, 1]),
                  d_dest,
                  recvbuf=buffer,
                  source=d_source)
    # Populate proba density based on buffer
    proba_density[:, :, -1] = buffer

    # send to the left, receive from the right
    buffer = np.ascontiguousarray(proba_density[:, -1, :])
    cartcomm.Sendrecv(np.ascontiguousarray(proba_density[:, 1, :]),
                      l_dest,
                      recvbuf=buffer,
                      source=l_source)
    # Populate proba density based on buffer
    proba_density[:, -1, :] = buffer

    # send to right, receive from the left
    buffer = np.ascontiguousarray(proba_density[:, 0, :])
    comm.Sendrecv(np.ascontiguousarray(proba_density[:, -2, :]),
                  r_dest,
                  recvbuf=buffer,
                  source=r_source)
    # Populate proba density based on buffer
    proba_density[:, 0, :] = buffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="sliding_lid_parallel.py",
        description="Parallel Execution of sliding lid experiment",
    )

    parser.add_argument(
        "-b",
        "--benchmarking",
        help="Pass this argument if measuring execution time to avoid saving velocity arrays."
             "Default value False.",
        default=False,
        action="store_true"
    )

    parser.add_argument(
        "-o",
        "--omega",
        help="The collision frequency. Default value is 1.7.",
        type=float,
        default=1.7
    )

    parser.add_argument(
        '-v',
        "--velocity",
        help="The velocity of the moving lid.",
        type=float,
        default=0.1
    )

    parser.add_argument('-g',
                        '--grid_size',
                        nargs='+',
                        help='Space separated list of grid size (dim_0, dim_1). For '
                             'example: -g 50 50',
                        type=int,
                        default=(300, 300)
                        )

    parser.add_argument('-d',
                        '--discretization',
                        nargs='+',
                        help='Discretization of domain to parallel processes.'
                             'For example: -d 2 2',
                        type=int,
                        default=(2, 2)
                        )

    parser.add_argument(
        "-s",
        "--steps",
        help="The simulation steps. Default value is 10000.",
        type=int,
        default=10000,
    )

    # Parse arguments
    args = parser.parse_args()

    # Benchmarking
    benchmarking = args.benchmarking

    # Collision frequency
    omega = args.omega

    # World communicator and MPI statistics
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # Unpack discretization and domain size
    nsub_x, nsub_y = args.discretization
    dim_x, dim_y = args.grid_size

    # Validate that correct discretization is chosen
    err_msg = f"Invalid discretization given. It should be {nsub_x} x {nsub_y} = {size}"
    assert nsub_x * nsub_y == size, err_msg

    # Define lid velocity
    lid_velocity = np.array([args.velocity, 0.])

    # Print information on rank 0
    if rank == 0:
        # Calculate viscosity
        viscosity = theoretical_viscosity(omega)
        # Calculate reynolds number
        reynolds = dim_x * args.velocity / viscosity

        print(f"Parallel execution of sliding lid in {size} processes.")
        print(f"Parallel execution of sliding lid in {size} processes and the following setup:"
              f"\n\tGrid size: \t\t{dim_x} x {dim_y}"
              f"\n\tCollision Frequency: \t{args.omega}"
              f"\n\tViscosity: \t\t{viscosity:.4f}"
              f"\n\tLid velocity: \t\t{args.velocity}"
              f"\n\tReynolds number: \t{reynolds:.2f}"
              f"\n\tSimulation steps: \t{args.steps}"
              )

    # Create a cartesian communicator
    cartcomm = comm.Create_cart(dims=[nsub_x, nsub_y],
                                periods=(False, False),
                                reorder=False
                                )

    # Get the cartesian coordinates of the core
    coords = cartcomm.Get_coords(rank)

    # Define source and destination for each core
    # Send to up, receive from down
    u_source, u_dest = cartcomm.Shift(1, 1)
    # Send to down, receive from up
    d_source, d_dest = cartcomm.Shift(1, -1)
    # Send to right, receive from left
    r_source, r_dest = cartcomm.Shift(0, 1)
    # Send to left, receive from right
    l_source, l_dest = cartcomm.Shift(0, -1)

    # Gather all sources and destinations into a single list
    all_source_dest = [
        r_source, r_dest,
        l_source, l_dest,
        u_source, u_dest,
        d_source, d_dest
    ]

    # Calculate the initial size of the subspaces in each process without ghost cells
    proc_dim_x = dim_x // nsub_x
    proc_dim_y = dim_y // nsub_y

    # Add ghost cells
    # since "periods" is set to False in the cartesian communicator
    # the destinations of domains in the edges are set to -2
    # Temp. variables to help locate the subspace and properly add ghost cells
    left_most = l_dest < 0
    right_most = r_dest < 0
    top_most = u_dest < 0
    bottom_most = d_dest < 0

    # Example for 30 x 30 grid and 3 x 2 discretization scheme to help calculations
    # ghost cells are denoted with *

    # |----------|*    *|----------|*    *|----------|
    # |----------|*    *|----------|*    *|----------|
    # |--rank 1--|*    *|--rank 3--|*    *|--rank 5--|
    # |----------|*    *|----------|*    *|----------|
    # |----------|*    *|----------|*    *|----------|
    # *************    **************    *************
    #
    # *************    **************    *************
    # |----------|*    *|----------|*    *|----------|
    # |----------|*    *|----------|*    *|----------|
    # |--rank 0--|*    *|--rank 2--|*    *|--rank 4--|
    # |----------|*    *|----------|*    *|----------|
    # |----------|*    *|----------|*    *|----------|

    # The following cases require special handling in case of not directly discretized grids
    # Rightmost needs special attention in case the grid is not directly discretized
    # with the chosen discretization scheme, equation derived from the solution of the diffusion
    # equation example given in Learning Module 4: Parallelization
    if right_most:          # ranks 4, 5
        proc_dim_x = dim_x - proc_dim_x * (nsub_x - 1)  # -1 because the rest are equally distributed

    # Topmost subspace needs special treatment in case the grid is not directly discretized
    # with the chosen discretization scheme, equation derived from the solution of the diffusion
    # equation example given in Learning Module 4: Parallelization
    if top_most:            # ranks 1, 3, 5
        proc_dim_y = dim_y - proc_dim_y * (nsub_y - 1)  # -1 because the rest are equally distributed

    # At this point the grid is correctly divided into subspaces
    # Define the active limits here to be used later when the velocities are gathered
    # Note that no ghost cells are added up to this point so slices will start from zero
    x_start, x_end = 0, proc_dim_x
    y_start, y_end = 0, proc_dim_y

    # Add space for ghost cells
    # If it's not the rightmost subspace, add ghost cell on the right
    if not right_most:      # ranks 0, 1, 2, 3
        # Add ghost cell
        proc_dim_x += 1
        # Update active limits to account for this ghost cell
        x_start, x_end = 0, proc_dim_x - 1

    # If it's not the left most subspace, add space for ghost cells on the left
    if not left_most:       # ranks: 2, 3, 4, 5
        # Add ghost cell
        proc_dim_x += 1
        # Update active limits to account for this ghost cell
        x_start, x_end = 1, proc_dim_x + 1

    # If it's not the topmost subspace, add space for ghost cells on the top
    if not top_most:        # ranks 0, 2, 4
        # Add ghost cell
        proc_dim_y += 1
        # Update active limits to account for this ghost cell
        y_start, y_end = 0, proc_dim_y - 1

    # If it's not the bottom most subspace, add space for ghost cells on the bottom
    if not bottom_most:     # ranks 1, 2, 3
        # Add ghost cell
        proc_dim_y += 1
        # Update active limits to account for this ghost cell
        y_start, y_end = 1, proc_dim_y + 1

    print(f'Rank: {rank} is located at coordinates ({coords[0]},{coords[1]})')

    # Initialize density ρ(0) = 1.0 at time t=0
    density = np.ones((proc_dim_x, proc_dim_y), dtype=float)
    # Initialize velocity u(0) = 0.0 at time t=0
    velocity = np.zeros((2, proc_dim_x, proc_dim_y), dtype=float)
    # Initialize proba density to equilibrium
    proba_density = lbm.calculate_equilibrium_distro(density, velocity)

    # Iterate over steps, density and velocity are already calculated for the first iteration
    for step in range(args.steps):

        # Communicate
        communicate(proba_density, cartcomm, all_source_dest)

        # Keep the probability density function pre-streaming
        pre_stream_proba_density = proba_density.copy()

        # Streaming
        lbm.streaming(proba_density)

        # Apply boundary conditions on the bottom rigid wall
        lbm.rigid_wall(proba_density, pre_stream_proba_density, "lower")

        # Apply boundary condition on the top moving wall
        lbm.moving_wall(proba_density, pre_stream_proba_density, lid_velocity, density, "upper")

        # Apply boundary conditions on the left rigid wall
        lbm.rigid_wall(proba_density, pre_stream_proba_density, "left")

        # Apply boundary conditions on the right rigid wall
        lbm.rigid_wall(proba_density, pre_stream_proba_density, "right")

        # Calculate density and velocity
        density = lbm.calculate_density(proba_density)
        velocity = lbm.calculate_velocity(proba_density)

        # Perform collision and relaxation
        lbm.collision_relaxation(proba_density, velocity, density, omega=omega)

    # If not measuring execution time, save velocity arrays to be visualized later
    if not benchmarking:
        save_mpiio(cartcomm, 'ux.npy', velocity[0, x_start:x_end, y_start:y_end])
        save_mpiio(cartcomm, 'uy.npy', velocity[1, x_start:x_end, y_start:y_end])
