"""Lattice Boltzmann Method core functions"""
import matplotlib.pyplot as plt
import os

from tqdm import trange
from typing import Optional
from globals import *
from utils import InitMode


def init_proba_density(width: Optional[int] = 15,
                       height: Optional[int] = 10,
                       n_channels: Optional[int] = 9,
                       mode: Optional[int] = InitMode.EQUILIBRIUM,
                       seed: Optional[bool] = False) -> np.array:
    """
    Initialize the probability density function.
    Args:
        width (Optional[int]): The width of the lattice. Default 15.
        height (Optional[int]): The height of the lattice. Default 10.
        n_channels (Optional[int]): Number of channels of velocity discretization. Default 9.
        mode (Optional[int]): Parameter determining the initialization mode. Default EQUILIBRIUM.
        seed (Optional[bool]): Boolean flag for seed in random initialization. Used for testing.
            Default False.
    Returns:
        Probability density function for all channels and every point in the
            lattice.
        """
    # Initialization
    proba_density = np.zeros((n_channels, width, height))

    # Development mode initialized with one on the center position
    if mode == InitMode.DEV:
        proba_density[:, int(width // 2), int(height // 2)] = 1
        return proba_density

    # Equilibrium occupation numbers, initialized with equilibrium weights
    elif mode == InitMode.EQUILIBRIUM:

        for i in range(width):
            for j in range(height):
                proba_density[:, i, j] = weights

    # Initialize with random numbers
    elif mode == InitMode.RAND:
        if seed:
            np.random.seed(42)

        proba_density = np.random.rand(n_channels * width * height). \
            reshape((n_channels, width, height))
    return proba_density


def calculate_density(proba_density: np.array) -> np.array:
    """
    Calculate the mass density at each given point.
    Args:
        proba_density (ndarray): Probability density function of Lattice Boltzmann
            Equation.
    Returns:
        density (ndarray): Mass density at each position of the grid of shape
            (X, Y)
    """
    # Sum over the different channels of probability density function
    return np.sum(proba_density, axis=0)


def calculate_velocity(proba_density: np.array) -> \
        np.array:
    """
    Calculate the velocity field at each given point.
    Args:
        proba_density (ndarray): Probability density function of Lattice Boltzmann
            Equation.
    Returns:
        The velocity field as a numpy array of shape (2, X, Y).
        At each point in real space we get a vector depicting the average velocity
        at the x- and y- direction.
    """

    # First calculate the density at each given point
    density = calculate_density(proba_density)

    # Then calculate velocity field
    return np.einsum("cij,ca->aij", proba_density, velocity_channels) / (
            density + epsilon)


def streaming(proba_density: np.array) -> None:
    """
    Calculate the L.H.S. streaming operation of the Lattice Boltzmann equation
    by shifting the components of the probability density function along a grid.
    Args:
        proba_density (ndarray): Probability density function of Lattice Boltzmann
            Equation.
    Returns:
        None
    """

    # Get the number of channels
    n_channels = velocity_channels.shape[0]

    # Iterate over the number of channels
    for i in range(n_channels):
        # For the iterated channel, roll each value of the proba density
        # function based on the channel's velocity
        proba_density[i, :, :] = np.roll(proba_density[i, :, :],
                                         velocity_channels[i], axis=(0, 1))


def calculate_equilibrium_distro(density: np.array,
                                 velocity: np.array) -> np.array:
    """
    Calculate the equilibrium distribution given the density and average
    velocity.
    Args:
        density (nd.array): Mass density at each position of the grid of shape
            (X, Y).
        velocity (nd.array): Average velocity at each position of the grid of shape
            (2, X, Y)
    Returns:
        Equilibrium distribution at each (x, y) point of the grid.
    """
    # Calculate temporary variables for convenience.
    # Multiply every (x, y) coordinate value of the velocity with the
    # corresponding velocity channel
    temp_v = np.matmul(velocity.T, velocity_channels.T).T

    # For each channel, calculate the squared second order norm of the vector
    # of the (x, y) coordinates of the average velocity.
    temp_v_squared = np.einsum("cij,cij -> ij", velocity, velocity)

    # Return the equilibrium distribution value
    return (weights * (density * (
            1 + 3 * temp_v + 9./2 * temp_v ** 2 - 3./2 * temp_v_squared)).T).T


def collision_relaxation(
        proba_density: np.array,
        velocity: np.array,
        density: np.array,
        omega: Optional[float] = 0.5) -> np.array:
    """
    Calculate the collision operation.
    Args:
        proba_density (ndarray): Probability density function of Lattice Boltzmann
        Equation.
        velocity (ndarray): Average velocity at each position of the grid of shape
            (2, X, Y)
        density (ndarray): Mass density at each position of the grid of shape
            (X, Y).
        omega (Optional[float]): The collision frequency. Default value is 0.5
    Returns:
        The probability density function at each point in the grid after the
        streaming and collision operations are applied.
    """
    # Calculate the equilibrium distribution
    eql_proba_density = calculate_equilibrium_distro(density, velocity)

    # Return the new probability distribution
    return proba_density + omega * (eql_proba_density - proba_density)


def plot_density(density: np.array,
                 iter: Optional[int] = 0,
                 show: Optional[bool] = False) -> None:
    """
    Create a heatmap of the density at each lattice point.

    Args:
        density (ndarray): Mass density at each position of the grid.
        show (Optional[bool]): Whether to display the graphs or save them.
        iter (Optional[int]): Used to generate filename when saving the figures.
    Returns:
        None
    """
    # Calculate the labels of x- and y-axis
    width, height = density.shape

    column_labels = list(range(width))
    row_labels = list(range(height))

    fig, ax = plt.subplots()
    c = ax.pcolor(np.moveaxis(density, 0, 1), cmap=plt.cm.Reds)

    # put the major ticks at the middle of each cell
    _ = ax.set_xticks(np.arange(width) + 0.5, minor=False)
    _ = ax.set_yticks(np.arange(height) + 0.5, minor=False)

    _ = ax.invert_yaxis()

    _ = ax.set_xticklabels(column_labels, minor=False)
    _ = ax.set_yticklabels(row_labels, minor=False)
    _ = ax.set_ylabel('y-coordinate')
    _ = ax.set_xlabel('x-coordinate')
    _ = ax.set_title('Density')

    fig.colorbar(c, ax=ax)
    plt.grid()

    if show:
        plt.show()

    else:
        path_exists = os.path.exists("data")
        if not path_exists:
            # Create path if it does not exist
            os.makedirs("data")
        plt.savefig('data/density_{}'.format(iter))


def plot_velocity_field(velocity: np.array,
                        fig: plt.Figure,
                        ax: plt.Axes,
                        title: Optional[str] = "Velocity Field",
                        y_label: Optional[str] = "y-coordinate",
                        x_label: Optional[str] = "x-coordinate") -> None:
    """
    Create a streamplot of the velocity field.
    Args:
        velocity (ndarray): Average velocity at each position of the grid of shape
            (2, X, Y).
        fig (plt.Figure) : Matplotlib figure element for the streamplot.
        ax (plt.Axes): Matplotlib axes element for the streamplot.
        title (Optional[str]): The title of the plot.
        y_label (Optional[str]): The y-label of the plot.
        x_label (Optional[str]): The x-label of the plot.
    Returns:
        None
    """
    # Get dimensions
    X, Y = velocity.shape[1:]

    x, y = np.meshgrid(range(width), range(height))
    # with the np.moveaxis the velocities will have shape (y_shape, x_shape)
    # this means that when indexing velocity_x[0] we get all the
    # x-component velocities for y = 0 and when indexing velocity_y[0]
    # we get all the y-component velocities for y = 0.
    # In its general form it holds that:
    #   - velocity_x[y_index, x_index]
    #   - velocity_y[y_index, i_index]
    # To get all the velocities for a given x, e.g. x = 0
    # we have to index as follows: velocity_x[:, x_index]
    # the same holds for velocity_y. I.E. velocity_y[:, x_index]
    velocity_x = np.moveaxis(velocity[0], 0, 1)
    velocity_y = np.moveaxis(velocity[1], 0, 1)

    column_labels = list(range(width))
    row_labels = list(range(height))

    # Set up columns and labels
    xticks = column_labels[::5] + [column_labels[-1]]
    yticks = row_labels[::5] + [row_labels[-1]]

    _ = ax.set_xticks(xticks, minor=False)
    _ = ax.set_yticks(yticks, minor=False)

    max_velocity = np.max(velocity_x)
    min_velocity = np.min(velocity_y)

    # Create colorbar ticks
    cbar_ticks = [np.around(i, 2) for i in np.linspace(max_velocity, min_velocity)]
    # Generate streamplot with varying colors
    streamplot = ax.streamplot(x, y, velocity_x, velocity_y, color=velocity_x, density=3)
    # Add colorbar
    fig.colorbar(streamplot.lines, ax=ax, ticks=cbar_ticks)

    _ = ax.set_xticklabels(xticks, minor=False)
    _ = ax.set_yticklabels(yticks, minor=False)

    # Add labels and title
    _ = ax.set_ylabel(y_label)
    _ = ax.set_xlabel(x_label)
    _ = ax.set_title(title)


def rigid_wall(
        proba_density: np.array,
        pre_streaming_proba_density: np.array,
        location: Optional[str] = "lower") -> None:
    """
    Apply rigid wall boundary conditions.
    Args:
        proba_density (ndarray): Probability density function of Lattice.
        pre_streaming_proba_density (ndarray): Probability density function before the
            streaming operator is applied
        location (Optional[str]): Physical location of the boundary. For Couette flow only
            two possible positions: "upper" or "lower".
    Returns:
         None.
    """

    # Lower wall
    if location == "lower":
        # Channels going out
        out_channels = down_out_channels
        # Channels going in
        in_channels = down_in_channels

    # Upper wall
    elif location == "upper":
        # Channels going out
        out_channels = up_out_channels
        # Channels going in
        in_channels = up_in_channels

    # Right or left wall
    elif location == "right" or location == "left":
        raise NotImplementedError("Not Implemented.")

    else:
        raise ValueError("Invalid location given: '" + location + "'. "
                         "Allowed values are: 'upper', 'lower', 'right', or 'left'.")

    # Loop over channels and apply boundary conditions
    for i in range(len(in_channels)):
        # Set temporary variables for convenience
        temp_in, temp_out = in_channels[i], out_channels[i]
        # Index of y's that are on the lower boundary is 0
        proba_density[temp_in, :, 0] = \
            pre_streaming_proba_density[temp_out, :, 0]


def moving_wall(
        proba_density: np.array,
        pre_streaming_proba_density: np.array,
        wall_velocity: np.array,
        density: np.array,
        location: Optional[str] = "bottom") -> None:
    """
    Apply moving wall boundary conditions.
    Args:
        proba_density (ndarray): Probability density function of Lattice.
        pre_streaming_proba_density (ndarray): Probability density function before the
            streaming operator is applied
        wall_velocity (ndarray): Velocity of the moving wall as a vector.
        density (ndarray): Mass density at each position of the grid.
        location (Optional[str]): Physical location of the boundary. For Couette flow only
            two possible positions: "upper" or "lower".

    Returns:
        None.
    """
    # Calculate average density
    avg_density = density.mean()

    if location == "lower":
        raise NotImplementedError("Not Implemented yet.")

    elif location == "upper":
        # Channels going out
        out_channels = up_out_channels
        # Channels going in
        in_channels = up_in_channels
        # Loop over channels and apply boundary conditions
        for i in range(len(in_channels)):
            # Set temporary variables for convenience
            temp_in, temp_out = in_channels[i], out_channels[i]
            # Calculate term due to velocity based on the channels going out
            temp_term = \
                (-2 * weights[temp_out] * avg_density / c_s_squared) * \
                np.dot(velocity_channels[temp_out], wall_velocity)
            # Index of y's that are on the upper boundary is equal to the
            # size of the lattice - 1, for simplicity use "-1" to access
            proba_density[temp_in, :, -1] = \
                pre_streaming_proba_density[temp_out, :, -1] + temp_term

    else:
        raise ValueError("Invalid location given: '" + location + "'. "
                         "Allowed values are: 'upper' or 'lower'.")


if __name__ == "__main__":
    # Initialize density function at equilibrium
    p_stream = init_proba_density(15, 15, mode=InitMode.EQUILIBRIUM)
    # Calculate density
    density = calculate_density(p_stream)
    # Calculate velocity
    v = calculate_velocity(p_stream)

    # Increase the mass at a somewhat central point in the grid
    p_stream[:, 7, 7] += 0.01 * p_stream[:, 7, 5]

    # Calculate streaming part and plot density for 4 timesteps
    for i in trange(4):
        streaming(p_stream)
        density = calculate_density(p_stream)
        v = calculate_velocity(p_stream)
        plot_density(density, i, True)
