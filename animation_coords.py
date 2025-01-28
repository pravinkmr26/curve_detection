import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np


# read the path coordinates from the file - coords_mask_0_reduced.txt
coordinates_file = open("coords_mask_0_reduced.txt", "r")
path_coords = []
for line in coordinates_file:
    # each line contains co-ordinates x, y in format [x, y]
    # so need to remove [ and ] and split by comma
    x, y = line[1:-2].split(",")
    path_coords.append((float(x), float(y)))

reverse_coords = path_coords[::-1]

# trasform the reverse coords with translation vector of 50, 100
reverse_coords = [(x + 50, y + 100) for x, y in reverse_coords]

# Number of frames per second
frames_per_second = 20

# Time interval between frames in milliseconds
interval = int(1000 / frames_per_second)

# Prepare the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 1000)
ax.set_ylim(0, 1000)
(line,) = ax.plot([], [], "ro-", lw=2)  # Vehicle path in red

# create anotehr line for the reverse path with color blue
(line2,) = ax.plot([], [], "bo-", lw=2)  # Vehicle path in blue


def init():
    """Initialize the background of the animation."""
    line.set_data([], [])
    return (line,)


def update(frame):
    """Update the plot for each frame."""
    x, y = zip(*path_coords[: frame + 1])
    line.set_data(x, y)

    x, y = zip(*reverse_coords[: frame + 1])
    line2.set_data(x, y)
    return line, line2


def update2(frame):
    """Update the plot for each frame."""


# Create the animation
ani = animation.FuncAnimation(
    fig, update, frames=len(path_coords), init_func=init, blit=True, interval=interval
)


# Show the plot
plt.title("Vehicle Movement Animation")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.grid(True)
plt.show()
