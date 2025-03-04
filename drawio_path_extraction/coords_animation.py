import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Animation:
    def __init__(self, paths):
        self.paths = paths

    def animate_paths(self):
        # Number of frames per second
        frames_per_second = 20

        # Time interval between frames in milliseconds
        interval = int(1000 / frames_per_second)

        # Prepare the figure and axis
        fig, ax = plt.subplots()
        ax.set_xlim(0, 1000)
        ax.set_ylim(0, 1000)

        for path in self.paths:
            print("len of each path ", len(path))
        
        # Colors for each path
        colors = ['r', 'b', 'g', 'c', 'm', 'y', 'orange', 'purple', 'brown', 'pink']

        max_frames = max(len(path) for path in self.paths)
        
        # Create empty plot objects for each vehicle and trail        
        trails = [ax.plot([], [], '-', color=colors[i % len(colors)], linewidth=2, alpha=0.5)[0] for i in range(len(self.paths))]

        def init():
            """Initialize all objects to empty."""
            for trail in trails:                
                trail.set_data([], [])
            return trails

        def update(frame):
            """Update positions of all vehicles and their trails for each frame."""
            for i, path in enumerate(self.paths):
                if frame < len(path):  # Only update if the path has more points                    
                    trail_x, trail_y = zip(*path[:frame + 1])
                    print("train x, y", trail_x, trail_y)
                    trails[i].set_data(trail_x, trail_y)  # Update trail path

            return trails

        # Create the animation
        ani = animation.FuncAnimation(fig, update, frames=max_frames, init_func=init, blit=True, interval=interval)

        # Show the animation
        plt.title("Extracted paths")
        plt.xlabel("X Position")
        plt.ylabel("Y Position")
        plt.grid(True)
        plt.show()
        pass
