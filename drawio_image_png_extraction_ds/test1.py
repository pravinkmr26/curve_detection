import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from matplotlib.colors import ListedColormap

def process_image(image_path, background_color=None):
    # Load image and convert to RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, _ = image.shape

    # Determine background color (top-left pixel if not provided)
    if background_color is None:
        background_color = image[0, 0]

    # Find unique colors excluding background
    mask = np.all(image != background_color, axis=2)
    unique_colors = np.unique(image[mask].reshape(-1, 3), axis=0)

    curves = []

    print(unique_colors)

    for color in unique_colors:
        color_mask = np.all(image == color, axis=2)
        skeleton = skeletonize(color_mask)
        y_coords, x_coords = np.where(skeleton)
        
        if len(x_coords) == 0:
            continue

        # Find endpoints
        endpoints = []
        skeleton_points = set(zip(x_coords, y_coords))
        for (x, y) in skeleton_points:
            neighbors = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    if (x + dx, y + dy) in skeleton_points:
                        neighbors += 1
            if neighbors == 1:
                endpoints.append((x, y))

        # Traverse skeleton
        path = []
        if len(endpoints) == 2:
            # Open curve
            start, end = endpoints
            current = start
            prev = None
            while current != end:
                path.append(current)
                next_points = []
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        neighbor = (current[0]+dx, current[1]+dy)
                        if neighbor in skeleton_points and neighbor != prev:
                            next_points.append(neighbor)
                if not next_points:
                    break
                prev = current
                current = next_points[0]
            path.append(end)
        else:
            # Closed loop (approximate using contour)
            skeleton_uint8 = skeleton.astype(np.uint8) * 255
            contours, _ = cv2.findContours(skeleton_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if contours:
                contour = max(contours, key=lambda c: cv2.contourArea(c))
                path = [tuple(point[0]) for point in contour[:, 0, :]]

        if not path:
            continue

        # Extract coordinates
        x = [p[0] for p in path]
        y = [p[1] for p in path]

        # Calculate headings and velocities
        headings = []
        velocities = []
        for i in range(len(x)-1):
            dx = x[i+1] - x[i]
            dy = y[i+1] - y[i]
            headings.append(np.arctan2(dy, dx))
            velocities.append(np.sqrt(dx**2 + dy**2))
        
        # Duplicate last values for array length consistency
        headings.append(headings[-1] if headings else 0)
        velocities.append(velocities[-1] if velocities else 0)

        curves.append({
            'x': x,
            'y': y,
            'heading': headings,
            'velocity': velocities,
            'color': color / 255.0
        })

    return curves

def plot_curves(curves):
    plt.figure(figsize=(10, 10))
    for curve in curves:
        x, y = curve['x'], curve['y']
        color = curve['color']
        plt.plot(x, y, color=color, linewidth=2)
        
        if len(x) >= 2:
            dx = x[-1] - x[-2]
            dy = y[-1] - y[-2]
            plt.arrow(x[-2], y[-2], dx, dy, 
                      color=color, 
                      head_width=5, 
                      head_length=7, 
                      length_includes_head=True)
    
    plt.gca().invert_yaxis()
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Extracted Arrows with Direction')
    plt.grid(True)
    plt.show()

# Usage example
if __name__ == "__main__":
    curves = process_image('data/img2.png')
    for i, curve in enumerate(curves):
        print(f"Curve {i+1}:")
        print(f"Positions: {list(zip(curve['x'], curve['y']))}")
        print(f"Heading: {curve['heading']}")
        print(f"Velocity: {curve['velocity']}")
        print()
    
    plot_curves(curves)