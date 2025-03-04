import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Circle
from scipy.interpolate import interp1d

from coords_animation import Animation

def parse_arrows_from_xml(xml_content):
    arrows = []
    root = ET.fromstring(xml_content)
    
    for cell in root.findall(".//mxCell"):
        if cell.get("edge") == "1" and "curved=1" in cell.get("style", ""):
            style = cell.get("style", "")
            color = style.split("strokeColor=")[-1].split(";")[0].strip("#")
            if color:
                color = f"#{color}"
            
            geometry = cell.find("mxGeometry")
            source = geometry.find("mxPoint[@as='sourcePoint']")
            target = geometry.find("mxPoint[@as='targetPoint']")
            points = geometry.find("Array")
            
            if source is not None and target is not None:
                arrow = {
                    "source": np.array([float(source.get("x")), float(source.get("y"))]),
                    "target": np.array([float(target.get("x")), float(target.get("y"))]),
                    "waypoints": [],
                    "color": color or "#000000"
                }
                
                if points is not None:
                    for pt in points.findall("mxPoint"):
                        arrow["waypoints"].append(
                            np.array([float(pt.get("x")), float(pt.get("y"))])
                        )
                
                arrows.append(arrow)
    
    return arrows

def catmull_rom_to_bezier(p0, p1, p2, p3, tension=0.5):
    """Convert 4 Catmull-Rom points to Bezier control points with tension"""
    d1 = np.linalg.norm(p1 - p0)
    d2 = np.linalg.norm(p2 - p1)
    d3 = np.linalg.norm(p3 - p2)
    
    d1 = max(d1, 1e-3)
    d2 = max(d2, 1e-3)
    d3 = max(d3, 1e-3)
    
    b1 = p1 + (p2 - p0) * (tension * d2 / (d1 + d2)) / 3
    b2 = p2 - (p3 - p1) * (tension * d2 / (d2 + d3)) / 3
    return [p1, b1, b2, p2]

def generate_path_data(points, step=2):
    """Generate parameterized path data with heading and velocity"""
    if len(points) < 2:
        return []
    
    # Clean points: remove duplicates and near-duplicates
    clean_points = [points[0]]
    for p in points[1:]:
        if np.linalg.norm(p - clean_points[-1]) > 1e-3:
            clean_points.append(p)
    clean_points = np.array(clean_points)
    
    if len(clean_points) < 2:
        return []
    
    # Calculate cumulative distance along path
    dist = np.zeros(len(clean_points))
    for i in range(1, len(clean_points)):
        dist[i] = dist[i-1] + np.linalg.norm(clean_points[i] - clean_points[i-1])
    
    # Handle potential duplicate distances
    _, unique_indices = np.unique(dist, return_index=True)
    dist = dist[unique_indices]
    clean_points = clean_points[unique_indices]
    
    if len(clean_points) < 2:
        return []
    
    # Create interpolation functions
    fx = interp1d(dist, clean_points[:, 0], kind='cubic')
    fy = interp1d(dist, clean_points[:, 1], kind='cubic')
    
    # Generate samples every 2 units
    max_dist = dist[-1]
    num_samples = int(max_dist // step) + 1
    sample_dist = np.linspace(0, max_dist, num_samples)
    
    # Calculate positions and derivatives
    x = fx(sample_dist)
    y = fy(sample_dist)
    dx = np.gradient(x, sample_dist)
    dy = np.gradient(y, sample_dist)
    
    # Calculate heading (in radians) and velocity
    heading = np.arctan2(dy, dx)
    velocity = np.sqrt(dx**2 + dy**2)
    
    return [{
        "x": float(x[i]),
        "y": float(y[i]),
        "velocity": float(velocity[i]),
        "heading": float(heading[i])
    } for i in range(len(sample_dist))]

def plot_arrows(arrows):
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for arrow in arrows:
        color = arrow["color"]
        points = [arrow["source"]] + arrow["waypoints"] + [arrow["target"]]
        
        # Remove consecutive duplicates
        points = [p for i, p in enumerate(points) 
                 if i == 0 or not np.allclose(p, points[i-1], atol=1e-3)]
        
        if len(points) < 2:
            continue
            
        # Extended points for Catmull-Rom
        extended = np.vstack([points[0]] * 2 + points + [points[-1]] * 2)
        
        # Generate full path
        full_path = []
        for i in range(1, len(extended) - 2):
            p0, p1, p2, p3 = extended[i-1:i+3]
            bezier_ctrl = catmull_rom_to_bezier(p0, p1, p2, p3)
            
            t = np.linspace(0, 1, 200)
            curve = np.column_stack([
                (1 - t)**3 * bezier_ctrl[0][0] + 
                3 * (1 - t)**2 * t * bezier_ctrl[1][0] + 
                3 * (1 - t) * t**2 * bezier_ctrl[2][0] + 
                t**3 * bezier_ctrl[3][0],
                (1 - t)**3 * bezier_ctrl[0][1] + 
                3 * (1 - t)**2 * t * bezier_ctrl[1][1] + 
                3 * (1 - t) * t**2 * bezier_ctrl[2][1] + 
                t**3 * bezier_ctrl[3][1]
            ])
            full_path.append(curve)
        
        if not full_path:
            continue
            
        full_path = np.vstack(full_path)
        
        # Remove near-duplicate points in full path
        mask = [True]
        for i in range(1, len(full_path)):
            if np.linalg.norm(full_path[i] - full_path[i-1]) > 1e-3:
                mask.append(True)
            else:
                mask.append(False)
        full_path = full_path[mask]
        
        if len(full_path) < 2:
            continue
      
        print("path length ", len(full_path))
            
        # Add arrowhead
        arrow_head = FancyArrowPatch(
            posA=full_path[-2], 
            posB=full_path[-1],
            arrowstyle='->',
            color=color,
            mutation_scale=20,
            linewidth=2
        )
        ax.add_patch(arrow_head)
        
        # Plot path
        ax.plot(full_path[:,0], full_path[:,1], color=color, lw=2)
        
        # Add start circle
        start_circle = Circle(
            full_path[0], 
            radius=4, 
            facecolor=color, 
            edgecolor='black',
            zorder=3
        )
        ax.add_patch(start_circle)
        
        # Generate and store path data
        arrow["path_data"] = generate_path_data(full_path)
        arrow["full_path"] = full_path
    
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 700)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.show()
    return arrows

# Process XML and plot
f = open("image1.xml")
xml_content = f.read()
arrows = parse_arrows_from_xml(xml_content)
arrows = plot_arrows(arrows)

arrows_2d = []
for arrow in arrows:
    arrows_2d.append(arrow["full_path"])
    #print(arrow["full_path"])

animation = Animation(arrows_2d)
animation.animate_paths()






# Example output
if arrows:
    print("Path coordinates and parameters for first arrow:")
    print(f"{'X':>8} {'Y':>8} {'Velocity':>10} {'Heading(deg)':>12}")
    for i, arrow in enumerate(arrows):
        with open(f"output_{i}.txt", "w") as ff:
            ff.write(f"{'X':>8} {'Y':>8} {'Velocity':>10} {'Heading(deg)':>12}\n")
            for p in arrow["path_data"]:  # Print every 5th point
                ff.write(f"{p['x']:8.1f} {p['y']:8.1f} {p['velocity']:10.2f} {np.degrees(p['heading']):12.1f}\n")