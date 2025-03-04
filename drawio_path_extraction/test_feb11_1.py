import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev
from matplotlib.patches import FancyArrowPatch, Circle
from scipy.interpolate import interp1d

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
                    "source": (float(source.get("x")), float(source.get("y"))),
                    "target": (float(target.get("x")), float(target.get("y"))),
                    "waypoints": [],
                    "color": color or "#000000"
                }
                
                if points is not None:
                    for pt in points.findall("mxPoint"):
                        arrow["waypoints"].append(
                            (float(pt.get("x")), float(pt.get("y")))
                        )
                
                arrows.append(arrow)
    
    return arrows

def bspline_to_bezier(points, smoothness=0.0):
    """Convert points to smooth B-spline and return Bezier control points"""
    points = np.array(points)
    tck, u = splprep(points.T, s=smoothness)
    unew = np.linspace(0, 1, 100)
    xnew, ynew = splev(unew, tck)
    return np.column_stack((xnew, ynew))

def generate_path_data(points, step=2):
    """Generate parameterized path data with 2-unit spacing"""
    if len(points) < 2:
        return []
    
    # Calculate cumulative distance
    dist = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1)))
    dist = np.insert(dist, 0, 0)
    
    # Create interpolation functions
    fx = interp1d(dist, points[:, 0], kind='cubic')
    fy = interp1d(dist, points[:, 1], kind='cubic')
    
    # Generate samples
    max_dist = dist[-1]
    sample_dist = np.arange(0, max_dist, step)
    
    # Calculate derivatives
    x = fx(sample_dist)
    y = fy(sample_dist)
    dx = np.gradient(x, sample_dist)
    dy = np.gradient(y, sample_dist)
    
    return [{
        "x": x[i],
        "y": y[i],
        "velocity": np.hypot(dx[i], dy[i]),
        "heading": np.arctan2(dy[i], dx[i])
    } for i in range(len(sample_dist))]

def plot_arrows(arrows):
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for arrow in arrows:
        color = arrow["color"]
        all_points = [arrow["source"]] + arrow["waypoints"] + [arrow["target"]]
        
        # Remove duplicates
        clean_points = []
        for p in all_points:
            if not clean_points or np.linalg.norm(np.subtract(p, clean_points[-1])) > 1e-3:
                clean_points.append(p)
        clean_points = np.array(clean_points)
        
        if len(clean_points) < 2:
            continue
            
        # Generate B-spline
        curve = bspline_to_bezier(clean_points)
        
        # Add arrowhead
        dx = curve[-1,0] - curve[-2,0]
        dy = curve[-1,1] - curve[-2,1]
        arrow_head = FancyArrowPatch(
            (curve[-2,0], curve[-2,1]), 
            (curve[-1,0], curve[-1,1]),
            arrowstyle='->',
            color=color,
            mutation_scale=20,
            linewidth=2
        )
        ax.add_patch(arrow_head)
        
        # Plot curve
        ax.plot(curve[:,0], curve[:,1], color=color, lw=2)
        
        # Add start marker
        start_circle = Circle(
            curve[0], 
            radius=4, 
            facecolor=color, 
            edgecolor='black',
            zorder=3
        )
        ax.add_patch(start_circle)
        
        # Generate path data
        arrow["path_data"] = generate_path_data(curve)
    
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 700)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.show()
    return arrows

# Process XML and plot
xml_content = open("image1.xml").read()
arrows = parse_arrows_from_xml(xml_content)
arrows = plot_arrows(arrows)

# Export path data
if arrows:
    print("Path coordinates for first arrow (2-unit spacing):")
    print(f"{'X':>8} {'Y':>8} {'Velocity':>10} {'Heading':>12}")
    for p in arrows[0]["path_data"][::5]:
        print(f"{p['x']:8.1f} {p['y']:8.1f} {p['velocity']:10.2f} {np.degrees(p['heading']):12.1f}°")