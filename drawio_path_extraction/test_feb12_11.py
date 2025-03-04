import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle, FancyArrowPatch


import numpy as np
from matplotlib.path import Path
from scipy.interpolate import interp1d

def parse_arrows_from_xml(xml_content):
    """Extract arrow data from draw.io XML"""
    arrows = []
    root = ET.fromstring(xml_content)
    
    for cell in root.findall(".//mxCell"):
        if cell.get("edge") == "1" and "curved=1" in cell.get("style", ""):
            style = cell.get("style", "")
            color = style.split("strokeColor=")[-1].split(";")[0].strip("#")
            color = f"#{color}" if color else "#000000"
            
            geometry = cell.find("mxGeometry")
            source = geometry.find("mxPoint[@as='sourcePoint']")
            target = geometry.find("mxPoint[@as='targetPoint']")
            points = geometry.find("Array")
            
            if source is not None and target is not None:
                arrow = {
                    "source": (float(source.get("x")), float(source.get("y"))),
                    "target": (float(target.get("x")), float(target.get("y"))),
                    "waypoints": [],
                    "color": color
                }
                
                if points is not None:
                    for pt in points.findall("mxPoint"):
                        arrow["waypoints"].append((
                            float(pt.get("x")), 
                            float(pt.get("y"))
                        ))
                
                arrows.append(arrow)
    
    return arrows

def create_mxgraph_path(points):
    """Create draw.io compatible path using mxGraph's algorithm"""
    verts = []
    codes = []
    
    if not points:
        return verts, codes
    
    # Start point
    verts.append(points[0])
    codes.append(Path.MOVETO)
    
    for i in range(1, len(points)):
        x0, y0 = points[i-1]
        x3, y3 = points[i]
        
        # mxGraph's curve control point calculation
        dx = x3 - x0
        dy = y3 - y0
        l = np.hypot(dx, dy)
        coeff = 0.2 * (1 if l > 20 else l/20)
        
        # Control points
        x1 = x0 + dx * 0.5 - dy * coeff
        y1 = y0 + dy * 0.5 + dx * coeff
        x2 = x3 - dx * 0.5 - dy * coeff
        y2 = y3 - dy * 0.5 + dx * coeff
        
        # Cubic Bézier segment
        verts.extend([(x1, y1), (x2, y2), (x3, y3)])
        codes.extend([Path.CURVE4, Path.CURVE4, Path.CURVE4])
    
    return verts, codes

# def generate_path_parameters(verts, codes, step=2):
#     """Generate path coordinates with velocity and heading"""
#     path = Path(verts, codes)
#     length = len(path)
#     num_samples = int(length // step) + 1
#     sample_distances = np.linspace(0, length, num_samples)
    
#     points = path.interpolated(sample_distances).vertices
#     tangents = np.gradient(points, axis=0)
    
#     return [{
#         "x": p[0],
#         "y": p[1],
#         "velocity": np.hypot(t[0], t[1]),
#         "heading": np.arctan2(t[1], t[0])
#     } for p, t in zip(points, tangents)]


def generate_path_parameters(verts, codes, step=2):
    """Generate path coordinates with velocity and heading"""
    path = Path(verts, codes)

    # Compute cumulative distances along the path
    distances = np.cumsum([0] + [np.hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(verts[:-1], verts[1:])])
    total_length = distances[-1]
    
    # Generate sample distances
    num_samples = int(total_length // step) + 1
    sample_distances = np.linspace(0, total_length, num_samples)

    # Interpolation functions for x and y coordinates
    interp_x = interp1d(distances, np.array(verts)[:, 0], kind='linear')
    interp_y = interp1d(distances, np.array(verts)[:, 1], kind='linear')

    # Interpolated points
    points = np.column_stack((interp_x(sample_distances), interp_y(sample_distances)))

    # Compute tangents using gradients
    tangents = np.gradient(points, axis=0)

    return [{
        "x": p[0],
        "y": p[1],
        "velocity": np.hypot(t[0], t[1]),
        "heading": np.arctan2(t[1], t[0])
    } for p, t in zip(points, tangents)]

def plot_arrows(arrows):
    """Visualize arrows with exact draw.io styling"""
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for arrow in arrows:
        color = arrow["color"]
        points = [arrow["source"]] + arrow["waypoints"] + [arrow["target"]]
        
        # Remove duplicate points
        clean_points = []
        for p in points:
            if not clean_points or np.linalg.norm(np.array(p) - np.array(clean_points[-1])) > 1e-3:
                clean_points.append(p)
        
        # Create draw.io compatible path
        verts, codes = create_mxgraph_path(clean_points)
        
        if not verts:
            continue
            
        # Draw main path
        patch = PathPatch(
            Path(verts, codes),
            facecolor='none',
            edgecolor=color,
            lw=2
        )
        ax.add_patch(patch)
        
        # Add arrowhead
        if len(verts) >= 2:
            arrow_head = FancyArrowPatch(
                verts[-2],
                verts[-1],
                arrowstyle='->',
                color=color,
                mutation_scale=20,
                lw=2
            )
            ax.add_patch(arrow_head)
        
        # Add start marker
        start_circle = Circle(
            verts[0],
            radius=4,
            facecolor=color,
            edgecolor='black',
            zorder=3
        )
        ax.add_patch(start_circle)
        
        # Generate path data
        arrow["path_data"] = generate_path_parameters(verts, codes)
    
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 700)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.show()
    return arrows

# Example usage
xml_content = open("image1.xml").read()
arrows = parse_arrows_from_xml(xml_content)
arrows = plot_arrows(arrows)

# Print sample data
if arrows:
    print("Path coordinates with parameters:")
    print(f"{'X':>8} {'Y':>8} {'Velocity':>10} {'Heading':>12}")
    for p in arrows[0]["path_data"][::5]:  # Print every 5th point
        print(f"{p['x']:8.1f} {p['y']:8.1f} {p['velocity']:10.2f} {np.degrees(p['heading']):12.1f}°")