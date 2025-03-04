from drawpyo.diagram import Diagram, Layer, Page
from drawpyo.lib import base
import matplotlib.pyplot as plt

def parse_drawio(file_path):
    """Parse draw.io file using drawpyo"""
    diagram = Diagram()
    diagram.import_from_file(file_path)
    
    arrows = []
    for page in diagram.pages:
        for obj in page.objects:
            if isinstance(obj, base.Edge):
                arrows.append({
                    "source": (obj.source_point.x, obj.source_point.y),
                    "target": (obj.target_point.x, obj.target_point.y),
                    "waypoints": [(pt.x, pt.y) for pt in obj.waypoints],
                    "style": obj.style
                })
    return arrows

def plot_arrows(arrows):
    """Plot arrows with matplotlib"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for arrow in arrows:
        # Extract points
        points = [arrow["source"]] + arrow["waypoints"] + [arrow["target"]]
        x = [p[0] for p in points]
        y = [p[1] for p in points]
        
        # Plot curve
        ax.plot(x, y, color=_extract_color(arrow["style"]), lw=2)
        
        # Add arrowhead
        ax.annotate('', 
                   xy=arrow["target"], 
                   xytext=arrow["waypoints"][-1] if arrow["waypoints"] else arrow["source"],
                   arrowprops=dict(arrowstyle='->', color=_extract_color(arrow["style"])))
        
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.show()

def _extract_color(style_str):
    """Extract color from style string"""
    if "strokeColor=" in style_str:
        return "#" + style_str.split("strokeColor=#")[1].split(";")[0]
    return "#000000"

def generate_path_data(arrows, step=2):
    """Generate coordinate data with velocity and heading"""
    path_data = []
    for arrow in arrows:
        points = [arrow["source"]] + arrow["waypoints"] + [arrow["target"]]
        data = []
        
        for i in range(len(points)-1):
            x0, y0 = points[i]
            x1, y1 = points[i+1]
            dx = x1 - x0
            dy = y1 - y0
            distance = (dx**2 + dy**2)**0.5
            
            num_points = int(distance // step) + 1
            for t in np.linspace(0, 1, num_points):
                data.append({
                    "x": x0 + dx*t,
                    "y": y0 + dy*t,
                    "velocity": distance/num_points,
                    "heading": np.arctan2(dy, dx)
                })
        
        path_data.append(data)
    return path_data

# Usage
file_path = "image1.xml"
arrows = parse_drawio(file_path)
plot_arrows(arrows)
path_data = generate_path_data(arrows)

# Print sample data
print("Path coordinates for first arrow:")
for p in path_data[0][::5]:  # Print every 5th point
    print(f"X: {p['x']:.1f}, Y: {p['y']:.1f}, Velocity: {p['velocity']:.2f}, Heading: {np.degrees(p['heading']):.1f}°")