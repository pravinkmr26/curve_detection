import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import math

def parse_xml(xml_content):
    tree = ET.ElementTree(ET.fromstring(xml_content))
    root = tree.getroot()

    arrows = []
    for cell in root.findall(".//mxCell[@edge='1']"):
        style = cell.get('style', '')
        if 'curved=1' in style:
            geometry = cell.find('mxGeometry')
            source_point = geometry.find('mxPoint[@as="sourcePoint"]')
            target_point = geometry.find('mxPoint[@as="targetPoint"]')
            points = geometry.find('Array')

            if source_point is not None and target_point is not None:
                source = (float(source_point.get('x')), float(source_point.get('y')))
                target = (float(target_point.get('x')), float(target_point.get('y')))
                control_points = []
                if points is not None:
                    for point in points.findall('mxPoint'):
                        control_points.append((float(point.get('x')), float(point.get('y'))))

                # Extract color from style
                color = None
                if 'strokeColor=#' in style:
                    color = '#' + style.split('strokeColor=#')[1][:6]

                arrows.append({
                    'source': source,
                    'target': target,
                    'control_points': control_points,
                    'color': color
                })

    return arrows

def bezier_curve(t, points):
    """Calculate a point on a Bézier curve defined by control points."""
    print("control from bezier_curve()", t)
    n = len(points) - 1
    x, y = 0, 0
    for i, (px, py) in enumerate(points):
        coeff = math.comb(n, i) * (1 - t)**(n - i) * t**i
        x += coeff * px
        y += coeff * py
    return x, y

    

def generate_high_res_points(source, target, control_points, resolution=1):
    """Generate high-resolution points along the Bézier curve."""
    points = [source] + control_points + [target]
    curve_length = sum(np.linalg.norm(np.array(points[i+1]) - np.array(points[i])) for i in range(len(points)-1))
    num_points = int(curve_length / resolution)

    t_values = np.linspace(0, 1, num_points)
    high_res_points = [bezier_curve(t, points) for t in t_values]
    return high_res_points

def plot_arrows(arrows):
    """Plot arrows with high-resolution points, arrowheads, and start circles."""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 700)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert y-axis to match draw.io's coordinate system

    for arrow in arrows:
        high_res_points = generate_high_res_points(arrow['source'], arrow['target'], arrow['control_points'])
        x_coords, y_coords = zip(*high_res_points)

        # Plot the curve
        ax.plot(x_coords, y_coords, color=arrow['color'], label=f"Color: {arrow['color']}")

        # Add arrowhead at the end
        ax.annotate('', xy=arrow['target'], xytext=high_res_points[-2],
                    arrowprops=dict(arrowstyle='->', color=arrow['color'], lw=1.5))

        # Add circle at the start
        circle = Circle(arrow['source'], radius=5, color=arrow['color'], fill=True)
        ax.add_patch(circle)

    plt.legend()
    plt.title("Curved Arrows with Arrowheads and Start Circles")
    plt.show()

def main():
    f = open("data/image1.xml")
    xml_content = f.read()
    arrows = parse_xml(xml_content)
    plot_arrows(arrows)

if __name__ == "__main__":
    main()