import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

def parse_drawio_xml(xml_content):
    root = ET.fromstring(xml_content)
    edges = []

    for cell in root.findall(".//mxCell[@edge='1']"):
        style = cell.get('style', '')
        geometry = cell.find('mxGeometry')
        if geometry is not None:
            source_point = geometry.find("mxPoint[@as='sourcePoint']")
            target_point = geometry.find("mxPoint[@as='targetPoint']")
            control_points = geometry.find("Array[@as='points']")

            edge_data = {
                'style': style,
                'source_point': (float(source_point.get('x')), float(source_point.get('y'))) if source_point is not None else None,
                'target_point': (float(target_point.get('x')), float(target_point.get('y'))) if target_point is not None else None,
                'control_points': [(float(pt.get('x')), float(pt.get('y'))) for pt in control_points.findall('mxPoint')] if control_points is not None else []
            }
            edges.append(edge_data)
    return edges

def quadratic_bezier(p0, p1, p2, t):
    return (1 - t)**2 * p0 + 2 * (1 - t) * t * p1 + t**2 * p2

def plot_edges(edges):
    fig, ax = plt.subplots()

    for edge in edges:
        points = [edge['source_point']] + edge['control_points'] + [edge['target_point']]
        points = [pt for pt in points if pt is not None]

        if len(points) == 3:  # Quadratic Bézier curve
            p0, p1, p2 = points
            t_values = np.linspace(0, 1, 100)
            bezier_points = [quadratic_bezier(np.array(p0), np.array(p1), np.array(p2), t) for t in t_values]
            x_values, y_values = zip(*bezier_points)
            ax.plot(x_values, y_values, label='Quadratic Bézier Curve')
        elif len(points) == 2:  # Straight line
            x_values, y_values = zip(*points)
            ax.plot(x_values, y_values, label='Straight Line')
        else:
            print("Unsupported number of points for Bézier curve.")

        if edge['target_point']:
            ax.annotate('', xy=edge['target_point'], xytext=points[-2],
                        arrowprops=dict(arrowstyle='->', color='black'))

    ax.set_aspect('equal')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Reconstructed Draw.io Arrows')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    xml_content = open("image1.xml").read()
    edges = parse_drawio_xml(xml_content)
    plot_edges(edges)
