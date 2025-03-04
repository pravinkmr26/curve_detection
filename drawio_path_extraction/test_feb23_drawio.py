import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

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

def plot_edges(edges):
    fig, ax = plt.subplots()
    
    for edge in edges:
        points = [edge['source_point']] + edge['control_points'] + [edge['target_point']]
        points = [pt for pt in points if pt is not None]
        x_values, y_values = zip(*points)
        
        if len(points) > 2:
            t = np.linspace(0, 1, len(points))
            cs_x = CubicSpline(t, x_values)
            cs_y = CubicSpline(t, y_values)
            t_fine = np.linspace(0, 1, 100)
            x_smooth, y_smooth = cs_x(t_fine), cs_y(t_fine)
            ax.plot(x_smooth, y_smooth, '-', label='Curve')
        else:
            ax.plot(x_values, y_values, 'o-', label='Straight Line')
        
        # if edge['target_point']:
        #     ax.annotate('', xy=edge['target_point'], xytext=points[-2],
        #                 arrowprops=dict(arrowstyle='->', color='black'))
    
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
