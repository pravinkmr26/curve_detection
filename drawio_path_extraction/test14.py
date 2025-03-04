import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
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
    
    # Calculate cumulative distance along path
    dist = np.zeros(len(points))
    for i in range(1, len(points)):
        dist[i] = dist[i-1] + np.linalg.norm(points[i] - points[i-1])
    
    # Create interpolation functions
    fx = interp1d(dist, points[:, 0], kind='quadratic')
    fy = interp1d(dist, points[:, 1], kind='quadratic')
    
    # Generate samples every 2 units
    max_dist = dist[-1]
    num_samples = int(max_dist // step) + 1
    sample_dist = np.linspace(0, max_dist, num_samples)
    
    # Calculate positions and derivatives
    x = fx(sample_dist)
    y = fy(sample_dist)
    dx = np.gradient(x, sample_dist)
    dy = np.gradient(y, sample_dist)
    
    # Calculate heading (in radians)
    heading = np.arctan2(dy, dx)
    
    # Calculate velocity magnitude
    velocity = np.sqrt(dx**2 + dy**2)
    
    return [{
        "x": x[i],
        "y": y[i],
        "velocity": velocity[i],
        "heading": heading[i]
    } for i in range(len(sample_dist))]

def plot_arrows(arrows):
    fig, ax = plt.subplots(figsize=(14, 10))
    
    for arrow in arrows:
        color = arrow["color"]
        points = [arrow["source"]] + arrow["waypoints"] + [arrow["target"]]
        
        # Convert to numpy array and remove duplicates
        points = np.array([p for i, p in enumerate(points) if i == 0 or not np.all(p == points[i-1])])
        
        if len(points) < 2:
            continue
            
        # Extended points for Catmull-Rom
        extended = np.vstack([points[0]] * 2 + [points] + [points[-1]] * 2)
        
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
        
        # Add arrowhead
        dx = full_path[-1,0] - full_path[-2,0]
        dy = full_path[-1,1] - full_path[-2,1]
        arrow_head = FancyArrowPatch(
            posA=full_path[-2], 
            posB=full_path[-1],
            arrowstyle='->',
            color=color,
            mutation_scale=15,
            linewidth=2
        )
        ax.add_patch(arrow_head)
        
        # Plot path
        ax.plot(full_path[:,0], full_path[:,1], color=color, lw=2)
        
        # Add start circle
        start_circle = Circle(
            full_path[0], 
            radius=3, 
            facecolor=color, 
            edgecolor='black'
        )
        ax.add_patch(start_circle)
        
        # Generate and store path data
        arrow["path_data"] = generate_path_data(full_path)
    
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 700)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.show()
    return arrows

# Process XML and plot
xml_content = '''<?xml version="1.0" encoding="UTF-8"?>
<mxfile host="app.diagrams.net" agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36" version="26.0.11">
  <diagram name="Page-1" id="AC31NZFUZoCqsNr8ATr-">
    <mxGraphModel dx="1119" dy="651" grid="1" gridSize="10" guides="1" tooltips="1" connect="1" arrows="1" fold="1" page="1" pageScale="1" pageWidth="827" pageHeight="1169" math="0" shadow="0">
      <root>
        <mxCell id="0" />
        <mxCell id="1" parent="0" />
        <mxCell id="e6Z4spZAflRjsUE7cnlo-1" value="" style="curved=1;endArrow=classic;html=1;rounded=0;fillColor=#6a00ff;strokeColor=#3700CC;" parent="1" edge="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="90" y="330" as="sourcePoint" />
            <mxPoint x="750" y="190" as="targetPoint" />
            <Array as="points">
              <mxPoint x="330" y="500" />
              <mxPoint x="380" y="300" />
            </Array>
          </mxGeometry>
        </mxCell>
        <mxCell id="e6Z4spZAflRjsUE7cnlo-2" value="" style="curved=1;endArrow=classic;html=1;rounded=0;fillColor=#f8cecc;strokeColor=#b85450;" parent="1" edge="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="110" y="370" as="sourcePoint" />
            <mxPoint x="770" y="230" as="targetPoint" />
            <Array as="points">
              <mxPoint x="350" y="540" />
              <mxPoint x="400" y="340" />
            </Array>
          </mxGeometry>
        </mxCell>
        <mxCell id="e6Z4spZAflRjsUE7cnlo-3" value="" style="curved=1;endArrow=classic;html=1;rounded=0;fillColor=#a20025;strokeColor=#6F0000;" parent="1" edge="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="60" y="270" as="sourcePoint" />
            <mxPoint x="720" y="130" as="targetPoint" />
            <Array as="points">
              <mxPoint x="300" y="440" />
              <mxPoint x="350" y="240" />
            </Array>
          </mxGeometry>
        </mxCell>
        <mxCell id="e6Z4spZAflRjsUE7cnlo-7" value="" style="curved=1;endArrow=classic;html=1;rounded=0;fillColor=#008a00;strokeColor=#005700;" parent="1" edge="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="440" y="600" as="sourcePoint" />
            <mxPoint x="380" y="170" as="targetPoint" />
            <Array as="points">
              <mxPoint x="530" y="470" />
              <mxPoint x="480" y="420" />
            </Array>
          </mxGeometry>
        </mxCell>
        <mxCell id="e6Z4spZAflRjsUE7cnlo-8" value="" style="curved=1;endArrow=classic;html=1;rounded=0;fillColor=#d80073;strokeColor=#A50040;" parent="1" edge="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="540" y="490" as="sourcePoint" />
            <mxPoint x="480" y="120" as="targetPoint" />
            <Array as="points">
              <mxPoint x="540" y="630" />
              <mxPoint x="610" y="390" />
            </Array>
          </mxGeometry>
        </mxCell>
        <mxCell id="VYg1U8xn8T35IRUEJENV-1" value="" style="curved=1;endArrow=classic;html=1;rounded=0;fillColor=#1ba1e2;strokeColor=#006EAF;" parent="1" edge="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="284" y="410" as="sourcePoint" />
            <mxPoint x="354" y="380" as="targetPoint" />
            <Array as="points">
              <mxPoint x="284" y="570" />
              <mxPoint x="284" y="510" />
              <mxPoint x="314" y="470" />
              <mxPoint x="334" y="470" />
              <mxPoint x="374" y="500" />
              <mxPoint x="414" y="540" />
              <mxPoint x="354" y="450" />
              <mxPoint x="284" y="360" />
            </Array>
          </mxGeometry>
        </mxCell>
        <mxCell id="Nz6khTGTEIQUDmjKZeID-1" value="" style="curved=1;endArrow=classic;html=1;rounded=0;fillColor=#e3c800;strokeColor=#B09500;" edge="1" parent="1">
          <mxGeometry width="50" height="50" relative="1" as="geometry">
            <mxPoint x="280" y="250" as="sourcePoint" />
            <mxPoint x="140" y="70" as="targetPoint" />
            <Array as="points">
              <mxPoint x="270" y="230" />
              <mxPoint x="280" y="200" />
              <mxPoint x="320" y="210" />
              <mxPoint x="300" y="290" />
              <mxPoint x="200" y="270" />
              <mxPoint x="200" y="220" />
              <mxPoint x="200" y="190" />
              <mxPoint x="270" y="130" />
              <mxPoint x="380" y="170" />
              <mxPoint x="290" y="390" />
              <mxPoint x="60" y="210" />
            </Array>
          </mxGeometry>
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''
arrows = parse_arrows_from_xml(xml_content)
arrows = plot_arrows(arrows)

# Example of accessing path data for first arrow
if arrows:
    print("\nSample path data for first arrow:")
    for i, point in enumerate(arrows[0].get("path_data", [])[:3]):
        print(f"Point {i+1}:")
        print(f"  Position: ({point['x']:.1f}, {point['y']:.1f})")
        print(f"  Velocity: {point['velocity']:.2f} units/step")
        print(f"  Heading: {np.degrees(point['heading']):.1f}°\n")