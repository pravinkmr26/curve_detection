import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

def parse_arrows_from_xml(xml_content):
    arrows = []
    root = ET.fromstring(xml_content)
    
    for cell in root.findall(".//mxCell"):
        if cell.get("edge") == "1":
            style = cell.get("style", "")
            if "curved=1" in style:
                geometry = cell.find("mxGeometry")
                source_point = geometry.find("mxPoint[@as='sourcePoint']")
                target_point = geometry.find("mxPoint[@as='targetPoint']")
                points = geometry.find("Array")
                
                if source_point is not None and target_point is not None:
                    arrow = {
                        "source": (float(source_point.get("x")), float(source_point.get("y"))),
                        "target": (float(target_point.get("x")), float(target_point.get("y"))),
                        "points": [],
                        "color": style.split("strokeColor=")[1].split(";")[0] if "strokeColor=" in style else "#000000"
                    }
                    
                    if points is not None:
                        for point in points.findall("mxPoint"):
                            arrow["points"].append((float(point.get("x")), float(point.get("y"))))
                    
                    arrows.append(arrow)
    
    return arrows

def cubic_bezier(p0, p1, p2, p3, t):
    """Calculate a point on a cubic Bézier curve."""
    x = (1 - t)**3 * p0[0] + 3 * (1 - t)**2 * t * p1[0] + 3 * (1 - t) * t**2 * p2[0] + t**3 * p3[0]
    y = (1 - t)**3 * p0[1] + 3 * (1 - t)**2 * t * p1[1] + 3 * (1 - t) * t**2 * p2[1] + t**3 * p3[1]
    return (x, y)

def plot_arrows(arrows):
    fig, ax = plt.subplots()
    
    for arrow in arrows:
        source = arrow["source"]
        target = arrow["target"]
        points = arrow["points"]
        color = arrow["color"]
        
        # Ensure there are exactly 2 control points for a cubic Bézier curve
        if len(points) == 2:
            p0 = source
            p1 = points[0]
            p2 = points[1]
            p3 = target
            
            # Generate points along the curve
            t_values = np.linspace(0, 1, 100)
            curve_points = [cubic_bezier(p0, p1, p2, p3, t) for t in t_values]
            
            # Unzip the points for plotting
            x_vals, y_vals = zip(*curve_points)
            
            # Plot the curve
            ax.plot(x_vals, y_vals, color=color, lw=2)
    
    ax.set_xlim(0, 800)
    ax.set_ylim(0, 700)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()  # Invert y-axis to match draw.io's coordinate system
    plt.show()

# Sample XML content
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
</mxfile>
'''

# Parse the XML content
arrows = parse_arrows_from_xml(xml_content)

# Plot the arrows
plot_arrows(arrows) 