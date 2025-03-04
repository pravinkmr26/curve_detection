import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np

def parse_arrows_from_xml(xml_content):
    arrows = []
    root = ET.fromstring(xml_content)
    
    for cell in root.findall(".//mxCell"):
        if cell.get("edge") == "1" and "curved=1" in cell.get("style", ""):
            geometry = cell.find("mxGeometry")
            source = geometry.find("mxPoint[@as='sourcePoint']")
            target = geometry.find("mxPoint[@as='targetPoint']")
            points = geometry.find("Array")
            
            if source is not None and target is not None:
                arrow = {
                    "source": np.array([float(source.get("x")), float(source.get("y"))]),
                    "target": np.array([float(target.get("x")), float(target.get("y"))]),
                    "waypoints": [],
                    "color": cell.get("style").split("strokeColor=")[1].split(";")[0]
                }
                
                if points is not None:
                    for pt in points.findall("mxPoint"):
                        arrow["waypoints"].append(
                            np.array([float(pt.get("x")), float(pt.get("y"))])
                        )
                
                arrows.append(arrow)
    
    return arrows

def catmull_rom_to_bezier(p0, p1, p2, p3, tension=0.5):
    """Convert 4 Catmull-Rom points to 4 Bezier control points using NumPy"""
    d1 = np.linalg.norm(p1 - p0)
    d2 = np.linalg.norm(p2 - p1)
    d3 = np.linalg.norm(p3 - p2)
    
    # Avoid division by zero
    d1 = max(d1, 1e-3)
    d2 = max(d2, 1e-3)
    d3 = max(d3, 1e-3)
    
    # Calculate Bezier control points using vector operations
    b1 = p1 + (p2 - p0) * (tension * d2 / (d1 + d2)) / 3
    b2 = p2 - (p3 - p1) * (tension * d2 / (d2 + d3)) / 3
    return [p1, b1, b2, p2]

def plot_arrows(arrows):
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for arrow in arrows:
        points = [arrow["source"]] + arrow["waypoints"] + [arrow["target"]]
        color = arrow["color"]
        
        if len(points) < 2:
            continue
            
        # Extend points for Catmull-Rom (add phantom points)
        extended = [points[0].copy()] * 2 + points + [points[-1].copy()] * 2
        
        # Generate segments
        for i in range(1, len(extended) - 2):
            p0, p1, p2, p3 = extended[i-1:i+3]
            bezier_ctrl = catmull_rom_to_bezier(p0, p1, p2, p3)
            
            # Generate curve points
            t = np.linspace(0, 1, 100)
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
            
            ax.plot(curve[:, 0], curve[:, 1], color=color, lw=2)

    ax.set_xlim(0, 800)
    ax.set_ylim(0, 700)
    ax.set_aspect('equal')
    plt.gca().invert_yaxis()
    plt.show()

# Parse and plot
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
plot_arrows(arrows)