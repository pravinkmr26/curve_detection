import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splprep, splev

def parse_drawio_arrows(xml_content):
    root = ET.fromstring(xml_content)
    arrows = []
    
    for cell in root.findall(".//mxCell"):
        if 'edge' in cell.attrib:
            style = cell.get("style", "")
            stroke_color = extract_style_value(style, "strokeColor")
            
            geometry = cell.find(".//mxGeometry")
            if geometry is not None:
                source = geometry.find(".//mxPoint[@as='sourcePoint']")
                target = geometry.find(".//mxPoint[@as='targetPoint']")
                points = geometry.findall(".//Array/mxPoint")
                
                coords = []
                if source is not None and source.get("x") and source.get("y"):
                    coords.append((float(source.get("x")), float(source.get("y"))))
                
                for point in points:
                    if point.get("x") and point.get("y"):
                        coords.append((float(point.get("x")), float(point.get("y"))))
                
                if target is not None and target.get("x") and target.get("y"):
                    coords.append((float(target.get("x")), float(target.get("y"))))
                
                if stroke_color and coords:
                    arrows.append({"color": stroke_color, "points": coords})
    
    return arrows

def extract_style_value(style, key):
    for item in style.split(";"):
        if item.startswith(key + "="):
            return item.split("=")[1].strip("#")
    return None

def plot_arrows(arrows):
    plt.figure(figsize=(8, 6))
    
    for arrow in arrows:
        x_vals, y_vals = zip(*arrow["points"])
        if len(x_vals) > 2:
            tck, u = splprep([x_vals, y_vals], s=0)
            u_fine = np.linspace(0, 1, 100)
            smooth_curve = splev(u_fine, tck)
            plt.plot(smooth_curve[0], smooth_curve[1], label=f"Color: #{arrow['color']}", color=f"#{arrow['color']}")
        else:
            plt.plot(x_vals, y_vals, marker='o', label=f"Color: #{arrow['color']}", color=f"#{arrow['color']}")
    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.title("Extracted Arrows from draw.io XML")
    plt.grid()
    plt.show()

xml_data = '''<?xml version="1.0" encoding="UTF-8"?>
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

arrows = parse_drawio_arrows(xml_data)
plot_arrows(arrows)
