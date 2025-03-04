import xml.etree.ElementTree as ET
import numpy as np
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

def group_arrows_by_color(arrows):
    """Group arrows by their color."""
    color_groups = {}
    for arrow in arrows:
        color = arrow['color']
        if color not in color_groups:
            color_groups[color] = []
        color_groups[color].append(arrow)
    return color_groups

def main():
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
            <mxPoint x="540" y="550" as="sourcePoint" />
            <mxPoint x="480" y="120" as="targetPoint" />
            <Array as="points">
              <mxPoint x="630" y="420" />
              <mxPoint x="580" y="370" />
            </Array>
          </mxGeometry>
        </mxCell>
      </root>
    </mxGraphModel>
  </diagram>
</mxfile>'''

    arrows = parse_xml(xml_content)
    color_groups = group_arrows_by_color(arrows)

    for color, arrows_in_color in color_groups.items():
        print(f"Arrows with color {color}:")
        for arrow in arrows_in_color:
            high_res_points = generate_high_res_points(arrow['source'], arrow['target'], arrow['control_points'])
            print(f"Start: {arrow['source']}, End: {arrow['target']}")
            print(f"Coordinates: {high_res_points}")
            print()

            # write start, coordinates, end to a csv file
            


if __name__ == "__main__":
    main()