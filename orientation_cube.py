# orientation_cube.py
import pyqtgraph.opengl as gl
import numpy as np
from PyQt5.QtGui import QVector3D


class OrientationCube(gl.GLViewWidget):
    """Enhanced 3D visualization of MPU6050 orientation using PyQtGraph's OpenGL"""

    def __init__(self):
        super().__init__()

        # Set up camera and view
        self.setCameraPosition(distance=40, elevation=30, azimuth=45)
        self.setBackgroundColor('#101520')  # Deep blue-black

        # Create a highly visible grid floor
        grid = gl.GLGridItem()
        grid.setSize(x=50, y=50)
        grid.setSpacing(x=10, y=10)
        grid.translate(0, 0, -5)  # Move below origin
        grid.setColor((0.5, 0.5, 0.8, 1.0))  # Bright blue, fully opaque
        self.addItem(grid)

        # Create a bow-shaped cube
        self.cube = gl.GLBoxItem(size=QVector3D(8, 12, 2))
        self.cube.setColor((0.2, 0.5, 0.9, 0.8))
        self.addItem(self.cube)

        # Add coordinate axes
        self.axis = gl.GLAxisItem(size=QVector3D(20, 20, 20))
        self.addItem(self.axis)

        # Create a simple arrow (line with different color sections)
        pts = np.array([
            [0, 0, 0],  # Origin
            [0, 0, 12],  # Arrow shaft end
            [0, 0, 15]  # Arrow tip
        ])

        # Create arrow shaft
        self.arrow_shaft = gl.GLLinePlotItem(
            pos=pts[0:2],  # First two points (origin to shaft end)
            color=(1, 0.6, 0, 1),  # Orange
            width=3  # Thicker line
        )
        self.addItem(self.arrow_shaft)

        # Create arrow tip (different color)
        self.arrow_tip = gl.GLLinePlotItem(
            pos=pts[1:3],  # Last two points (shaft end to tip)
            color=(1, 0.3, 0, 1),  # Darker orange
            width=5  # Even thicker
        )
        self.addItem(self.arrow_tip)

        # Add yellow dot at origin for reference
        self.origin_dot = gl.GLScatterPlotItem(
            pos=np.array([[0, 0, 0]]),
            size=10,
            color=(1, 1, 0, 1)
        )
        self.addItem(self.origin_dot)

        # Add a sphere at the tip for a more noticeable arrowhead
        self.tip_dot = gl.GLScatterPlotItem(
            pos=np.array([[0, 0, 15]]),
            size=12,
            color=(1, 0.2, 0, 1)  # Bright red-orange
        )
        self.addItem(self.tip_dot)

    def update_orientation(self, roll, pitch, yaw):
        """Update visualization orientation based on roll, pitch, yaw angles (in degrees)"""
        # Reset all transformations
        self.cube.resetTransform()
        self.arrow_shaft.resetTransform()
        self.arrow_tip.resetTransform()
        self.tip_dot.resetTransform()

        # Apply rotations to each item
        for item in [self.cube, self.arrow_shaft, self.arrow_tip, self.tip_dot]:
            item.rotate(roll, 1, 0, 0)
            item.rotate(pitch, 0, 1, 0)
            item.rotate(yaw, 0, 0, 1)