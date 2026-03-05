import numpy as np
import torch
import pyvista as pv
import random
from shape_process.gm_transformer import GMTransformer


class VisualizationManager:

    colors=[
        "aliceblue",
        #"antiquewhite",
        "aquamarine",
        "azure",
        # "beige",
        "bisque",
        "black",
        #"blanchedalmond",
        "blue",
        "blueviolet",
        "brown",
        "burlywood",
        "cadetblue",
        "chartreuse",
        "chocolate",
        "coral",
        "cornflowerblue",
        "cornsilk",
        "crimson",
        "cyan",
        "darkblue",
        "darkcyan",
        "darkgoldenrod",
        "darkgray",
        "darkgreen",
        "darkkhaki",
        "darkmagenta",
        "darkolivegreen",
        "darkorange",
        "darkorchid",
        "darkred",
        "darksalmon",
        "darkseagreen",
        "darkslateblue",
        "darkslategray",
        "darkturquoise",
        "darkviolet",
        "deeppink",
        "deepskyblue",
        "dimgray",
        "dodgerblue",
        "firebrick",
        #"floralwhite",
        "forestgreen",
        #"gainsboro",
        #"ghostwhite",
        "gold",
        "goldenrod",
        "gray",
        "green",
        "greenyellow",
        "honeydew",
        "hotpink",
        "indianred",
        "indigo",
        "ivory",
        "khaki",
        "lavender",
        "lavenderblush",
        "lawngreen",
        "lemonchiffon",
        # "lightblue",
        # "lightcoral",
        # "lightcyan",
        # "lightgoldenrodyellow",
        # "lightgray",
        # "lightgreen",
        # "lightpink",
        # "lightsalmon",
        # "lightseagreen",
        # "lightskyblue",
        # "lightslategray",
        # "lightsteelblue",
        # "lightyellow",
        "lime",
        "limegreen",
        #"linen",
        "magenta",
        "maroon",
        "mediumaquamarine",
        "mediumblue",
        "mediumorchid",
        "mediumpurple",
        "mediumseagreen",
        "mediumslateblue",
        "mediumspringgreen",
        "mediumturquoise",
        "mediumvioletred",
        "midnightblue",
        "mintcream",
        "mistyrose",
        "moccasin",
        #"navajowhite",
        "navy",
        #"oldlace",
        "olive",
        "olivedrab",
        "orange",
        "orangered",
        "orchid",
        "palegoldenrod",
        "palegreen",
        "paleturquoise",
        "palevioletred",
        "papayawhip",
        "paraview_background",
        "peachpuff",
        "peru",
        "pink",
        "plum",
        "powderblue",
        "purple",
        "raw_sienna",
        "rebeccapurple",
        "red",
        "rosybrown",
        "royalblue",
        "saddlebrown",
        "salmon",
        "sandybrown",
        "seagreen",
        #"seashell",
        "sienna",
        #"silver",
        "skyblue",
        "slateblue",
        "slategray",
        "snow",
        "springgreen",
        "steelblue",
        "tan",
        "teal",
        "thistle",
        "tomato",
        "turquoise",
        "violet",
        "wheat",
        #"white",
        #"whitesmoke",
        "yellow",
        "yellowgreen",
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
        "tab:pink",
        "tab:gray",
        "tab:olive",
        "tab:cyan",
    ]

    def __init__(self, shape=(1, 1), off_screen=False):
        self.plotter = pv.Plotter(shape=shape, off_screen=off_screen)
        self.plotter.set_background("white")
        self.last_subplot = (0, 0)  # 新增：用于跟踪最后一次使用的subplot位置

    def _set_subplot(self, subplot):
        if subplot is None:
            subplot = self.last_subplot
        else:
            self.last_subplot = subplot  # 更新最后使用的subplot位置
        self.plotter.subplot(*subplot)

    def add_axes_at_origin(self, subplot=None):
        self._set_subplot(subplot)
        axes_length = 1  # 设置坐标轴的长度
        # X轴
        self.plotter.add_lines(
            np.array([[0, 0, 0], [axes_length, 0, 0]]), color="green"
        )
        # Y轴
        self.plotter.add_lines(
            np.array([[0, 0, 0], [0, axes_length, 0]]), color="green"
        )
        # Z轴
        self.plotter.add_lines(
            np.array([[0, 0, 0], [0, 0, axes_length]]), color="green"
        )

    def add_points(
        self,
        points,
        labels=None,
        subplot=None,
        color=None,
        point_size=3,
        colors_to_show=None,
    ):
        self._set_subplot(subplot)
        points = (
            points.detach().cpu().numpy()
            if isinstance(points, torch.Tensor)
            else points
        )
        if labels is not None:
            labels = (
                labels.detach().cpu().numpy()
                if isinstance(labels, torch.Tensor)
                else labels
            )
            true_points = points[labels > 0.01]
            false_points = points[labels < -0.01]
            surface_points = points[(-0.01 <= labels) & (labels <= 0.01)]

            # Set default color display if colors_to_show is not provided
            if colors_to_show is None:
                colors_to_show = ["blue", "red", "green"]

            if "blue" in colors_to_show and len(true_points) > 0:
                self.plotter.add_points(true_points, color="blue", point_size=3)
            if "red" in colors_to_show and len(false_points) > 0:
                self.plotter.add_points(false_points, color="red", point_size=3)
            if "green" in colors_to_show and len(surface_points) > 0:
                self.plotter.add_points(surface_points, color="green", point_size=3)
        else:
            if color is None:
                color = random.choice(self.colors)
            self.plotter.add_points(points, color=color, point_size=point_size)

    def add_lines(self, start_points, end_points, color="blue", line_width=2):
        lines = pv.PolyData()
        points = np.vstack((start_points, end_points))
        lines.points = points

        num_lines = len(start_points)
        start_indices = np.arange(num_lines)
        end_indices = np.arange(num_lines, 2 * num_lines)

        indices = np.stack((start_indices, end_indices), axis=1)
        cells = np.hstack((np.full((num_lines, 1), 2), indices)).flatten()
        lines.lines = cells

        self.plotter.add_mesh(lines, color=color, line_width=line_width)

    def add_gmm(self, gmm, subplot=None, color="yellow"):
        self._set_subplot(subplot)
        gmm = gmm.detach().cpu().numpy() if isinstance(gmm, torch.Tensor) else gmm
        transformer = GMTransformer(gmm)
        sphere = pv.Sphere(radius=1, center=(0, 0, 0))
        transformed_points = transformer.transform(sphere.points)
        transformed_sphere = pv.PolyData(np.array(transformed_points), sphere.faces)
        self.plotter.add_mesh(transformed_sphere, color=color)

    def add_gmm_axes(self, gmm, subplot=None, color="black"):
        self._set_subplot(subplot)
        gmm = gmm.detach().cpu().numpy() if isinstance(gmm, torch.Tensor) else gmm
        transformer = GMTransformer(gmm)

        # Define points along the coordinate axes in both positive and negative directions
        axis_points = np.array(
            [
                [0, 0, 0],  # Origin
                [1, 0, 0],
                [-1, 0, 0],  # X-axis positive and negative
                [0, 1, 0],
                [0, -1, 0],  # Y-axis positive and negative
                [0, 0, 1],
                [0, 0, -1],  # Z-axis positive and negative
            ]
        )

        # Transform these points
        transformed_points = transformer.transform(axis_points)

        # Create lines from the origin to each point
        lines = [
            [2, 0, 1],
            [2, 0, 2],  # Lines along the X-axis
            [2, 0, 3],
            [2, 0, 4],  # Lines along the Y-axis
            [2, 0, 5],
            [2, 0, 6],  # Lines along the Z-axis
        ]

        # Create a polydata object with these lines
        line_polydata = pv.PolyData(transformed_points, lines=lines)

        # Add the transformed line segments to the plot
        self.plotter.add_mesh(line_polydata, color=color, line_width=1)

    def add_points_with_gmm(
        self, points, gmm, labels=None, subplot=None, color="blue", point_size=3
    ):
        self._set_subplot(subplot)

        # Check if points or labels are PyTorch tensors and move them to CPU if so
        points = (
            points.detach().cpu().numpy()
            if isinstance(points, torch.Tensor)
            else points
        )
        if labels is not None:
            labels = (
                labels.detach().cpu().numpy()
                if isinstance(labels, torch.Tensor)
                else labels
            )

        # Create a GMM transformer and transform the points
        transformer = GMTransformer(
            gmm.detach().cpu().numpy() if isinstance(gmm, torch.Tensor) else gmm
        )
        transformed_points = transformer.transform(points)

        # Use the existing add_points method to plot the transformed points
        self.add_points(
            transformed_points,
            labels=labels,
            subplot=subplot,
            color=color,
            point_size=point_size,
        )

    def add_voxels(
        self,
        voxels,
        coords=None,
        subplot=None,
        point_color="red",
        lines_color="blue",
        line_width=1,
        point_size=8,
    ):
        self._set_subplot(subplot)
        space_range = [-2, 2]
        voxel_scale = 64
        voxel_size = np.array([(space_range[1] - space_range[0]) / voxel_scale] * 3)

        voxels = np.array(voxels.cpu())

        if coords is not None:
            coords = np.array(coords.cpu())
            voxel_centers = (coords[..., ::-1] + 0.5) * voxel_size + space_range[0]
            voxels[:, :3] = (voxels[:, :3] * voxel_size) + voxel_centers

        start_points = voxels[:, :3]
        # Assume voxels[:, 3:] contains the normal vectors
        normals = voxels[:, 3:]

        # Call add_normals to display the points and normals
        self.add_normals(
            start_points,
            normals,
            subplot=subplot,
            point_color=point_color,
            lines_color=lines_color,
            line_width=line_width,
            point_size=point_size,
        )

    def add_normals(
        self,
        start_points,
        normals,
        subplot=None,
        point_color="red",
        lines_color="blue",
        line_width=2,
        point_size=8,
    ):
        self._set_subplot(subplot)
        end_points = (
            start_points + normals * 0.01
        )  # Normals are assumed to be pre-scaled appropriately

        # Add points and lines to the plotter
        self.add_lines(
            start_points, end_points, color=lines_color, line_width=line_width
        )
        self.add_points(
            start_points, subplot=subplot, color=point_color, point_size=point_size
        )

    def add_obb(self, obb, subplot=None, color="green", opacity=0.5):
        self._set_subplot(subplot)
        # 获取OBB的变换矩阵和范围
        transform = obb.transform
        extents = obb.extents

        # 创建一个单位立方体
        unit_cube = pv.Cube(center=(0, 0, 0), x_length=1, y_length=1, z_length=1)

        # 应用OBB的缩放和变换
        scaled_cube = unit_cube.scale(extents)
        transformed_cube = scaled_cube.transform(transform)

        # 添加OBB到plotter中
        self.plotter.add_mesh(transformed_cube, color=color, opacity=opacity)

    def add_aabb(self, extents, color="red", opacity=0.5):
        # 创建一个单位立方体，其中心在原点
        unit_cube = pv.Cube(center=(0, 0, 0), x_length=1, y_length=1, z_length=1)

        # 将单位立方体缩放到指定的范围
        scaled_cube = unit_cube.scale(extents)

        # 添加AABB到plotter中
        self.plotter.add_mesh(scaled_cube, color=color, opacity=opacity)

    def show(self):
        self.plotter.show()
