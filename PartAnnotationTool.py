import vtk
import numpy as np
import json

"""
PartAnnotationTool 操作说明书
=================================

本工具旨在帮助用户在3D模型上进行多个部件的标注、包围盒生成以及OBB（Oriented Bounding Box）保存。以下是本工具的操作说明：

1. 初始化：
   - 使用 PartAnnotationTool 类时需要提供模型文件路径（例如：.obj 文件）以及保存 OBB 的路径（例如：.json 文件）。
   - 例如：
     `PartAnnotationTool(shape_filename, save_path)`

2. 基本交互操作：
   - 左键点击：
     - 在模型上选择一个点，系统会为当前部件添加一个标注点。
     - 如果尚未创建任何部件，会自动创建一个新部件。
     - 被选中的点将用小球进行可视化标记。
   
   - 右键点击：
     - 如果在已有颜色标注的区域内点击，则系统会切换到该区域的部件进行编辑。
     - 否则，系统会创建一个新的部件，并切换到新的部件。
   
   - 按键操作：
     - `Enter` 键：
       - 保存所有部件的立方体（而不是重新计算 OBB），并将其保存到指定路径中。
     - `Escape` 键：
       - 清除当前部件的所有标注点、包围盒及染色。
     - `Backspace` 键：
       - 撤销当前部件的上一次标注点，并重新计算现有包围盒和染色。

3. 颜色标注和包围盒：
   - 系统为每个新创建的部件分配一个不同的颜色，并用该颜色进行标注。
   - 如果选中的点超过 2 个，系统会自动生成该部件的最小包围盒（OBB），并对包围盒内部的面片进行染色。
   - 包围盒以线框形式展示，颜色与部件标注一致。

4. 保存 OBB 数据：
   - 按下 `Enter` 键时，系统会将当前立方体的数据保存到用户在初始化时提供的 JSON 文件路径中。
   - 每个保存的数据包含中心点、尺寸和方向向量，适用于后续的分析或处理。

5. 注意事项：
   - 工具支持对多个部件进行标注和管理。
   - 如果部件的数量超过预定义的颜色数量，将会随机生成新的颜色。

操作指南结束。
"""

class PartAnnotation:
    def __init__(self, annotation_color):
        self.picked_points = []
        self.box_actor = None
        self.picked_points_actors = []
        self.colored_cells = []
        self.annotation_color = annotation_color
        self.obbs = []  # List to store OBBs calculated at the end
        self.box_info = None  # Store the bounding box information directly after calculation

class PartAnnotationTool:
    def __init__(self, shape_filename, save_path):
        # Annotation colors for different parts
        self.annotation_colors = [
            (1.0, 0.0, 0.0),  # Red
            (0.0, 1.0, 0.0),  # Green
            (0.0, 0.0, 1.0),  # Blue
            (1.0, 1.0, 0.0),  # Yellow
            (1.0, 0.0, 1.0),  # Magenta
            (0.0, 1.0, 1.0),  # Cyan
            (0.5, 0.5, 0.5),  # Gray
            (0.5, 0.0, 0.5),  # Purple
            (0.5, 0.5, 0.0),  # Olive
            (0.0, 0.5, 0.5),  # Teal
            (0.75, 0.25, 0.0), # Orange
            (0.25, 0.75, 0.0), # Lime Green
            (0.25, 0.0, 0.75), # Indigo
            (0.75, 0.0, 0.25), # Rose
            (0.0, 0.75, 0.25), # Mint
            (0.0, 0.25, 0.75), # Azure
            (0.25, 0.5, 0.75), # Steel Blue
            (0.75, 0.5, 0.25), # Brown
            (0.75, 0.25, 0.5), # Salmon
            (0.5, 0.75, 0.25)  # Chartreuse
        ]

        # Save path for OBBs
        self.save_path = save_path

        # Create the render window, renderer, and interactor
        self.render_window = vtk.vtkRenderWindow()
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(0.5, 0.5, 0.5)  # Set background color to gray
        self.render_window.AddRenderer(self.renderer)
        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        # Setup the interactor style
        self.style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(self.style)

        # Setup picker to pick points in the scene
        self.picker = vtk.vtkCellPicker()
        self.interactor.SetPicker(self.picker)

        # Load the shape file (e.g., OBJ file)
        self.load_shape(shape_filename)

        # Storage for multiple part annotations
        self.parts = []
        self.current_part_index = -1

        # Connect events
        self.interactor.AddObserver("LeftButtonPressEvent", self.on_left_button_press)
        self.interactor.AddObserver("RightButtonPressEvent", self.on_right_button_press)
        self.interactor.AddObserver("KeyPressEvent", self.on_key_press)

        # Start the visualization
        self.render_window.Render()
        self.interactor.Start()

    def load_shape(self, filename):
        # Read the shape file
        reader = vtk.vtkOBJReader()
        reader.SetFileName(filename)
        reader.Update()

        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(reader.GetOutputPort())
        actor = vtk.vtkActor()
        actor.SetMapper(mapper)

        # Add the actor to the renderer
        self.renderer.AddActor(actor)
        self.shape_actor = actor
        self.shape_polydata = reader.GetOutput()

        # Initialize cell colors if not already present
        cell_colors = vtk.vtkUnsignedCharArray()
        cell_colors.SetNumberOfComponents(3)
        cell_colors.SetName("Colors")
        for _ in range(self.shape_polydata.GetNumberOfCells()):
            cell_colors.InsertNextTuple3(200, 200, 200)  # Default gray color for cells
        self.shape_polydata.GetCellData().SetScalars(cell_colors)

    def on_left_button_press(self, obj, event):
        # Get the click position
        click_pos = self.interactor.GetEventPosition()
        # Use the picker to pick the position in the scene

        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        actor = self.picker.GetActor()
        if actor == self.shape_actor:  # Only proceed if the picked actor is the shape
            position = self.picker.GetPickPosition()

            # Create a new part annotation if no part is currently active
            if self.current_part_index == -1 or len(self.parts) == 0:
                self.create_new_part()

            # Store the picked position
            current_part = self.parts[self.current_part_index]
            current_part.picked_points.append(position)

            # Visualize the picked point
            self.visualize_picked_point(position, current_part.annotation_color)

            # Store the point actor for future visibility control
            current_part.picked_points_actors.append(self.renderer.GetActors().GetLastActor())

            # Update the bounding box and coloring
            self.update_bounding_box_and_coloring(current_part)

    def on_right_button_press(self, obj, event):
        # Get the click position
        click_pos = self.interactor.GetEventPosition()

        # Use the picker to pick the position in the scene
        self.picker.Pick(click_pos[0], click_pos[1], 0, self.renderer)
        actor = self.picker.GetActor()

        if actor == self.shape_actor:
            cell_id = self.picker.GetCellId()
            if cell_id >= 0:
                cell_colors = self.shape_polydata.GetCellData().GetScalars()
                picked_color = cell_colors.GetTuple3(cell_id)

                # Check if the picked cell already has a color other than the default gray
                for part_index, part in enumerate(self.parts):
                    part_color = tuple(int(c * 255) for c in part.annotation_color)
                    if picked_color == part_color:
                        # Switch to the part that matches the picked color
                        self.hide_current_part()
                        self.current_part_index = part_index
                        self.load_current_part()
                        return

            # If no existing part matches, create a new part
            self.create_new_part()
            self.hide_current_part()

    def create_new_part(self):
        # Hide the current part before creating a new one
        self.hide_current_part()
        import random
        if len(self.parts) >= len(self.annotation_colors):
            annotation_color = (random.random(), random.random(), random.random())  # Generate a random color
        else:
            annotation_color = self.annotation_colors[len(self.parts) % len(self.annotation_colors)]
        new_part = PartAnnotation(annotation_color)
        self.parts.append(new_part)
        self.current_part_index = len(self.parts) - 1

    def hide_current_part(self):
        # Hide the current part (box and picked points)
        if self.current_part_index != -1:
            current_part = self.parts[self.current_part_index]
            if current_part.box_actor:
                current_part.box_actor.VisibilityOff()
            for point_actor in current_part.picked_points_actors:
                point_actor.VisibilityOff()
                self.renderer.RemoveActor(point_actor)  # Ensure the actor is removed from the scene

    def load_current_part(self):
        # Show the current part (box and picked points)
        if self.current_part_index != -1:
            current_part = self.parts[self.current_part_index]
            if current_part.box_actor:
                current_part.box_actor.VisibilityOn()
            for point_actor in current_part.picked_points_actors:
                self.renderer.AddActor(point_actor)  # Ensure the actor is added back to the scene
                point_actor.VisibilityOn()

    def update_annotation_color(self):
        # Update the color for the next part
        self.annotation_color = self.annotation_colors[len(self.parts) % len(self.annotation_colors)]

    def visualize_picked_point(self, position, color):
        # Create a small sphere to visualize the picked point
        sphere = vtk.vtkSphereSource()
        sphere.SetCenter(position)
        sphere.SetRadius(0.02)

        # Create a mapper and actor for the sphere
        sphere_mapper = vtk.vtkPolyDataMapper()
        sphere_mapper.SetInputConnection(sphere.GetOutputPort())
        sphere_actor = vtk.vtkActor()
        sphere_actor.SetMapper(sphere_mapper)
        sphere_actor.GetProperty().SetColor(*color)  # Set point color based on current part color

        # Add the sphere actor to the renderer
        self.renderer.AddActor(sphere_actor)

        # Render the scene again to show the new point
        self.render_window.Render()

    def on_key_press(self, obj, event):
        key = self.interactor.GetKeySym()
        if key == "Return":  # When the 'Enter' key is pressed
            # Save the bounding boxes for all parts
            self.save_all_obbs()
        elif key == "Escape":  # When the 'Escape' key is pressed
            # Clear all picked points, box, and color for the current part
            if self.current_part_index != -1:
                current_part = self.parts[self.current_part_index]
                for point_actor in current_part.picked_points_actors:
                    self.renderer.RemoveActor(point_actor)  # Ensure the picked point actor is removed from the scene
                current_part.picked_points_actors.clear()
                current_part.picked_points.clear()
                # Update colors
                self.update_bounding_box_and_coloring(current_part)
        elif key == "BackSpace":  # When the 'Backspace' key is pressed
            # Remove the last picked point and update the bounding box and coloring
            if self.current_part_index != -1 and len(self.parts[self.current_part_index].picked_points) > 0:
                current_part = self.parts[self.current_part_index]
                # Remove the last picked point actor from the renderer
                last_point_actor = current_part.picked_points_actors.pop()
                self.renderer.RemoveActor(last_point_actor)
                # Remove the last picked point from the list
                current_part.picked_points.pop()
                # Update the bounding box and coloring
                self.update_bounding_box_and_coloring(current_part)

    def update_bounding_box_and_coloring(self, part):
        # Update the bounding box and coloring based on the current picked points
        if len(part.picked_points) > 1:
            self.place_oriented_bounding_box(part)
        else:
            # Remove the box actor from the renderer if it exists
            if part.box_actor:
                self.renderer.RemoveActor(part.box_actor)
                part.box_actor = None
            # Update colored cells
            self.update_colored_cells(part)

    def place_oriented_bounding_box(self, part):
        # Convert points to a numpy array
        points_np = np.array(part.picked_points)

        # Use PCA to find the principal axes
        centroid = np.mean(points_np, axis=0)
        centered_points = points_np - centroid
        cov_matrix = np.cov(centered_points, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        sorted_indices = np.argsort(eig_vals)[::-1]
        eig_vecs = eig_vecs[:, sorted_indices]

        # Calculate the transformed points in the new basis
        transformed_points = centered_points.dot(eig_vecs)

        # Find the extents in the new basis (minimum bounding box)
        min_bounds = np.min(transformed_points, axis=0) - 0.02
        max_bounds = np.max(transformed_points, axis=0) + 0.02
        lengths = max_bounds - min_bounds
        center_in_new_basis = (max_bounds + min_bounds) / 2.0

        # Calculate the actual center in the original space
        center = centroid + center_in_new_basis.dot(eig_vecs.T)

        # Create an oriented bounding box
        box = vtk.vtkCubeSource()
        box.SetXLength(lengths[0])
        box.SetYLength(lengths[1])
        box.SetZLength(lengths[2])

        # Create a transform to apply the rotation and translation
        transform = vtk.vtkTransform()
        transform_matrix = vtk.vtkMatrix4x4()
        for i in range(3):
            for j in range(3):
                transform_matrix.SetElement(i, j, eig_vecs[i, j])
        transform_matrix.SetElement(0, 3, center[0])
        transform_matrix.SetElement(1, 3, center[1])
        transform_matrix.SetElement(2, 3, center[2])
        transform.SetMatrix(transform_matrix)

        # Apply the transformation to the box
        transform_filter = vtk.vtkTransformPolyDataFilter()
        transform_filter.SetInputConnection(box.GetOutputPort())
        transform_filter.SetTransform(transform)
        transform_filter.Update()

        # Create a mapper and actor for the oriented box
        box_mapper = vtk.vtkPolyDataMapper()
        box_mapper.SetInputConnection(transform_filter.GetOutputPort())
        box_actor = vtk.vtkActor()
        box_actor.SetMapper(box_mapper)
        box_actor.GetProperty().SetRepresentationToWireframe()  # Set box to wireframe representation
        box_actor.GetProperty().SetColor(*part.annotation_color)  # Set box color based on current part color
        box_actor.GetProperty().SetAmbient(0.3)
        box_actor.GetProperty().SetDiffuse(0.6)
        box_actor.GetProperty().SetSpecular(0.8)
        box_actor.GetProperty().SetSpecularPower(20.0)

        # Store the bounding box information in part
        part.box_info = {
            "center": center.tolist(),
            "lengths": lengths.tolist(),
            "eigenvectors": eig_vecs.T.tolist()
        }

        # Remove the existing box actor if any
        if part.box_actor is not None:
            self.renderer.RemoveActor(part.box_actor)

        # Set the box actor to be non-pickable
        box_actor.PickableOff()

        # Add the new box actor to the renderer
        part.box_actor = box_actor
        self.renderer.AddActor(part.box_actor)

        # Color the internal faces of the shape that are inside the bounding box
        self.update_colored_cells(part)

        # Render the scene again to show the new bounding box
        self.render_window.Render()

    def update_colored_cells(self, current_part):
        # Clear the current part's colored cells
        current_part.colored_cells.clear()

        # If there is a bounding box, update the colored cells for the current part
        if current_part.box_actor is not None:
            enclosed_points_filter = vtk.vtkSelectEnclosedPoints()
            enclosed_points_filter.SetInputData(self.shape_polydata)
            enclosed_points_filter.SetSurfaceData(current_part.box_actor.GetMapper().GetInput())
            enclosed_points_filter.Update()

            for i in range(self.shape_polydata.GetNumberOfCells()):
                cell = self.shape_polydata.GetCell(i)
                cell_points = cell.GetPoints()
                all_inside = True
                for j in range(cell_points.GetNumberOfPoints()):
                    point_id = cell.GetPointId(j)
                    if enclosed_points_filter.IsInside(point_id) == 0:
                        all_inside = False
                        break
                if all_inside:
                    current_part.colored_cells.append(i)

        # Reset all cells to default color
        cell_colors = self.shape_polydata.GetCellData().GetScalars()
        for i in range(self.shape_polydata.GetNumberOfCells()):
            cell_colors.SetTuple3(i, 200, 200, 200)

        # Apply the coloring for all parts, ensuring the current part is processed last
        for idx, part in enumerate(self.parts):
            if part != current_part:
                for cell_id in part.colored_cells:
                    cell_colors.SetTuple3(cell_id, int(part.annotation_color[0] * 255), int(part.annotation_color[1] * 255), int(part.annotation_color[2] * 255))

        # Finally, apply the color for the current part
        for cell_id in current_part.colored_cells:
            cell_colors.SetTuple3(cell_id, int(current_part.annotation_color[0] * 255), int(current_part.annotation_color[1] * 255), int(current_part.annotation_color[2] * 255))

        # Update the mapper and actor to reflect the changes
        self.shape_polydata.Modified()
        self.shape_actor.GetMapper().Update()

        # Render the scene again to show the updated colors
        self.render_window.Render()

    def save_all_obbs(self):
        # Save the bounding box data to a JSON file
        obb_data = []
        for part in self.parts:
            if part.box_info is not None:
                # Store the OBB information
                obb_data.append({
                    "part_index": self.parts.index(part),
                    "center": part.box_info["center"],
                    "lengths": part.box_info["lengths"],
                    "eigenvectors": part.box_info["eigenvectors"]
                })
        # Write to file
        with open(self.save_path, 'w') as f:
            json.dump(obb_data, f, indent=4)

if __name__ == "__main__":
    shape_filename = r"C:\code\noodles\mesh\fold_chair.obj"
    save_path = r"C:\code\noodles\mesh\fold_chair.json"
    PartAnnotationTool(shape_filename, save_path)
