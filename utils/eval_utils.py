import os
import json
import glob
from pathlib import Path
from typing import Callable

import numpy as np
import open3d as o3d
from tqdm import tqdm


def load_and_align_pcd_mesh(pcd_path: str,
                            mesh_path: str,
                            rotation_vector: tuple = None,
                            visualize_mesh: bool = False) -> tuple:
    """
    加载和对齐点云与网格
    GT: 点云
    GEN: 网格

    Args:
        pcd_path: 点云文件路径（支持 .pcd/.ply/.xyz 等格式）
        mesh_path: 网格文件路径（支持 .obj/.ply 等格式）
        rotation_vector: 旋转向量，格式为 (rx, ry, rz)，表示绕 X、Y、Z 轴的旋转角度（弧度），默认为 None
        visualize: 是否可视化对齐结果，默认 False

    Returns:
        tuple: (aligned_pcd, aligned_mesh) 对齐后的点云和网格
    """
    # 加载点云和网格
    np_gt_point_cloud = np.load(pcd_path)
    np_gt_point_cloud = np_gt_point_cloud['points']
    gt_point_cloud = o3d.geometry.PointCloud()
    gt_point_cloud.points = o3d.utility.Vector3dVector(np_gt_point_cloud)
    gt_point_cloud = gt_point_cloud.random_down_sample(0.3)
    pcd = gt_point_cloud
    mesh = o3d.io.read_triangle_mesh(mesh_path)

    # 根据旋转向量旋转点云
    if rotation_vector is not None:
        R = o3d.geometry.get_rotation_matrix_from_xyz(rotation_vector)
        pcd.rotate(R, center=(0, 0, 0))

    # 对齐质心
    center_pcd = pcd.get_center()
    center_mesh = mesh.get_center()
    pcd.translate(-center_pcd)
    mesh.translate(-center_mesh)

    # 按各自最大轴长归一化
    bbox_pcd = pcd.get_axis_aligned_bounding_box()
    extents_pcd = bbox_pcd.get_max_bound() - bbox_pcd.get_min_bound()
    max_extent_pcd = np.max(extents_pcd) if np.any(extents_pcd) else 0.0
    bbox_mesh = mesh.get_axis_aligned_bounding_box()
    extents_mesh = bbox_mesh.get_max_bound() - bbox_mesh.get_min_bound()
    max_extent_mesh = np.max(extents_mesh) if np.any(extents_mesh) else 0.0
    scale_pcd = 1.0 / max_extent_pcd if max_extent_pcd > 0 else 1.0
    scale_mesh = 1.0 / max_extent_mesh if max_extent_mesh > 0 else 1.0
    pcd.scale(scale_pcd, center=(0, 0, 0))
    mesh.scale(scale_mesh, center=(0, 0, 0))

    # ICP 微调：使用网格采样点与点云进行配准
    mesh_sampled = mesh.sample_points_uniformly(number_of_points=5000)
    pcd_icp = pcd.voxel_down_sample(voxel_size=(max_extent_pcd / 100.0) if max_extent_pcd > 0 else 0.01)
    reg = o3d.pipelines.registration.registration_icp(
        pcd_icp, mesh_sampled,
        max_correspondence_distance=0.02,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    trans = reg.transformation
    pcd.transform(trans)

    # 可视化对齐结果
    if visualize_mesh:
        pcd.paint_uniform_color([0, 1, 0])
        mesh.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([pcd, mesh],
                                          window_name=f"pcd_mesh id: {Path(pcd_path).stem}")

    return pcd, mesh


def load_and_align_meshes(gen_mesh_path: str,
                         gt_mesh_path: str,
                         rotation_vector: tuple = None,
                         visualize_mesh: bool = False) -> tuple:
    """
    加载和对齐两个网格
    
    Args:
        gen_mesh_path : 生成网格文件路径（支持 .obj/.ply 等格式）
        gt_mesh_path  : 真值网格文件路径
        rotation_vector : 旋转向量，格式为 (rx, ry, rz)，表示绕 X、Y、Z 轴的旋转角度（弧度），默认为 None
        visualize_mesh: 是否可视化对齐结果，默认 False
        
    Returns:
        tuple: (gen_mesh, gt_mesh_static) 对齐后的生成网格和真值网格
    """
    # 加载网格
    gen_mesh = o3d.io.read_triangle_mesh(gen_mesh_path)
    gt_mesh_static = o3d.io.read_triangle_mesh(gt_mesh_path)

    # 根据旋转向量旋转生成网格
    if rotation_vector is not None:
        R = o3d.geometry.get_rotation_matrix_from_xyz(rotation_vector)
        gen_mesh.rotate(R, center=(0, 0, 0))

    # 对齐生成网格和真值网格到同一坐标系
    # 1. 对齐质心
    center_gen = gen_mesh.get_center()
    center_gt = gt_mesh_static.get_center()
    gen_mesh.translate(-center_gen)
    gt_mesh_static.translate(-center_gt)

    # 2. 按各自最大轴长归一化
    # 计算两个网格各自的最大轴长并归一化到 1
    bbox_gen = gen_mesh.get_axis_aligned_bounding_box()
    extents_gen = bbox_gen.get_max_bound() - bbox_gen.get_min_bound()
    max_extent_gen = np.max(extents_gen) if np.any(extents_gen) else 0.0
    bbox_gt = gt_mesh_static.get_axis_aligned_bounding_box()
    extents_gt = bbox_gt.get_max_bound() - bbox_gt.get_min_bound()
    max_extent_gt = np.max(extents_gt) if np.any(extents_gt) else 0.0
    scale_gen = 1.0 / max_extent_gen if max_extent_gen > 0 else 1.0
    scale_gt = 1.0 / max_extent_gt if max_extent_gt > 0 else 1.0
    gen_mesh.scale(scale_gen, center=(0, 0, 0))
    gt_mesh_static.scale(scale_gt, center=(0, 0, 0))

    # 3. ICP 微调
    pcd_gen_icp = gen_mesh.sample_points_uniformly(number_of_points=5000)
    pcd_gt_icp = gt_mesh_static.sample_points_uniformly(number_of_points=5000)
    reg = o3d.pipelines.registration.registration_icp(
        pcd_gen_icp, pcd_gt_icp,
        max_correspondence_distance=0.02,
        init=np.eye(4),
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    trans = reg.transformation
    gen_mesh.transform(trans)

    # 可视化微调后的对齐结果
    if visualize_mesh:
        gen_mesh.paint_uniform_color([1, 0, 0])
        gt_mesh_static.paint_uniform_color([0, 1, 0])
        coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([gen_mesh, gt_mesh_static, coordinate_frame],
                                        window_name=f"mesh id: {Path(gen_mesh_path).stem}")
    
    return gen_mesh, gt_mesh_static


def compute_statistics(data_values: list) -> tuple:
    """
    计算一组数据的基本统计量
    
    Args:
        data_values: 浮点数列表，原始尺度的测量值
        
    Returns:
        tuple: (avg, var, median, min, max, 放大1000倍后的这些值)
    """
    if not data_values:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
        
    # 基本统计量
    avg = np.mean(data_values)
    var = np.var(data_values)
    median = np.median(data_values)
    min_val = np.min(data_values)
    max_val = np.max(data_values)
    
    # 放大后的值
    avg_scaled = avg * 1000
    var_scaled = var * 1000 * 1000  # 方差需要平方放大
    median_scaled = median * 1000
    min_scaled = min_val * 1000
    max_scaled = max_val * 1000
    
    return avg, var, median, min_val, max_val, avg_scaled, var_scaled, median_scaled, min_scaled, max_scaled


def filter_outliers(data_values: list, threshold: float) -> tuple:
    """
    根据阈值过滤掉异常值并计算过滤后的统计量
    
    Args:
        data_values: 原始数据列表
        threshold: 过滤阈值，大于此值的数据将被过滤
        
    Returns:
        tuple: (filtered_stats, filtered_count, removed_count)
            - filtered_stats: 包含过滤后统计量的元组
            - filtered_count: 保留的数据点数
            - removed_count: 移除的数据点数
    """
    if not data_values or threshold is None:
        return compute_statistics([]), 0, 0
        
    print(f"Removing outliers with threshold: {threshold}")
    filtered_values = [val for val in data_values if val <= threshold]
    filtered_count = len(filtered_values)
    removed_count = len(data_values) - filtered_count
    
    if filtered_count == 0:
        print(f"\nNo values below threshold {threshold}")
        return compute_statistics([]), 0, removed_count
        
    stats = compute_statistics(filtered_values)
    
    print(f"\nFiltered statistics (threshold={threshold}, removed {removed_count} outliers):")
    print(f"   Average:  {stats[0]:<12.6f} / {stats[5]:<12.6f}")
    print(f"   Variance: {stats[1]:<12.6f} / {stats[6]:<12.6f}")
    print(f"   Median:   {stats[2]:<12.6f} / {stats[7]:<12.6f}")
    print(f"   Minimum:  {stats[3]:<12.6f} / {stats[8]:<12.6f}")
    print(f"   Maximum:  {stats[4]:<12.6f} / {stats[9]:<12.6f}")
    print(f"   Count:    {filtered_count} (kept) / {removed_count} (removed)")
    
    return stats, filtered_count, removed_count


def save_results_to_json(results: dict, stats: tuple, filtered_stats: tuple, counts: tuple, output_json: str) -> dict:
    """
    将结果和统计数据保存为JSON格式
    
    Args:
        results: 每个模型的结果字典
        stats: 原始统计量元组 (avg, var, median, min, max, 放大后的值...)
        filtered_stats: 过滤后的统计量元组
        counts: (总数, 过滤后数量, 移除数量, 阈值)的元组
        output_json: 输出文件路径
        
    Returns:
        dict: 输出的数据结构
    """
    count, filtered_count, removed_count, threshold = counts
    
    output_data = {
        "stats": {
            "avg": stats[5],  # 放大后的均值
            "var": stats[6],  # 放大后的方差
            "median": stats[7],  # 放大后的中位数
            "min": stats[8],  # 放大后的最小值
            "max": stats[9],  # 放大后的最大值
            "count": count,
            "original": {
                "avg": float(stats[0]),
                "var": float(stats[1]),
                "median": float(stats[2]),
                "min": float(stats[3]),
                "max": float(stats[4])
            }
        },
        "filtered_stats": {
            "avg": filtered_stats[5],
            "var": filtered_stats[6],
            "median": filtered_stats[7],
            "min": filtered_stats[8],
            "max": filtered_stats[9],
            "count": filtered_count,
            "removed_count": removed_count,
            "threshold": float(threshold) if threshold is not None else None,
            "original": {
                "avg": float(filtered_stats[0]),
                "var": float(filtered_stats[1]),
                "median": float(filtered_stats[2]),
                "min": float(filtered_stats[3]),
                "max": float(filtered_stats[4])
            }
        },
        "results": results
    }
    
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    print(f"Results saved to {output_json}")
    
    return output_data


def batch_compute_metrics(
    root_dir: str,
    compute_metric_func: Callable,
    output_json: str = "metrics_results.json",
    rotation_vector: tuple = None,
    visualize_mesh: bool = False,
    outlier_threshold: float = None,
    override_gen_dir: str = None,
    override_gt_dir: str = None,
    max_compute_num: int = -1,
    gen_file_type: str = "obj",
    gt_file_type: str = "obj",
    **metric_kwargs
) -> dict:
    """
    批量计算文件夹下 gt 和 gen 子文件夹中所有同名网格的指定度量
    
    Args:
        root_dir: 根目录，包含 gt 和 gen 两个子文件夹
        compute_metric_func: 用于计算度量的函数
        output_json: 输出的 JSON 结果文件路径
        rotation_vector: 旋转向量，格式为 (rx, ry, rz)，表示绕 X、Y、Z 轴的旋转角度（弧度），默认为 None
        visualize_mesh: 是否可视化对齐结果
        outlier_threshold: 异常值阈值，大于该值的数据将被排除在统计之外
        override_gen_dir: 覆盖 gen 文件夹路径
        override_gt_dir: 覆盖 gt 文件夹路径
        **metric_kwargs: 传递给计算度量函数的额外参数
        
    Returns:
        dict: 包含每个模型度量值和统计数据的字典
    """
    if override_gen_dir:
        gen_dir = override_gen_dir
    else:
        gen_dir = os.path.join(root_dir, 'gen')

    if override_gt_dir:
        gt_dir = override_gt_dir
    else:
        gt_dir = os.path.join(root_dir, 'gt')

    if not os.path.exists(gt_dir) or not os.path.exists(gen_dir):
        raise ValueError(f"Directory structure error, please ensure the existence of {gt_dir} and {gen_dir} directories")

    # 获取所有 gt 网格文件
    if gt_file_type == "obj":
        gt_files = glob.glob(os.path.join(gt_dir, '*.obj'))
    elif gt_file_type == "ply":
        gt_files = glob.glob(os.path.join(gt_dir, '*.ply'))
    elif gt_file_type == "npz":
        gt_files = glob.glob(os.path.join(gt_dir, '*.npz'))
    else:
        raise ValueError(f"Unsupported file type: {gt_file_type}")
    gt_basenames = {Path(f).stem: f for f in gt_files}

    # 获取所有 gen 网格文件
    if gen_file_type == "obj":
        gen_files = glob.glob(os.path.join(gen_dir, '*.obj'))
    elif gen_file_type == "ply":
        gen_files = glob.glob(os.path.join(gen_dir, '*.ply'))
    elif gen_file_type == "npz":
        gen_files = glob.glob(os.path.join(gen_dir, '*.npz'))
    else:
        raise ValueError(f"Unsupported file type: {gen_file_type}")
    gen_basenames = {Path(f).stem: f for f in gen_files}

    # 找到同名文件并计算度量
    results = {}
    common_ids = set(gt_basenames.keys()) & set(gen_basenames.keys())

    if not common_ids:
        raise ValueError("Can't find any matched GT/GEN file pairs, please check the file naming format")

    total_metric = 0.0
    count = 0

    print(f"Found {len(common_ids)} pairs of matched mesh files")

    # 只计算前 n 个
    if max_compute_num > 0:
        common_ids = list(common_ids)[:min(max_compute_num, len(common_ids))]
        print(f"Only compute the first {min(max_compute_num, len(common_ids))} pairs of matched mesh files")

    
    pbar = tqdm(sorted(common_ids))
    # 用于计算当前均值的累积变量
    running_sum = 0.0
    valid_count = 0
    for model_id in pbar:
        gt_file = gt_basenames[model_id]
        gen_file = gen_basenames[model_id]

        try:
            metric_value = compute_metric_func(
                gen_file, gt_file, 
                rotation_vector=rotation_vector,
                visualize_mesh=visualize_mesh,
                **metric_kwargs
            )
            metric_value_scaled = float(metric_value) * 1000  # 结果放大 1000 倍
            results[model_id] = metric_value_scaled
            total_metric += metric_value
            count += 1
            
            # 计算当前均值
            running_sum += metric_value
            valid_count += 1
            current_avg = running_sum / valid_count
            current_avg_scaled = current_avg * 1000
            
            pbar.set_postfix({
                "id": model_id, 
                "value": f"{metric_value_scaled:.6f}",
                "current_avg": f"{current_avg_scaled:.6f}"
            })
        except Exception as e:
            print(f"Error processing {model_id}: {str(e)}")
            results[model_id] = "Processing failed"

    # 收集有效的值用于统计计算
    valid_values = []
    for model_id, value in results.items():
        if isinstance(value, (int, float)):
            valid_values.append(float(value)/1000.0)  # 转回原始值进行统计
    
    # 计算基本统计量并打印
    stats = compute_statistics(valid_values)
    if valid_values:
        print(f"Statistics (original value/multiplied by 1000):")
        print(f"   Average:  {stats[0]:<12.6f} / {stats[5]:<12.6f}")
        print(f"   Variance: {stats[1]:<12.6f} / {stats[6]:<12.6f}")
        print(f"   Median:   {stats[2]:<12.6f} / {stats[7]:<12.6f}")
        print(f"   Minimum:  {stats[3]:<12.6f} / {stats[8]:<12.6f}")
        print(f"   Maximum:  {stats[4]:<12.6f} / {stats[9]:<12.6f}")
        print(f"   Count:    {count} (kept)")
    else:
        print("No valid metric values for statistical calculation")
    
    # 过滤异常值并计算过滤后的统计量
    threshold = outlier_threshold
    filtered_stats, filtered_count, removed_count = filter_outliers(valid_values, threshold)
    
    # 保存结果到 JSON 文件
    counts = (count, filtered_count, removed_count, threshold)
    output_data = save_results_to_json(results, stats, filtered_stats, counts, output_json)
    
    return output_data


def parse_rotation_vector(rotation_vector_input, parser=None):
    """
    解析旋转向量，支持字符串格式和元组格式的输入
    
    Args:
        rotation_vector_input: 旋转向量输入，可以是以下格式：
            - 字符串："rx,ry,rz" 或 "rx, ry, rz"，表示绕 X、Y、Z 轴的旋转角度（弧度）
            - 元组或列表：(rx, ry, rz) 或 [rx, ry, rz]
            - 包含 numpy 表达式的字符串："(0, np.pi, 0)"
            - None: 不进行旋转
        parser: argparse.ArgumentParser 实例，如果提供，则使用 parser.error 抛出错误
        
    Returns:
        tuple: (rx, ry, rz) 旋转向量，如果输入为 None 则返回 None
        
    Raises:
        ValueError: 如果解析失败且未提供 parser
    """
    # 如果输入为 None，直接返回 None
    if rotation_vector_input is None:
        return None
    
    try:
        # 处理已经是元组或列表的情况
        if isinstance(rotation_vector_input, (tuple, list)):
            rotation_vector = tuple(map(float, rotation_vector_input))
        # 处理字符串情况
        elif isinstance(rotation_vector_input, str):
            rotation_vector_str = rotation_vector_input.strip()
            # 尝试解析是否是 Python 风格元组字符串
            if rotation_vector_str.startswith('(') and rotation_vector_str.endswith(')'):
                # 移除括号并解析元素
                content = rotation_vector_str[1:-1]
                # 处理可能包含 np.pi 等表达式
                import numpy as np
                # 创建包含 numpy 的局部命名空间
                local_namespace = {'np': np}
                # 使用 eval 求值，但在隔离的环境中
                elements = eval(f"[{content}]", {"__builtins__": {}}, local_namespace)
                rotation_vector = tuple(elements)
            else:
                # 普通的逗号分隔字符串，处理空格
                values = [x.strip() for x in rotation_vector_str.split(',')]
                rotation_vector = tuple(map(float, values))
        else:
            raise ValueError(f"旋转向量输入格式错误：必须是字符串、元组或列表，而不是 {type(rotation_vector_input)}")
        
        # 验证长度
        if len(rotation_vector) != 3:
            raise ValueError("旋转向量必须包含 3 个值 (rx,ry,rz)")
        
        print(f"rotation_vector: {rotation_vector}")

        return rotation_vector
    except Exception as e:
        if parser:
            parser.error(f"旋转向量格式错误: {e}")
        else:
            raise ValueError(f"旋转向量格式错误: {e}")
