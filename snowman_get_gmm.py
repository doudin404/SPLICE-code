import os

import shape_process.visualizationManager
import torch


def save_gmm(path, gmm_name):
    with open(path, "r") as f:
        lines = f.readlines()

    data_as_lists = [list(map(float, line.split())) for line in lines]
    centroids = data_as_lists[0]
    scaleing_factors = data_as_lists[1]
    orientations = data_as_lists[2]
    valid_list = data_as_lists[3]

    valid_num = valid_list.count(1)
    for i in range(0, valid_num):
        centroid = centroids[i * 3 : i * 3 + 3]
        scaleing_factor = scaleing_factors[i * 3 : i * 3 + 3]
        orientation = orientations[i * 9 : i * 9 + 9]

        scale_num = 2.0
        scaleing_factor = [x * scale_num for x in scaleing_factor]

        gmm = centroid + scaleing_factor + orientation
        gmm = torch.tensor(gmm)

        print(f"gmm: {gmm}")
        transformed_sphere = visman.add_gmm(gmm)
        transformed_sphere.save(f"./gmm_mesh/{gmm_name}_gmm_{i}.obj")

    pass

def main():
    gmm_param_path = "./gmm_params/"
    
    gmm_files = [f for f in os.listdir(gmm_param_path) if f.endswith((".txt"))]

    for gmm_file in gmm_files:
        gmm_param_path = os.path.join(gmm_param_path, gmm_file)
        save_gmm(gmm_param_path, gmm_file.split(".")[0])
    


if __name__ == "__main__":
    visman = shape_process.visualizationManager.VisualizationManager()
    main()
