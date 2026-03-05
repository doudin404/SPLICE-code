import pickle

def load_pkl_file(file_path):
    """
    从给定路径读取 pkl 文件并返回其中的内容。

    参数:
        file_path (str): .pkl 文件的路径。

    返回:
        obj: 反序列化后的 Python 对象。
    """
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    return data

# 使用示例
file_path = r'C:\code\spaghetti-master\assets\checkpoints\spaghetti_chairs_large\samples\occ\4470.pkl'  # 替换为你的文件路径
data = load_pkl_file(file_path)
print(data)