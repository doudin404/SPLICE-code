'''
./ManifoldPlus --input input.obj --output output.obj --depth 8
请写一个类以能调用上面的程序,
类初始化时输入./ManifoldPlus程序的路径
调用时输入--input和--output路径和--depth,--depth可以不输入,默认为8
'''
import subprocess
from typing import Optional

class ManifoldPlusRunner:
    """
    这个类用于调用ManifoldPlus程序.

    初始化参数:
    - manifold_plus_path: ManifoldPlus程序的路径

    调用方法:
    - run(input_path: str, output_path: str, depth: Optional[int] = 8)
    """

    def __init__(self, manifold_plus_path: str):
        self.manifold_plus_path = manifold_plus_path

    def run(self, input_path: str, output_path: str, depth: Optional[int] = 8):
        """
        调用ManifoldPlus程序处理输入文件.

        参数:
        - input_path: 输入文件的路径
        - output_path: 输出文件的路径
        - depth: 可选, 默认为8
        """
        command = [
            self.manifold_plus_path,
            "--input", input_path.replace("\\", "/"),
            "--output", output_path.replace("\\", "/"),
            "--depth", str(depth)
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running ManifoldPlus: {e}")
        except FileNotFoundError as e:
            print(f"The specified ManifoldPlus executable was not found: {e}")