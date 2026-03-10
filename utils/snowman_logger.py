import logging
import os
import time
from datetime import datetime
import sys
import traceback

LOG_PREFIX = "splice"
LOG_DIR = "./logs/splice/"
LOG_LEVEL = logging.DEBUG
USE_SINGLE_LOG_FILE = False # 控制是否所有日志都输出到同一个文件
OVERWRITE_SINGLE_LOG_FILE_WHEN_ENABLED = True # 控制单文件时是否覆盖
OUTPUT_TO_CONSOLE = True # 控制是否将日志输出到控制台

class LogManager:
    """日志管理类，负责初始化和管理日志系统。
    
    该类封装了日志系统的初始化、配置和管理功能，提供带标签的日志记录功能。
    
    Attributes:
        log_dir: 日志存储目录路径。
        log_file: 当前日志文件的完整路径。
        base_logger: 底层的 logger 对象。
        formatter: 日志格式化器。
        logger: 默认的带标签日志适配器。
    """
    
    def __init__(self, log_dir_name=LOG_DIR):
        """初始化日志系统。
        
        Args:
            log_dir_name: 日志目录路径。
        """
        # 创建日志目录
        self.log_dir = log_dir_name
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 创建带时间戳的日志文件名
        if USE_SINGLE_LOG_FILE:
            self.log_file = os.path.join(self.log_dir, f"{LOG_PREFIX}-main.log")
        else:
            today_str = datetime.now().strftime("%Y-%m-%d")
            timestamp_sec = int(time.time())
            self.log_file = os.path.join(self.log_dir, f"{LOG_PREFIX}-{today_str}-{timestamp_sec}.log")
        
        # 初始化基础日志记录器
        self.base_logger = logging.getLogger(LOG_PREFIX)
        self.base_logger.setLevel(LOG_LEVEL)

        # 阻止向上级（root）传播，避免重复输出
        self.base_logger.propagate = False
        
        # 创建日志格式，包含标签字段
        self.formatter = logging.Formatter("%(asctime)s - %(levelname)s - [%(tag)s] - %(message)s")
        
        # 添加处理器
        self._setup_handlers()
        
        # 创建并暴露主日志记录器
        self.logger = self._create_main_logger()
    
    def _setup_handlers(self):
        """设置日志处理器。
        
        为基础日志记录器添加文件和控制台输出处理器。
        """
        # 避免重复添加 handler
        if not self.base_logger.handlers:
            # 文件输出
            file_open_mode = 'a' # 默认追加
            if USE_SINGLE_LOG_FILE and OVERWRITE_SINGLE_LOG_FILE_WHEN_ENABLED:
                file_open_mode = 'w' # 如果启用单文件且设置了覆盖，则使用写入模式
            file_handler = logging.FileHandler(self.log_file, mode=file_open_mode, encoding="utf-8")
            file_handler.setFormatter(self.formatter)
            self.base_logger.addHandler(file_handler)
            
            # 控制台输出
            if OUTPUT_TO_CONSOLE:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(self.formatter)
                self.base_logger.addHandler(console_handler)
    
    def _create_main_logger(self):
        """创建主日志记录器。
        
        Returns:
            TaggedLoggerAdapter: 使用默认标签的日志适配器实例。
        """
        return TaggedLoggerAdapter(self.base_logger, "DEFAULT")
    
    def create_tagged_logger(self, tag):
        """创建一个带有指定标签的日志记录器。
        
        Args:
            tag: 要使用的标签字符串。
        
        Returns:
            TaggedLoggerAdapter: 带有指定标签的日志适配器。
        """
        return TaggedLoggerAdapter(self.base_logger, tag)


class TaggedLoggerAdapter(logging.LoggerAdapter):
    """带有标签的日志适配器，允许在日志中添加标签信息。
    
    这个适配器扩展了标准的日志适配器，自动将标签添加到日志记录中。
    
    Attributes:
        extra: 包含标签信息的字典。
    """
    
    def __init__(self, logger, tag=""):
        """初始化适配器。
        
        Args:
            logger: 基础logger对象。
            tag: 默认标签，可以为空字符串。
        """
        super().__init__(logger, {"tag": tag})
    
    def process(self, msg, kwargs):
        """处理日志记录，确保包含标签信息。
        
        在记录日志时被调用，确保每条日志都包含标签信息。
        
        Args:
            msg: 日志消息字符串。
            kwargs: 关键字参数字典。
        
        Returns:
            tuple: (msg, kwargs) 元组，包含处理后的消息和参数。
        """
        # 如果在额外参数中提供了标签，则使用提供的标签
        if "extra" not in kwargs:
            kwargs["extra"] = self.extra
        elif "tag" not in kwargs["extra"]:
            kwargs["extra"]["tag"] = self.extra["tag"]
        return msg, kwargs
    
    def set_tag(self, tag):
        """设置该适配器的默认标签。
        
        Args:
            tag: 要设置的标签字符串。
        """
        self.extra["tag"] = tag


def get_detailed_error(e: Exception) -> tuple[str, str]:
    """获取详细的异常信息，包括具体的文件和行号。
    
    此函数从异常对象中提取堆栈跟踪信息，生成格式化的错误详情。
    
    Args:
        e: 要处理的异常对象。
        
    Returns:
        tuple: 包含两个字符串的元组。
            - 第一个元素是格式化的错误详情，包含文件名和行号。
            - 第二个元素是完整的堆栈跟踪信息。
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    tb_details = traceback.extract_tb(exc_traceback)
    
    # 构建详细错误信息
    if tb_details:
        # 获取最后一个堆栈帧（通常是错误发生的地方）
        last_frame = tb_details[-1]
        file_name = os.path.basename(last_frame.filename)
        line_no = last_frame.lineno
        error_detail = f"{str(e)} (在 {file_name}:{line_no})"
        
        # 如果需要完整堆栈跟踪
        full_traceback = ''.join(traceback.format_exception(exc_type, exc_value, exc_traceback))
        return error_detail, full_traceback
    
    return str(e), traceback.format_exc()


# 创建日志系统实例
log_manager = LogManager()

# 导出主要的日志记录器和工具函数，保持原有接口不变
logger = log_manager.logger
create_tagged_logger = log_manager.create_tagged_logger

"""
# 使用说明

# 使用默认日志记录器
from splice_logger import logger
logger.info("系统消息")  # 使用默认标签 DEFAULT

# 使用自定义标签
logger.warning("警告", extra={"tag": "ALERT"})

# 创建特定模块的日志记录器
from splice_logger import create_tagged_logger
ai_logger = create_tagged_logger("AI")
ai_logger.info("AI模块启动")
"""

