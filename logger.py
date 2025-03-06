import logging
import platform

system = platform.system()
if system == "Windows":
    from colorama import init
    init(autoreset=True)



COLORS = {
    "DEBUG": "\033[36m",    # 青色
    "INFO": "\033[32m",     # 绿色
    "WARNING": "\033[33m",  # 黄色
    "ERROR": "\033[31m",    # 红色
    "CRITICAL": "\033[31;1m",  # 红色加粗
    "RESET": "\033[0m"      # 重置颜色
}

class ColoredFormatter(logging.Formatter):
    def format(self, record):
        # 获取原始日志消息
        message = super().format(record)
        # 根据日志级别添加颜色
        color = COLORS.get(record.levelname, COLORS["RESET"])
        return f"{color}{message}{COLORS['RESET']}"

def get_logger(name):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = ColoredFormatter(
        "[%(levelname)s]{%(name)s}%(asctime)s: %(message)s",
        datefmt="%Y-%m-%d %I:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger