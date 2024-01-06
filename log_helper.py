import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime

class LogHelper:
    def __init__(self):
        # 获取当前的日期和时间
        current_datetime = datetime.now()
        formatted_datetime = current_datetime.strftime("%Y_%m_%d")
        # 配置日志记录器
        logger = logging.getLogger('my_logger')
        logger.setLevel(logging.DEBUG)

        # 创建一个 RotatingFileHandler 实例，将日志保存到文件中
        log_file = 'run_result_{}.txt'.format(formatted_datetime)
        handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024, backupCount=5)
        handler.setLevel(logging.DEBUG)

        # 设置日志格式
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        # 将处理器添加到日志记录器中
        logger.addHandler(handler)
        self.loger = logger

    def log(self, message):
        print(message)
        self.loger.info(message)

# 示例：往日志中写入一条消息

