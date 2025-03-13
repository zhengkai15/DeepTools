# from .utils import create_logger
# from .config import get_color_codes

# logger = create_logger()
# 可以在调用时配置路径，其他子模块中的的from loguru import logger的logger可以打印和写出到log文件
# 写在init中
# logger = deepsuit.create_logger(log_level="INFO", log_path="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/zhengkai/zhengkai_dev")
# config = load_yaml("/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/zhengkai/zhengkai_dev/sais-race.2024/demo_review_1h_log/conf/config.ymal")

# 不写在init中, 同样可以
# from deepsuit.utils import create_logger
# logger = create_logger(log_level="INFO", log_path="/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/zhengkai/zhengkai_dev")


# BG_RED, BG_GREEN, BG_YELLOW, BG_BLUE, BG_RESET = get_color_codes()
