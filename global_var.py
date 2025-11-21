from utils.parse import process_config
from utils.getArgs import getArgs

global myModelConfig

# 模型参数配置
try:
    args = getArgs()
    print("args got")
    myModelConfig = process_config(args.config)

except Exception:
    raise ValueError("missing or invalid arguments")
