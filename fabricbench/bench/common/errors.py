from enum import Enum, auto

class PlannerErrorCode(Enum):
    # --- 通用状态 ---
    NONE = 0                 # 无错误 / 成功
    UNKNOWN = 1              # 未知错误

    # --- 输入检查错误 ---
    INVALID_INPUT = 10       # 输入参数无效（如维度不对）
    TARGET_OUT_OF_LIMITS = 11 # 目标点超出关节限位

    # --- 运动学错误 (IK) ---
    IK_FAILED_LEFT = 20      # 左臂 IK 无解
    IK_FAILED_RIGHT = 21     # 右臂 IK 无解
    IK_FAILED_GLOBAL = 22    # IK 求解失败 (单臂或通用)

    # --- 碰撞检测错误 ---
    COLLISION_SELF = 30      # 自碰撞
    COLLISION_SCENE = 31     # 环境碰撞
    
    # --- 规划算法错误 ---
    PLANNING_TIMEOUT = 40    # 规划超时
    NO_PATH_FOUND = 41       # 路径搜索失败（如RRT没找到路）
    TIME_PARAM_FAILED = 42   # 时间参数化失败 (Toppra 报错)
    DUAL_ARM_PLAN_FAIlED= 43

    # --- 系统/后端错误 ---
    BACKEND_UNAVAILABLE = 50 # 规划器后端未加载 (如 Curobo 没装)