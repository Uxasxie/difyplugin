"""
插件公共工具函数

将凭据解析逻辑放在此独立模块中，避免 provider 和 tools 之间的循环导入。
provider/modelscope_provider.py 和 tools/*.py 都从这里导入。
"""
from typing import Any


def get_base_url(credentials: dict[str, Any]) -> str:
    """
    从凭据中获取 API 网关根地址（不包含 /v1 前缀），如果未配置则返回 ModelScope 默认地址
    
    用户可能填写以下格式：
    - http://10.17.68.1:40034/       （不含 /v1，正确）
    - http://10.17.68.1:40034/v1    （含 /v1，需要剥离）
    - http://10.17.68.1:40034/v1/  （含 /v1/，需要剥离）
    
    统一返回不含 /v1 的根地址，确保后续拼接 v1/images/generations 正确
    """
    base_url = credentials.get("base_url", "").strip()
    if not base_url:
        base_url = "https://api-inference.modelscope.cn/"
    
    # 剥离末尾的 /v1 或 /v1/，避免拼接时出现 /v1/v1/ 的问题
    while True:
        stripped = base_url.rstrip("/")
        if stripped.endswith("/v1"):
            base_url = stripped[:-3] + "/"
        else:
            break
    
    # 确保以 / 结尾，避免 URL 拼接问题
    if not base_url.endswith("/"):
        base_url += "/"
    
    return base_url


def get_model(credentials: dict[str, Any], tool_default: str = "") -> str:
    """
    从凭据中获取默认模型名称，如果未配置则返回工具级默认值
    """
    credential_model = credentials.get("model", "").strip()
    return credential_model if credential_model else tool_default


def get_api_mode(credentials: dict[str, Any]) -> str:
    """
    从凭据中获取 API 调用模式
    
    Returns:
        str: 'async' (ModelScope 异步模式) 或 'sync' (OpenAI 兼容同步模式)
    """
    mode = credentials.get("api_mode", "").strip().lower()
    if mode not in ("async", "sync"):
        mode = "async"
    return mode


def is_custom_gateway(credentials: dict[str, Any]) -> bool:
    """判断是否配置了自定义网关"""
    return bool(credentials.get("base_url", "").strip())
