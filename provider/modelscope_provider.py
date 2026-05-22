from typing import Any
from dify_plugin.errors.tool import ToolProviderCredentialValidationError
from dify_plugin import ToolProvider
from utils import get_base_url


class ModelScopeProvider(ToolProvider):
    def _validate_credentials(self, credentials: dict[str, Any]) -> None:
        """
        验证 API 凭据有效性，包括 API Key 和网关地址
        
        Args:
            credentials: 包含 API key 和 base_url 的字典
            
        Raises:
            ToolProviderCredentialValidationError: 当凭据验证失败时
        """
        try:
            # 1. 检查 API key
            api_key = credentials.get("api_key")
            if not api_key:
                raise ToolProviderCredentialValidationError(
                    "API Key 不能为空"
                )
            
            if not api_key.startswith("ms-"):
                # 内部部署的网关可能使用不同的 API Key 格式，仅给出提示而非拒绝
                # 如果配置了自定义网关地址，则允许非 ms- 开头的 API Key
                base_url = credentials.get("base_url", "").strip()
                if not base_url:
                    raise ToolProviderCredentialValidationError(
                        "使用 ModelScope 默认网关时，API Key 应以 'ms-' 开头。"
                        "如果使用自定义网关，请同时填写网关地址。"
                    )
            
            if len(api_key) < 6:
                raise ToolProviderCredentialValidationError(
                    "API Key 长度不正确"
                )
            
            # 2. 检查网关地址格式
            base_url = credentials.get("base_url", "").strip()
            if base_url:
                if not base_url.startswith(("http://", "https://")):
                    raise ToolProviderCredentialValidationError(
                        "网关地址必须以 http:// 或 https:// 开头"
                    )
                # 去除末尾斜杠后检查长度，避免过短的无效地址
                clean_url = base_url.rstrip("/")
                if len(clean_url) < 10:
                    raise ToolProviderCredentialValidationError(
                        "网关地址格式不正确，请输入有效的 URL"
                    )
            
            # 3. 检查 API 模式
            api_mode = credentials.get("api_mode", "").strip().lower()
            if api_mode and api_mode not in ("async", "sync"):
                raise ToolProviderCredentialValidationError(
                    "API 模式只能为 'async' 或 'sync'"
                )
            
        except ToolProviderCredentialValidationError:
            # 重新抛出已知的验证错误
            raise
        except Exception as e:
            raise ToolProviderCredentialValidationError(
                f"API 凭据验证失败: {str(e)}"
            )