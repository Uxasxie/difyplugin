import re
import requests
import time
import json
from collections.abc import Generator
from PIL import Image
from io import BytesIO
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin import Tool
from utils import get_base_url, get_model, get_api_mode

# 工具级模型默认值（当凭据和工具参数都未指定时使用）
TOOL_DEFAULT_MODEL = "Qwen/Qwen-Image-2512"


class Text2ImageTool(Tool):
    def _invoke(
        self, tool_parameters: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        文生图工具，支持两种 API 模式：
        - async: ModelScope 风格的异步提交+轮询模式
        - sync: OpenAI 兼容的同步模式（vLLM、TGI 等内部部署常用）
        """
        # 1. 获取 API 配置
        api_key = self.runtime.credentials.get("api_key")
        base_url = get_base_url(self.runtime.credentials)
        api_mode = get_api_mode(self.runtime.credentials)
        
        # 2. 获取和验证参数
        prompt = tool_parameters.get("prompt", "")
        if not prompt:
            yield self.create_text_message("❌ 请输入提示词")
            return

        size = tool_parameters.get("size", "1024x1024")
        if re.match(r"^\d+x\d+$", size) is None:
            yield self.create_text_message("❌ 尺寸参数格式错误，请使用 WxH 格式")
            yield self.create_text_message("💡 使用默认尺寸: 1024x1024")
            size = "1024x1024"

        # 模型名称优先级：工具参数 > 凭据配置 > 工具默认值
        tool_param_model = tool_parameters.get("model", "").strip()
        model = tool_param_model or get_model(self.runtime.credentials, TOOL_DEFAULT_MODEL)
        
        # 3. 设置请求头
        common_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            yield self.create_text_message("🚀 正在提交图像生成任务...")
            yield self.create_text_message(f"🔧 网关地址: {base_url}")
            yield self.create_text_message(f"🔧 使用模型: {model}")
            yield self.create_text_message(f"🔧 API 模式: {'同步' if api_mode == 'sync' else '异步'}")
            yield self.create_text_message(f"🔧 提示词长度: {len(prompt)} 字符")
            
            # 4. 构建请求数据
            request_data = {
                "model": model,
                "prompt": prompt,
                "n": 1,
                "size": size
            }
            
            if api_mode == "sync":
                # ===== 同步模式：直接获取结果 =====
                yield self._handle_sync_mode(base_url, common_headers, request_data)
            else:
                # ===== 异步模式：提交+轮询 =====
                yield self._handle_async_mode(base_url, common_headers, request_data)
        
        except requests.exceptions.HTTPError as e:
            yield self._handle_http_error(e)
        except requests.exceptions.RequestException as e:
            yield self.create_text_message(f"❌ 网络请求错误: {str(e)}")
        except json.JSONDecodeError as e:
            yield self.create_text_message(f"❌ API 响应解析错误: {str(e)}")
        except Exception as e:
            yield self.create_text_message(f"❌ 生成图像时出现未知错误: {str(e)}")

    def _handle_sync_mode(
        self, base_url: str, headers: dict, request_data: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        同步模式处理：POST 请求后直接在响应中获取图像 URL
        适用于 OpenAI 兼容 API（vLLM、TGI 等内部部署）
        
        响应格式（OpenAI 兼容）：
        {
            "data": [
                { "url": "https://..." }  或  { "b64_json": "..." }
            ]
        }
        """
        response = requests.post(
            f"{base_url}v1/images/generations",
            headers=headers,
            json=request_data,
            timeout=120  # 同步模式需要较长超时
        )
        
        if response.status_code != 200:
            yield self.create_text_message(f"🔧 API 响应状态码: {response.status_code}")
            yield self.create_text_message(f"🔧 响应内容: {response.text[:500]}")
        
        response.raise_for_status()
        data = response.json()
        
        # 从 OpenAI 兼容响应中提取图像
        image_items = data.get("data", [])
        if not image_items:
            # 尝试 ModelScope 风格的 output_images
            image_items = data.get("output_images", [])
        
        if not image_items:
            yield self.create_text_message("❌ 响应中未找到图像数据")
            yield self.create_text_message(f"🔧 完整响应: {json.dumps(data, ensure_ascii=False)[:500]}")
            return
        
        image_item = image_items[0]
        yield self._download_and_return_image(image_item)

    def _handle_async_mode(
        self, base_url: str, headers: dict, request_data: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        异步模式处理：提交任务 → 轮询状态 → 下载结果
        适用于 ModelScope API
        """
        # 提交异步任务
        response = requests.post(
            f"{base_url}v1/images/generations",
            headers={**headers, "X-ModelScope-Async-Mode": "true"},
            json=request_data,
            timeout=30
        )
        
        if response.status_code != 200:
            yield self.create_text_message(f"🔧 API 响应状态码: {response.status_code}")
            yield self.create_text_message(f"🔧 响应内容: {response.text[:500]}")
        
        response.raise_for_status()
        
        response_data = response.json()
        task_id = response_data.get("task_id")
        
        if not task_id:
            yield self.create_text_message("❌ 创建任务失败，未获取到任务ID")
            return
        
        yield self.create_text_message(f"✅ 任务已创建，ID: {task_id}")
        yield self.create_text_message("⏳ 正在生成图像，请稍候...")
        
        # 轮询任务状态
        max_retries = 60
        retry_count = 0
        
        while retry_count < max_retries:
            time.sleep(5)
            
            result = requests.get(
                f"{base_url}v1/tasks/{task_id}",
                headers={**headers, "X-ModelScope-Task-Type": "image_generation"},
            )
            
            result.raise_for_status()
            data = result.json()
            task_status = data.get("task_status")
            
            if task_status == "SUCCEED":
                output_images = data.get("output_images", [])
                if not output_images:
                    yield self.create_text_message("❌ 生成成功但未找到图像数据")
                    return
                
                yield self._download_and_return_image({"url": output_images[0]})
                return
                
            elif task_status == "FAILED":
                error_info = data.get("error", {})
                error_message = error_info.get("message", "未知错误")
                yield self.create_text_message(f"❌ 图像生成失败: {error_message}")
                return
            
            wait_time = (retry_count + 1) * 5
            yield self.create_text_message(
                f"⏳ 图像正在生成中，已等待 {wait_time} 秒..."
            )
            retry_count += 1
        
        yield self.create_text_message("⏰ 图像生成超时（5分钟），请稍后再试")

    def _download_and_return_image(
        self, image_item: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        从图像项中下载图像并返回，支持 URL 和 base64 两种格式
        """
        image_url = image_item.get("url") if isinstance(image_item, dict) else image_item
        b64_json = image_item.get("b64_json") if isinstance(image_item, dict) else None
        
        if b64_json:
            # base64 编码的图像
            import base64
            yield self.create_text_message("🎨 图像生成成功，正在解码...")
            image_bytes = base64.b64decode(b64_json)
            image = Image.open(BytesIO(image_bytes))
        elif image_url:
            # URL 方式
            yield self.create_text_message("🎨 图像生成成功，正在下载...")
            image_response = requests.get(image_url, timeout=60)
            image_response.raise_for_status()
            image = Image.open(BytesIO(image_response.content))
        else:
            yield self.create_text_message("❌ 未找到图像 URL 或 base64 数据")
            return
        
        # 转换为 PNG 格式
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()
        
        yield self.create_blob_message(
            blob=img_byte_arr,
            meta={"mime_type": "image/png"}
        )
        yield self.create_text_message("🎉 图像生成完成！")

    def _handle_http_error(
        self, e: requests.exceptions.HTTPError
    ) -> Generator[ToolInvokeMessage, None, None]:
        """统一处理 HTTP 错误"""
        if e.response.status_code == 401:
            yield self.create_text_message("❌ API Key 无效，请检查您的 API Key 配置")
        elif e.response.status_code == 429:
            yield self.create_text_message("❌ API 调用频率过高，请稍后再试")
        elif e.response.status_code == 500:
            yield self.create_text_message("❌ 服务器内部错误")
            yield self.create_text_message("💡 可能的解决方案:")
            yield self.create_text_message("1. 检查提示词是否包含敏感内容")
            yield self.create_text_message("2. 尝试简化提示词描述")
            yield self.create_text_message("3. 稍后重试，可能是服务器临时故障")
        else:
            yield self.create_text_message(f"❌ HTTP 错误: {e.response.status_code} - {str(e)}")
        
        if hasattr(e.response, 'text'):
            yield self.create_text_message(f"🔧 响应内容: {e.response.text[:200]}")
