import re
import requests
import time
import json
import base64
from collections.abc import Generator
from PIL import Image
from io import BytesIO
from dify_plugin.entities.tool import ToolInvokeMessage
from dify_plugin import Tool
from utils import get_base_url, get_model, get_api_mode, is_custom_gateway

# 工具级模型默认值
TOOL_DEFAULT_MODEL = "Qwen/Qwen-Image-Edit-2511"


def upload_to_catbox(image_bytes: bytes, filename: str = "image.png") -> str:
    """
    上传图像到 litterbox.catbox.moe（临时图床，1小时后过期）
    返回公开可访问的 URL
    """
    url = "https://litterbox.catbox.moe/resources/internals/api.php"
    files = {
        'fileToUpload': (filename, image_bytes, 'image/png')
    }
    data = {
        'reqtype': 'fileupload',
        'time': '1h'
    }

    response = requests.post(url, files=files, data=data, timeout=120)
    response.raise_for_status()

    result_url = response.text.strip()
    if result_url.startswith('http'):
        return result_url
    else:
        raise Exception(f"图床上传失败: {result_url}")


class Image2ImageTool(Tool):
    def _invoke(
        self, tool_parameters: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        图生图工具，支持两种 API 模式：
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
            
        image_url = tool_parameters.get("image_url", "")
        if not image_url:
            yield self.create_text_message("❌ 请输入图像URL")
            return

        # 下载输入图像
        try:
            image_response = requests.get(image_url, stream=True, timeout=30)
            image_response.raise_for_status()
            image = Image.open(BytesIO(image_response.content))
            width, height = image.size
            origin_size = f"{width}x{height}"

            # 将图像转换为 RGB 模式
            if image.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', image.size, (255, 255, 255))
                if image.mode == 'P':
                    image = image.convert('RGBA')
                background.paste(image, mask=image.split()[-1] if image.mode == 'RGBA' else None)
                image = background
            elif image.mode != 'RGB':
                image = image.convert('RGB')

            # 转换为字节数据
            img_buffer = BytesIO()
            image.save(img_buffer, format='PNG')
            image_bytes = img_buffer.getvalue()

        except requests.exceptions.RequestException as e:
            yield self.create_text_message(f"❌ 无法下载输入图像: {str(e)}")
            return
        except Exception as e:
            yield self.create_text_message(f"❌ 处理输入图像失败: {str(e)}")
            return

        size = tool_parameters.get("size")
        if size and re.match(r"^\d+x\d+$", size) is None:
            yield self.create_text_message("❌ 尺寸参数格式错误，请使用 WxH 格式")
            yield self.create_text_message(f"💡 使用原图尺寸: {origin_size}")
            size = None
            
        # 模型名称优先级：工具参数 > 凭据配置 > 工具默认值
        tool_param_model = tool_parameters.get("model", "").strip()
        model = tool_param_model or get_model(self.runtime.credentials, TOOL_DEFAULT_MODEL)
        
        # 3. 设置请求头
        common_headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        
        try:
            yield self.create_text_message("🚀 正在提交图像编辑任务...")
            yield self.create_text_message(f"🔧 网关地址: {base_url}")
            yield self.create_text_message(f"🔧 使用模型: {model}")
            yield self.create_text_message(f"🔧 API 模式: {'同步' if api_mode == 'sync' else '异步'}")

            # 4. 处理图像 URL：自定义网关直接用原始 URL，ModelScope 需上传到图床
            if is_custom_gateway(self.runtime.credentials):
                public_image_url = image_url
                yield self.create_text_message("✅ 使用自定义网关，直接使用图像URL")
            else:
                yield self.create_text_message("📤 正在上传图像到临时图床...")
                try:
                    public_image_url = upload_to_catbox(image_bytes, "input_image.png")
                    yield self.create_text_message("✅ 图像上传成功")
                except Exception as upload_err:
                    yield self.create_text_message(f"❌ 图像上传到图床失败: {str(upload_err)}")
                    yield self.create_text_message("💡 提示：请确保网络连接正常，或稍后重试")
                    return

            if size:
                yield self.create_text_message(f"🔧 图像尺寸: {size}")
            else:
                yield self.create_text_message(f"🔧 图像尺寸: {origin_size} (原图)")
            yield self.create_text_message(f"🔧 提示词长度: {len(prompt)} 字符")

            # 5. 构建请求数据
            request_data = {
                "model": model,
                "prompt": prompt,
                "image_url": [public_image_url]
            }
            if size:
                request_data["size"] = size
            
            if api_mode == "sync":
                yield from self._handle_sync_mode(base_url, common_headers, request_data, image_bytes)
            else:
                yield from self._handle_async_mode(base_url, common_headers, request_data)
        
        except requests.exceptions.HTTPError as e:
            yield from self._handle_http_error(e)
        except requests.exceptions.RequestException as e:
            yield self.create_text_message(f"❌ 网络请求错误: {str(e)}")
        except json.JSONDecodeError as e:
            yield self.create_text_message(f"❌ API 响应解析错误: {str(e)}")
        except Exception as e:
            yield self.create_text_message(f"❌ 编辑图像时出现未知错误: {str(e)}")

    def _handle_sync_mode(
        self, base_url: str, headers: dict, request_data: dict, image_bytes: bytes
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        同步模式：POST 后直接在响应中获取结果
        适用于 OpenAI 兼容 API
        
        对于图生图，同步模式下优先使用 image 字段传递 base64 编码的图像
        （与 OpenAI images/edits API 兼容），如果 API 不支持，则回退到 image_url
        """
        # 尝试方式1：使用 base64 编码的 image 字段（OpenAI images/edits 风格）
        sync_data = {
            "model": request_data["model"],
            "prompt": request_data["prompt"],
            "n": 1,
        }
        if request_data.get("size"):
            sync_data["size"] = request_data["size"]
        
        # 使用 image_url 方式传递（更通用）
        sync_data["image_url"] = request_data["image_url"]
        
        response = requests.post(
            f"{base_url}v1/images/generations",
            headers=headers,
            json=sync_data,
            timeout=120
        )
        
        if response.status_code != 200:
            yield self.create_text_message(f"🔧 API 响应状态码: {response.status_code}")
            yield self.create_text_message(f"🔧 响应内容: {response.text[:500]}")
        
        response.raise_for_status()
        data = response.json()
        
        # 从响应中提取图像
        image_items = data.get("data", [])
        if not image_items:
            image_items = data.get("output_images", [])
        
        if not image_items:
            yield self.create_text_message("❌ 响应中未找到图像数据")
            yield self.create_text_message(f"🔧 完整响应: {json.dumps(data, ensure_ascii=False)[:500]}")
            return
        
        yield self._download_and_return_image(image_items[0])

    def _handle_async_mode(
        self, base_url: str, headers: dict, request_data: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        异步模式：提交任务 → 轮询状态 → 下载结果
        适用于 ModelScope API
        """
        response = requests.post(
            f"{base_url}v1/images/generations",
            headers={**headers, "X-ModelScope-Async-Mode": "true"},
            data=json.dumps(request_data, ensure_ascii=False).encode('utf-8'),
            timeout=300
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
        yield self.create_text_message("⏳ 正在编辑图像，请稍候...")
        
        # 轮询任务状态
        max_retries = 60
        retry_count = 0
        
        while retry_count < max_retries:
            time.sleep(5)
            
            result = requests.get(
                f"{base_url}v1/tasks/{task_id}",
                headers={**headers, "X-ModelScope-Task-Type": "image_generation"},
                timeout=120
            )
            
            result.raise_for_status()
            data = result.json()
            task_status = data.get("task_status")
            
            if task_status == "SUCCEED":
                output_images = data.get("output_images", [])
                image_url_to_download = None
                
                if output_images and len(output_images) > 0:
                    image_url_to_download = output_images[0]
                else:
                    output = data.get("output", {})
                    results = output.get("results", [])
                    if results and isinstance(results[0], dict):
                        image_url_to_download = results[0].get("url")
                    elif results and isinstance(results[0], str):
                        image_url_to_download = results[0]
                
                if not image_url_to_download:
                    yield self.create_text_message("❌ 编辑成功但未找到图像下载地址")
                    yield self.create_text_message(f"🔧 完整响应数据: {json.dumps(data, ensure_ascii=False)}")
                    return
                
                yield self._download_and_return_image({"url": image_url_to_download})
                return
                
            elif task_status == "FAILED":
                error_info = data.get("error") or data.get("errors") or {}
                
                def get_valid_msg(info):
                    if isinstance(info, dict):
                        msg = info.get("message")
                        return msg if msg and len(msg.strip()) > 0 else None
                    return None

                error_message = (
                    get_valid_msg(error_info) or 
                    data.get("message") or 
                    data.get("task_status_msg") or
                    "未知错误（API未返回具体错误描述）"
                )
                
                yield self.create_text_message(f"❌ 图像编辑失败: {error_message}")
                yield self.create_text_message(f"🔧 完整响应数据: {json.dumps(data, ensure_ascii=False)}")
                return
            
            wait_time = (retry_count + 1) * 5
            yield self.create_text_message(
                f"⏳ 图像正在编辑中，已等待 {wait_time} 秒..."
            )
            retry_count += 1
        
        yield self.create_text_message("⏰ 图像编辑超时（5分钟），请稍后再试")

    def _download_and_return_image(
        self, image_item: dict
    ) -> Generator[ToolInvokeMessage, None, None]:
        """
        从图像项中下载图像并返回，支持 URL 和 base64 两种格式
        """
        image_url = image_item.get("url") if isinstance(image_item, dict) else image_item
        b64_json = image_item.get("b64_json") if isinstance(image_item, dict) else None
        
        if b64_json:
            yield self.create_text_message("🎨 图像编辑成功，正在解码...")
            image_bytes = base64.b64decode(b64_json)
            image = Image.open(BytesIO(image_bytes))
        elif image_url:
            yield self.create_text_message("🎨 图像编辑成功，正在下载...")
            try:
                image_response = requests.get(image_url, timeout=60)
                image_response.raise_for_status()
            except Exception as download_err:
                yield self.create_text_message(f"❌ 下载生成的图片失败: {str(download_err)}")
                yield self.create_text_message(f"🔗 图片地址: {image_url}")
                return
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
        yield self.create_text_message("🎉 图像编辑完成！")

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
            yield self.create_text_message("2. 检查输入图像URL是否有效")
            yield self.create_text_message("3. 尝试简化提示词描述")
            yield self.create_text_message("4. 稍后重试，可能是服务器临时故障")
        else:
            yield self.create_text_message(f"❌ HTTP 错误: {e.response.status_code} - {str(e)}")
        
        if hasattr(e.response, 'text'):
            yield self.create_text_message(f"🔧 响应内容: {e.response.text[:200]}")
