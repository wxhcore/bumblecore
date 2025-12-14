import argparse
import asyncio
import json
import os
import time
import traceback
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

import uvicorn
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


class ChatRequest(BaseModel):
    messages: str 


class OpenAIRequest(BaseModel):
    messages: List[Dict[str, str]]
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None
    do_sample: Optional[bool] = None


class ChatService:
    def __init__(
        self,
        model_path: str,
        device_map: Optional[str] = None,    
        dtype: Optional[str] = None,
        training_stage: Optional[str] = None,
        enable_history: Optional[bool] = None,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        do_sample: Optional[bool] = None,
    ):
        self.training_stage = training_stage
        self.enable_history = enable_history
        self.conversation_history: List[Dict[str, str]] = []

        self.default_params = {
            "system_prompt": system_prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": do_sample,
        }
        self.bot = self._load_model(model_path, device_map, dtype)

    def _load_model(self, model_path: str, device_map: str, dtype: str):
        print(f"正在加载模型: {model_path}...")
        from bumblecore.inference import BumblebeeChat, HFStreamChat
        bot = BumblebeeChat(
            model_path=model_path,
            device_map=device_map,
            dtype=dtype
        )
        status = "启用" if self.enable_history else "禁用"
        print(f"✅ 模型加载完成！训练阶段: {self.training_stage}，对话历史: {status}")
        return bot

    def _make_sse(self, data: dict) -> str:
        sse = f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
        if len(sse.encode()) < 1024:
            pad_len = 1024 - len(sse.encode()) + 10
            sse += ":" + " " * pad_len + "\n\n"
        return sse

    async def chat_stream(self, messages: str) -> AsyncGenerator[str, None]:

        full_response = ""
        try:
            if self.training_stage == "pretrain":
                stream = self.bot.stream_chat(messages=messages, **self.default_params)
            else:
                if self.enable_history:
                    messages = self.conversation_history + [{"role": "user", "content": messages}]
                else:
                    messages = [{"role": "user", "content": messages}]
                stream = self.bot.stream_chat(messages=messages, **self.default_params)

            for token in stream:
                if token:
                    yield self._make_sse({'token': token})
                    full_response += token
                    await asyncio.sleep(0)

            yield self._make_sse({'done': True})

            if self.training_stage != "pretrain" and self.enable_history:
                self.conversation_history.append({"role": "user", "content": messages})
                self.conversation_history.append({"role": "assistant", "content": full_response})

        except Exception as e:
            print(f"生成错误: {e}",traceback.format_exc())
            yield self._make_sse({'error': str(e)})
    
    def openai_chat(self, request: OpenAIRequest):
        gen_kwargs = {
            "messages": request.messages,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "top_k": request.top_k,
            "max_new_tokens": request.max_completion_tokens,
            "repetition_penalty": request.repetition_penalty,
            "do_sample": request.do_sample,
        }
        response_dict = self.bot.chat(**gen_kwargs)

        return {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": None,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": response_dict["response"],
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": response_dict["prompt_tokens"],
                    "completion_tokens": response_dict["completion_tokens"],
                    "total_tokens": response_dict["total_tokens"]
                }
            }


    def clear_session(self):
        self.conversation_history.clear()


app = FastAPI(title="BumbleChat")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).parent.parent.parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = FRONTEND_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

chat_service : ChatService = None

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    return StreamingResponse(
        chat_service.chat_stream(request.messages),
        media_type="text/event-stream",
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )

@app.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIRequest):
    return chat_service.openai_chat(request)


@app.delete("/session")
async def clear_session():
    chat_service.clear_session()
    return {"messages": "会话已清除"}


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    html_file = FRONTEND_DIR / "bumblechat.html"
    if html_file.exists():
        return HTMLResponse(html_file.read_text(encoding="utf-8"))
    else:
        return HTMLResponse(
            content="<h1>错误：找不到前端文件</h1><p>请确保 frontend/bumblechat.html 文件存在。</p>",
            status_code=404
        )


def start_bumblebee_chat_service():
    global chat_service

    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--yaml_config",
        type=str,
        default="",
        help="YAML 配置文件路径",
    )
    cfg_args, _ = base_parser.parse_known_args()
    config_path = cfg_args.yaml_config
    
    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    else:
        cfg = {}
    
    parser = argparse.ArgumentParser(
        description="启动 Bumblebee Chat Web 服务",
        parents=[base_parser],
    )
    parser.add_argument("--model_path", type=str, 
                        default=cfg.get("model_path"),
                        help="模型路径")
                        
    parser.add_argument("--device_map", type=str, 
                        default=cfg.get("device_map", "cpu"),
                        help="设备映射")
                        
    parser.add_argument("--dtype", type=str, 
                        default=cfg.get("dtype", "auto"),
                        help="模型数据类型")
                        
    parser.add_argument("--training_stage", type=str, 
                        default=cfg.get("training_stage", "sft"),
                        choices=["sft", "dpo", "pretrain"],
                        help="模型训练阶段")
                        
    parser.add_argument("--enable_history", action="store_true",
                        default=cfg.get("enable_history", False),
                        help="启用多轮对话历史")
                        
    parser.add_argument("--host", type=str, 
                        default=cfg.get("host", "127.0.0.1"),
                        help="服务器主机地址")
                        
    parser.add_argument("--port", type=int, 
                        default=cfg.get("port", 8000),
                        help="服务器端口")

    parser.add_argument("--system_prompt", type=str, 
                        default=cfg.get("system_prompt", None),
                        help="系统提示词")
                        
    parser.add_argument("--max_new_tokens", type=int, 
                        default=cfg.get("max_new_tokens", None),
                        help="最大生成 token 数")
                        
    parser.add_argument("--temperature", type=float, 
                        default=cfg.get("temperature", None),
                        help="采样温度")
                        
    parser.add_argument("--top_k", type=int, 
                        default=cfg.get("top_k", None),
                        help="Top_k 采样")
                        
    parser.add_argument("--top_p", type=float, 
                        default=cfg.get("top_p", None),
                        help="Top_p (nucleus) 采样")
                        
    parser.add_argument("--repetition_penalty", type=float, 
                        default=cfg.get("repetition_penalty", None),
                        help="重复惩罚系数")
                        
    parser.add_argument("--do_sample", action="store_true",
                        default=cfg.get("do_sample", False),
                        help="启用采样")

    args = parser.parse_args()
    chat_service = ChatService(
        model_path=args.model_path,
        device_map=args.device_map,
        dtype=args.dtype,
        training_stage=args.training_stage,
        enable_history=args.enable_history,
        system_prompt=args.system_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=args.do_sample,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )