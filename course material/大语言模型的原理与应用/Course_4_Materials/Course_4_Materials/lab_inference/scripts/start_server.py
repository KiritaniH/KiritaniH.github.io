#!/usr/bin/env python3
"""启动 vLLM 服务的脚本（绕过 uvloop 问题）"""

import sys
import asyncio
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai import api_server
from vllm.entrypoints.openai.api_server import build_app, run_server

MODEL_PATH = r"C:\Users\lenovo\.cache\modelscope\hub\models\Qwen\Qwen3-0___6B"

if __name__ == "__main__":
    # 设置 EngineArgs
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        dtype="auto",
        trust_remote_code=True,
    )

    # 创建 engine
    engine = AsyncLLMEngine.from_engine_args(engine_args)

    # 构建并运行 API 服务器
    app = build_app(engine)

    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
