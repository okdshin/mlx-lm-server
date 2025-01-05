import argparse
import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from mlx_lm import generate, load
from pydantic import BaseModel


def parse_args():
    parser = argparse.ArgumentParser(description="MLX-LM OpenAI Compatible API Server")
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the MLX-LM model"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to run the server on (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on (default: 8000)",
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=100,
        help="Maximum size of the request queue (default: 100)",
    )
    return parser.parse_args()


app = FastAPI()


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    max_tokens: Optional[int] = 16
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class CompletionChoice(BaseModel):
    text: str
    index: int
    logprobs: Optional[dict] = None
    finish_reason: str


class CompletionResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: dict


@dataclass
class QueueItem:
    request_id: str
    prompt: str
    config: dict
    future: asyncio.Future
    timestamp: float


class ModelWorker:
    def __init__(self, model_path: str, queue_size: int = 100):
        print(f"Loading model from {model_path}")
        self.model, self.tokenizer = load(model_path)
        self.request_queue = asyncio.Queue(maxsize=queue_size)
        self.worker_task = None

    def start(self):
        self.worker_task = asyncio.create_task(self._process_queue())

    async def _process_queue(self):
        while True:
            try:
                item: QueueItem = await self.request_queue.get()

                try:
                    outputs = generate(
                        self.model, self.tokenizer, prompt=item.prompt, **item.config
                    )

                    prompt_tokens = len(self.tokenizer.encode(item.prompt))
                    completion_tokens = sum(
                        len(self.tokenizer.encode(output)) for output in outputs
                    )

                    choices = [
                        CompletionChoice(
                            text=output,
                            index=i,
                            finish_reason=(
                                "length"
                                if len(output) >= item.config["max_tokens"]
                                else "stop"
                            ),
                        )
                        for i, output in enumerate(outputs[: item.config["n"]])
                    ]

                    response = CompletionResponse(
                        id=item.request_id,
                        object="text_completion",
                        created=int(item.timestamp),
                        model="mlx-model",
                        choices=choices,
                        usage={
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "total_tokens": prompt_tokens + completion_tokens,
                        },
                    )

                    item.future.set_result(response)

                except Exception as e:
                    item.future.set_exception(e)

                finally:
                    self.request_queue.task_done()

            except Exception as e:
                print(f"Worker error: {e}")
                await asyncio.sleep(1)

    async def enqueue_request(self, request: CompletionRequest) -> CompletionResponse:
        config = {
            "max_tokens": request.max_tokens,
            "temp": request.temperature,
            "top_p": request.top_p,
            "n": request.n,
        }

        request_id = f"cmpl-{uuid.uuid4()}"
        future = asyncio.Future()

        queue_item = QueueItem(
            request_id=request_id,
            prompt=request.prompt,
            config=config,
            future=future,
            timestamp=time.time(),
        )

        try:
            await self.request_queue.put(queue_item)
        except asyncio.QueueFull:
            raise HTTPException(status_code=503, detail="Server is too busy")

        try:
            response = await asyncio.wait_for(future, timeout=30.0)
            return response
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Request timeout")


args = parse_args()
model_worker = ModelWorker(args.model_path, queue_size=args.queue_size)


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest):
    try:
        return await model_worker.enqueue_request(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    model_worker.start()


@app.on_event("shutdown")
async def shutdown_event():
    if model_worker.worker_task:
        model_worker.worker_task.cancel()
        try:
            await model_worker.worker_task
        except asyncio.CancelledError:
            pass


if __name__ == "__main__":
    import uvicorn

    print(f"Starting server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
