from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

BASE_URL = os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DEFAULT_MODEL = "qwen3.5-plus"
DEFAULT_POLL_INTERVAL_SEC = 20


def get_api_key() -> str:
    return os.getenv("DASHSCOPE_API_KEY", "").strip()


def _create_client():
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("DASHSCOPE_API_KEY is not set")

    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=BASE_URL)


def build_multimodal_message(image_b64_list: list[str], prompt: str) -> list[dict[str, Any]]:
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for b64 in image_b64_list
        if b64
    ]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _extract_response_text(response) -> str | None:
    choices = getattr(response, "choices", None) or []
    if not choices:
        return None
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                texts.append(item.get("text", ""))
        if texts:
            return "\n".join(texts)
    return str(content) if content is not None else None


def _extract_usage_dict(response) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0,
    }


def submit_batch_chat_requests_async(
    requests: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    completion_window: str = "24h",
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not requests:
        raise ValueError("requests must not be empty")

    client = _create_client()

    with tempfile.TemporaryDirectory() as tmp_dir:
        jsonl_path = Path(tmp_dir) / "batch_requests.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as f:
            for req in requests:
                payload = {
                    "custom_id": req["custom_id"],
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": req.get("model", model),
                        "messages": req["messages"],
                    },
                }
                body_extra = req.get("extra_body") or extra_body or {}
                if body_extra:
                    payload["body"].update(body_extra)
                f.write(json.dumps(payload, ensure_ascii=False) + "\n")

        file_obj = client.files.create(file=jsonl_path, purpose="batch")
        batch = client.batches.create(
            input_file_id=file_obj.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
        )

        log.info("Batch created: %s (%s requests)", batch.id, len(requests))
        return {
            "batch_id": batch.id,
            "input_file_id": file_obj.id,
            "status": batch.status,
            "request_count": len(requests),
            "model": model,
            "completion_window": completion_window,
        }


def retrieve_batch_status(batch_id: str) -> dict[str, Any]:
    client = _create_client()
    batch = client.batches.retrieve(batch_id)
    result = {
        "batch_id": batch.id,
        "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
    }
    if hasattr(batch, "request_counts") and batch.request_counts:
        rc = batch.request_counts
        result["request_counts"] = {
            "completed": rc.completed,
            "failed": rc.failed,
            "total": rc.total,
        }
    return result


def collect_batch_chat_requests(
    batch_id: str,
    *,
    poll_interval_sec: int = DEFAULT_POLL_INTERVAL_SEC,
    wait: bool = True,
) -> dict[str, Any]:
    client = _create_client()
    batch = client.batches.retrieve(batch_id)
    terminal = {"completed", "failed", "expired", "cancelled"}
    counts = ""
    if hasattr(batch, "request_counts") and batch.request_counts:
        rc = batch.request_counts
        counts = f" (done={rc.completed}, fail={rc.failed}, total={rc.total})"
    log.info("Batch status: %s %s%s", batch.id, batch.status, counts)

    while wait and batch.status not in terminal:
        time.sleep(poll_interval_sec)
        batch = client.batches.retrieve(batch.id)
        counts = ""
        if hasattr(batch, "request_counts") and batch.request_counts:
            rc = batch.request_counts
            counts = f" (done={rc.completed}, fail={rc.failed}, total={rc.total})"
        log.info("Batch status: %s %s%s", batch.id, batch.status, counts)

    result: dict[str, Any] = {
        "batch_id": batch.id,
        "status": batch.status,
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "results": {},
    }
    if hasattr(batch, "request_counts") and batch.request_counts:
        rc = batch.request_counts
        result["request_counts"] = {
            "completed": rc.completed,
            "failed": rc.failed,
            "total": rc.total,
        }

    if batch.status != "completed":
        return result
    if not batch.output_file_id:
        raise RuntimeError("Batch completed but output_file_id is missing")

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "batch_output.jsonl"
        client.files.content(batch.output_file_id).write_to_file(str(out_path))

        results: dict[str, dict[str, Any]] = {}
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            cid = obj["custom_id"]
            resp = obj.get("response", {})
            body = resp.get("body", {})

            text = None
            if body.get("choices"):
                text = body["choices"][0].get("message", {}).get("content", "")

            results[cid] = {
                "text": text,
                "usage": body.get("usage"),
                "status_code": resp.get("status_code", 200),
            }
        result["results"] = results
        return result


def submit_batch_chat_requests(
    requests: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    completion_window: str = "24h",
    poll_interval_sec: int = DEFAULT_POLL_INTERVAL_SEC,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    if not requests:
        return {}
    submission = submit_batch_chat_requests_async(
        requests,
        model=model,
        completion_window=completion_window,
        extra_body=extra_body,
    )
    collected = collect_batch_chat_requests(
        submission["batch_id"],
        poll_interval_sec=poll_interval_sec,
        wait=True,
    )
    if collected["status"] != "completed":
        raise RuntimeError(f"Batch ended with status: {collected['status']}")
    return collected["results"]


def submit_direct_chat_requests(
    requests: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    extra_body: dict[str, Any] | None = None,
    max_workers: int = 8,
) -> dict[str, dict[str, Any]]:
    if not requests:
        return {}

    client = _create_client()
    max_workers = max(1, max_workers)
    results: dict[str, dict[str, Any]] = {}

    def _submit_one(req: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        response = client.chat.completions.create(
            model=req.get("model", model),
            messages=req["messages"],
            extra_body=req.get("extra_body") or extra_body or {},
        )
        return req["custom_id"], {
            "text": _extract_response_text(response),
            "usage": _extract_usage_dict(response),
            "status_code": 200,
        }

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map = {
            executor.submit(_submit_one, req): req["custom_id"]
            for req in requests
        }
        for future in as_completed(future_map):
            cid = future_map[future]
            try:
                key, value = future.result()
                results[key] = value
            except Exception as e:
                results[cid] = {
                    "text": None,
                    "usage": None,
                    "status_code": 500,
                    "error": str(e),
                }
    return results
