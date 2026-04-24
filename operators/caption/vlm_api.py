from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import httpx

log = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

DEFAULT_PROVIDER = "dashscope"
DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DEFAULT_ARK_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_MODEL = "qwen3.5-plus"
DEFAULT_POLL_INTERVAL_SEC = 20
PROVIDER_DEFAULT_MODELS = {
    "dashscope": "qwen3.5-flash",
    "volcengine_ark": "doubao-seed-2-0-lite-260215",
}


def get_vlm_provider() -> str:
    provider = os.getenv("VLM_API_PROVIDER", DEFAULT_PROVIDER).strip().lower()
    return provider or DEFAULT_PROVIDER


def get_api_key() -> str:
    provider = get_vlm_provider()
    if provider == "volcengine_ark":
        return os.getenv("ARK_API_KEY", "").strip() or os.getenv("VLM_API_KEY", "").strip()
    return (
        os.getenv("DASHSCOPE_API_KEY", "").strip()
        or os.getenv("VLM_API_KEY", "").strip()
    )


def get_base_url() -> str:
    provider = get_vlm_provider()
    if provider == "volcengine_ark":
        return DEFAULT_ARK_BASE_URL
    return DEFAULT_BASE_URL


def get_provider_default_model(provider: str | None = None, fallback: str = DEFAULT_MODEL) -> str:
    resolved_provider = (provider or get_vlm_provider()).strip().lower()
    return PROVIDER_DEFAULT_MODELS.get(resolved_provider, fallback)


def get_default_model(task: str | None = None, fallback: str | None = None) -> str:
    task_key = f"VLM_{str(task or '').strip().upper()}_MODEL"
    provider_default = get_provider_default_model(fallback=DEFAULT_MODEL if fallback is None else fallback)
    return (
        os.getenv(task_key, "").strip()
        or os.getenv("VLM_MODEL", "").strip()
        or os.getenv("VLM_DEFAULT_MODEL", "").strip()
        or provider_default
    )


def provider_supports_batch_api(provider: str | None = None) -> bool:
    return (provider or get_vlm_provider()) == "dashscope"


def _create_openai_compatible_client():
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError(f"{get_vlm_provider()} API key is not set")

    from openai import OpenAI

    return OpenAI(api_key=api_key, base_url=get_base_url())


def build_multimodal_message(image_b64_list: list[str], prompt: str) -> list[dict[str, Any]]:
    content = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for b64 in image_b64_list
        if b64
    ]
    content.append({"type": "text", "text": prompt})
    return [{"role": "user", "content": content}]


def _extract_openai_response_text(response) -> str | None:
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


def _extract_openai_usage_dict(response) -> dict[str, Any]:
    usage = getattr(response, "usage", None)
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    return {
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0,
    }


def _extract_openai_payload_text(payload: dict[str, Any]) -> str | None:
    choices = payload.get("choices") or []
    if not choices:
        return None
    message = choices[0].get("message") or {}
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text = item.get("text")
                if text:
                    texts.append(str(text))
        if texts:
            return "\n".join(texts)
    return str(content) if content is not None else None


def _extract_openai_payload_usage_dict(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage") or {}
    if not isinstance(usage, dict):
        return {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0) or usage.get("input_tokens", 0) or 0,
        "completion_tokens": usage.get("completion_tokens", 0) or usage.get("output_tokens", 0) or 0,
    }


def _convert_messages_to_ark_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        if isinstance(content, str):
            items = [{"type": "input_text", "text": content}]
        else:
            items = []
            for entry in content or []:
                entry_type = entry.get("type")
                if entry_type == "text":
                    items.append({"type": "input_text", "text": entry.get("text", "")})
                    continue
                if entry_type == "image_url":
                    image_value = entry.get("image_url")
                    if isinstance(image_value, dict):
                        image_value = image_value.get("url")
                    if image_value:
                        items.append({"type": "input_image", "image_url": image_value})
        converted.append({"role": role, "content": items})
    return converted


def _extract_ark_response_text(payload: dict[str, Any]) -> str | None:
    output_text = payload.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    texts: list[str] = []
    for item in payload.get("output", []) or []:
        for content in item.get("content", []) or []:
            if content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if text:
                    texts.append(str(text))
    if texts:
        return "\n".join(texts)
    return None


def _extract_ark_usage_dict(payload: dict[str, Any]) -> dict[str, Any]:
    usage = payload.get("usage") or {}
    if not isinstance(usage, dict):
        return {}
    return {
        "prompt_tokens": usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0) or 0,
        "completion_tokens": usage.get("output_tokens", 0) or usage.get("completion_tokens", 0) or 0,
    }


def _submit_one_ark_request(
    client: httpx.Client,
    req: dict[str, Any],
    model: str,
    extra_body: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    response_payload: dict[str, Any] = {
        "model": req.get("model", model),
        "input": _convert_messages_to_ark_input(req["messages"]),
    }
    # Some OpenAI-compatible extras used by other providers are not accepted by
    # Ark Responses API for multimodal requests. Keep the upper-layer contract
    # stable by silently dropping them here.
    try:
        response = client.post(
            f"{get_base_url().rstrip('/')}/responses",
            headers={"Authorization": f"Bearer {get_api_key()}"},
            json=response_payload,
        )
        response.raise_for_status()
        body = response.json()
        return req["custom_id"], {
            "text": _extract_ark_response_text(body),
            "usage": _extract_ark_usage_dict(body),
            "status_code": response.status_code,
        }
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code != 404:
            raise

    log.warning(
        "Ark /responses is unavailable for %s; retrying %s via /chat/completions",
        req["custom_id"],
        req.get("model", model),
    )
    chat_payload: dict[str, Any] = {
        "model": req.get("model", model),
        "messages": req["messages"],
    }
    body_extra = req.get("extra_body") or extra_body or {}
    if body_extra:
        chat_payload.update(body_extra)
    response = client.post(
        f"{get_base_url().rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {get_api_key()}"},
        json=chat_payload,
    )
    response.raise_for_status()
    body = response.json()
    return req["custom_id"], {
        "text": _extract_openai_payload_text(body),
        "usage": _extract_openai_payload_usage_dict(body),
        "status_code": response.status_code,
    }


def _format_httpx_error(exc: Exception) -> str:
    if not isinstance(exc, httpx.HTTPStatusError):
        return str(exc)
    response = exc.response
    body_text = ""
    try:
        raw = response.text.strip()
        if raw:
            body_text = raw[:400]
    except Exception:
        body_text = ""
    if body_text:
        return f"{exc}; body={body_text}"
    return str(exc)


def submit_batch_chat_requests_async(
    requests: list[dict[str, Any]],
    *,
    model: str = DEFAULT_MODEL,
    completion_window: str = "24h",
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not requests:
        raise ValueError("requests must not be empty")
    if not provider_supports_batch_api():
        raise NotImplementedError(
            f"Provider {get_vlm_provider()!r} does not support async batch submission in this project; "
            "please disable batch_enabled."
        )

    client = _create_openai_compatible_client()

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
    if not provider_supports_batch_api():
        raise NotImplementedError(
            f"Provider {get_vlm_provider()!r} does not support batch status retrieval in this project."
        )

    client = _create_openai_compatible_client()
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
    if not provider_supports_batch_api():
        raise NotImplementedError(
            f"Provider {get_vlm_provider()!r} does not support batch collection in this project."
        )

    client = _create_openai_compatible_client()
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
    if not provider_supports_batch_api():
        return submit_direct_chat_requests(
            requests,
            model=model,
            extra_body=extra_body,
            max_workers=min(8, len(requests)),
        )
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

    provider = get_vlm_provider()
    max_workers = max(1, max_workers)
    results: dict[str, dict[str, Any]] = {}

    if provider == "dashscope":
        client = _create_openai_compatible_client()

        def _submit_one(req: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            response = client.chat.completions.create(
                model=req.get("model", model),
                messages=req["messages"],
                extra_body=req.get("extra_body") or extra_body or {},
            )
            return req["custom_id"], {
                "text": _extract_openai_response_text(response),
                "usage": _extract_openai_usage_dict(response),
                "status_code": 200,
            }
    elif provider == "volcengine_ark":
        headers = {
            "Authorization": f"Bearer {get_api_key()}",
            "Content-Type": "application/json",
        }
        client = httpx.Client(timeout=120.0, headers=headers)

        def _submit_one(req: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            return _submit_one_ark_request(client, req, model, extra_body)
    else:
        raise ValueError(f"Unsupported VLM provider: {provider!r}")

    try:
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
                        "error": _format_httpx_error(e),
                    }
    finally:
        if hasattr(client, "close"):
            client.close()
    return results
