"""
services/gemini_service.py

Shared Gemini transport (the "gemini" adapter of the common provider layer
in services/llm_service.py, which owns routing, quota handling, retries,
fallbacks and usage accounting).

Uses the current Google Gemini **Interactions API** (GA June 2026) via the
`google-genai` SDK rather than the legacy `generateContent` surface:

    client.interactions.create(model=..., input=..., system_instruction=...)

Design notes:
  - One call = one stateless interaction. Server-side storage of requests/
    responses is disabled (`store=False`) because this bot's analysis calls
    never resume a previous interaction, and the payloads contain portfolio
    data that has no reason to be retained server-side.
  - The client is created lazily and cached (same pattern as
    alpaca_service's cached trading client).
  - Model is configurable via GEMINI_MODEL (default: gemini-3.6-flash).
    A "models/" prefix is tolerated and stripped, since the Interactions API
    expects the bare model id.
  - Output is expected to be a single JSON object; extraction is tolerant of
    markdown fences and stray text, exactly like the agents' original parsers.
"""

import json
import logging
import re

from config import settings

logger = logging.getLogger("gemini_service")

_client = None  # cached genai.Client


def _get_client():
    """Lazily instantiate and cache the Gemini client.

    Hard timeout and a single HTTP attempt: long SDK-internal retries are
    disabled so an unreachable/slow Gemini never blocks a trading cycle —
    the router's own fallback chain handles failures instead of waiting.
    """
    global _client
    if _client is None:
        from google import genai
        from google.genai import types as genai_types

        if not settings.GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY not configured.")
        _client = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            http_options=genai_types.HttpOptions(
                timeout=max(1.0, float(settings.LLM_REQUEST_TIMEOUT_SECONDS)),
                retry_options=genai_types.HttpRetryOptions(
                    attempts=1 + max(0, int(settings.LLM_MAX_RETRIES)),
                ),
            ),
        )
    return _client


def _extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from a model response,
    tolerating stray markdown fences or extra text around the JSON."""
    text = text.strip()
    text = re.sub(r"^```(json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model response.")
    return json.loads(match.group(0))


def _normalize_model(model: str) -> str:
    """Accepts both 'gemini-3.6-flash' and 'models/gemini-3.6-flash'."""
    return model[len("models/"):] if model.startswith("models/") else model


def generate_raw(system_instructions: str, input_text: str, model: str = None) -> str:
    """
    Runs one stateless Gemini interaction and returns the model's raw
    output text. This is the transport used by services.llm_service (the
    common provider layer); JSON parsing/quota/retry policy live there.

    Args:
        system_instructions: the agent's system prompt (role + output contract)
        input_text: the user input carrying the data to analyze
        model: optional model id override (defaults to settings.GEMINI_MODEL)

    Returns:
        the interaction's output text (stripped)

    Raises:
        RuntimeError if the API key is missing; whatever the SDK raises on
        API/transport errors. Callers classify and degrade gracefully.
    """
    client = _get_client()

    interaction = client.interactions.create(
        model=_normalize_model(model or settings.GEMINI_MODEL),
        input=input_text,
        system_instruction=system_instructions,
        store=False,  # stateless analysis calls; no server-side retention
    )

    raw_text = (getattr(interaction, "output_text", None) or "").strip()
    if not raw_text:
        # Defensive: surface interaction-level errors if any were reported.
        errors = getattr(interaction, "errors", None)
        if errors:
            raise RuntimeError(f"Gemini interaction returned errors: {errors}")
        raise ValueError("Empty response from Gemini interaction.")
    return raw_text


def generate_json(system_instructions: str, input_text: str) -> dict:
    """
    Runs one Gemini interaction and returns the model's response parsed as
    a JSON object. Kept for direct callers and tests; the agent pipeline
    goes through services.llm_service (same transport, plus quota handling,
    usage accounting and status classification).

    Args:
        system_instructions: the agent's system prompt (role + output contract)
        input_text: the user input carrying the data to analyze

    Returns:
        dict parsed from the model's JSON response

    Raises:
        RuntimeError if the API key is missing; whatever the SDK raises on
        API/transport errors; ValueError if the response is not parseable
        JSON. Callers are expected to catch and degrade gracefully.
    """
    return _extract_json(generate_raw(system_instructions, input_text))

