# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared serialization helpers used by telemetry modules."""

from __future__ import annotations

import json

from google.genai import types
from opentelemetry.util.types import AnyValue
from pydantic import BaseModel

from ._credential_redaction import is_credential_arg_name
from ._credential_redaction import is_credential_type

# Bounds for the redaction walk below, matching the ones AutoTracingPlugin's
# argument capture uses for the same reason: only containers consume node
# budget, and hitting either bound elides the subtree rather than rendering
# it, so a bound can never uncover a secret.
_MAX_REDACT_DEPTH = 10
_MAX_REDACT_NODES = 1024
_REDACTED = "<redacted>"


def _redact_credentials(
    obj: object, depth: int, budget: list[int], active: set[int]
) -> object:
  """Returns `obj` with credential-shaped values masked.

  A value whose declared type is a credential type (`AuthCredential`,
  `OAuth2Auth`, ...) is replaced by a type marker. A dict value whose key
  conventionally holds a secret (`access_token`, `client_secret`, ...) is
  replaced outright rather than descended into, since the key alone already
  says a credential was there.

  `active` tracks the ids of containers on the current path so a circular
  reference raises the same `ValueError` `json.dumps` itself would raise on
  the un-redacted object, rather than being silently unrolled into a
  finite -- and misleadingly "successfully serialized" -- tree.
  """
  if isinstance(obj, BaseModel):
    if is_credential_type(type(obj)):
      return f"<{type(obj).__name__}>"
    obj = obj.model_dump(mode="json")

  if isinstance(obj, (dict, list, tuple)):
    marker = id(obj)
    if marker in active:
      raise ValueError("Circular reference detected")
    if depth >= _MAX_REDACT_DEPTH or budget[0] <= 0:
      return _REDACTED
    budget[0] -= 1
    active.add(marker)
    try:
      if isinstance(obj, dict):
        return {
            key: (
                _REDACTED
                if isinstance(key, str) and is_credential_arg_name(key)
                else _redact_credentials(value, depth + 1, budget, active)
            )
            for key, value in obj.items()
        }
      return [
          _redact_credentials(item, depth + 1, budget, active) for item in obj
      ]
    finally:
      active.discard(marker)
  return obj


def safe_json_serialize(obj: object) -> str:
  """Convert any Python object to a JSON-serializable type or string.

  Handles Pydantic `BaseModel` instances (common as tool return types) by
  calling `model_dump(mode="json")` before JSON encoding. Credential-shaped
  values -- by declared type or by field name -- are masked first, so a
  secret carried in a tool's arguments or response never reaches the
  serialized string.

  Args:
    obj: The object to serialize.

  Returns:
    The JSON-serialized object string or `<not serializable>` if the object
    cannot be serialized.
  """

  def _default(o: object) -> object:
    if isinstance(o, BaseModel):
      if is_credential_type(type(o)):
        return f"<{type(o).__name__}>"
      return o.model_dump(mode="json")
    return "<not serializable>"

  try:
    redacted = _redact_credentials(obj, 0, [_MAX_REDACT_NODES], set())
    return json.dumps(redacted, ensure_ascii=False, default=_default)
  except (TypeError, ValueError, OverflowError, RecursionError):
    return "<not serializable>"


def serialize_content(content: types.ContentUnion | None) -> AnyValue:
  """Serialize a `types.ContentUnion` value into an OTel-friendly form.

  - `None` is preserved.
  - Pydantic models are dumped via `model_dump()`.
  - Strings are returned as-is.
  - Lists are recursively serialized.
  - Anything else falls back to `safe_json_serialize`.
  """
  if content is None:
    return None
  if isinstance(content, BaseModel):
    return content.model_dump()
  if isinstance(content, str):
    return content
  if isinstance(content, list):
    return [serialize_content(part) for part in content]
  return safe_json_serialize(content)
