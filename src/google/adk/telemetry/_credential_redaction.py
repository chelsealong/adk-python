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

"""Shared credential-shape detection, used to keep secrets out of telemetry.

Both the auto-tracing plugin's argument capture and the default span
serializer need to recognize the same credential-bearing types and field
names, so the lists live here once instead of drifting apart in two files.
"""

from __future__ import annotations

import functools
from typing import Callable

# Types whose repr()/dump renders live secrets (tokens, keys, passwords).
# Matched by name over the MRO so this module never imports `google.adk.auth`.
CREDENTIAL_TYPE_NAMES = frozenset({
    "AuthConfig",
    "AuthCredential",
    "AuthToolArguments",
    "Credentials",
    "HttpAuth",
    "HttpCredentials",
    "OAuth2Auth",
    "OAuth2Session",
    "ServiceAccount",
    "ServiceAccountCredential",
})
# Parameter/field names that conventionally carry secret material.
CREDENTIAL_ARG_NAMES = frozenset({
    "api_key",
    "auth_config",
    "auth_credential",
    "authorization",
    "cookie",
    "cookies",
    "credential",
    "credentials",
    "password",
    "private_key",
    "secret",
    "token",
})
CREDENTIAL_ARG_SUFFIXES = (
    "_api_key",
    "_auth_config",
    "_authorization",
    "_cookie",
    "_cookies",
    "_credential",
    "_credentials",
    "_password",
    "_private_key",
    "_secret",
    "_token",
)


def _mro_holds_credential(cls: type) -> bool:
  """True iff `cls` or one of its bases is a credential-bearing type."""
  return any(k.__name__ in CREDENTIAL_TYPE_NAMES for k in cls.__mro__)


# Cached because callers ask this of every non-scalar node they visit.
is_credential_type: Callable[[type], bool] = functools.lru_cache(maxsize=512)(
    _mro_holds_credential
)


@functools.lru_cache(maxsize=1024)
def is_credential_arg_name(name: str) -> bool:
  """True iff a parameter/field called `name` conventionally holds a secret."""
  lowered = name.lower()
  return lowered in CREDENTIAL_ARG_NAMES or lowered.endswith(
      CREDENTIAL_ARG_SUFFIXES
  )
