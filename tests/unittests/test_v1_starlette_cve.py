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

"""Guard test for the v1-branch starlette/fastapi CVE fix.

Versions of starlette before 1.3.1 are affected by CVE-2026-48710,
CVE-2026-48818, and CVE-2026-54283 (see
https://github.com/advisories/GHSA-86qp-5c8j-p5mr and the related advisories
linked from it). fastapi<0.133 pins starlette to ``<1``, so bumping the
starlette floor alone would leave the resolver unable to satisfy both
constraints; the fastapi floor must move in lockstep.
"""

from __future__ import annotations

from pathlib import Path

try:
  import tomllib
except ImportError:
  import tomli as tomllib

from packaging.requirements import Requirement
from packaging.version import Version
import pytest

_PYPROJECT_PATH = Path(__file__).resolve().parents[2] / 'pyproject.toml'


@pytest.fixture(scope='module')
def dependencies() -> dict[str, Requirement]:
  with _PYPROJECT_PATH.open('rb') as fh:
    data = tomllib.load(fh)
  return {
      Requirement(raw).name: Requirement(raw)
      for raw in data['project']['dependencies']
  }


def test_starlette_floor_excludes_known_cves(
    dependencies: dict[str, Requirement],
) -> None:
  starlette = dependencies['starlette']
  assert Version('1.3.0') not in starlette.specifier, (
      'starlette<1.3.1 is affected by CVE-2026-48710, CVE-2026-48818, and '
      'CVE-2026-54283.'
  )
  assert (
      Version('1.3.1') in starlette.specifier
  ), 'starlette must allow 1.3.1, the first release that clears the known CVEs.'


def test_fastapi_floor_permits_fixed_starlette(
    dependencies: dict[str, Requirement],
) -> None:
  fastapi = dependencies['fastapi']
  assert Version('0.124.1') not in fastapi.specifier, (
      'fastapi<0.133 pins starlette<1, which conflicts with the starlette '
      '>=1.3.1 floor required to clear known CVEs.'
  )
  assert Version('0.133.1') in fastapi.specifier, (
      'fastapi must allow 0.133.1, which relaxes the starlette pin enough '
      'to permit installing a CVE-fixed starlette release.'
  )
