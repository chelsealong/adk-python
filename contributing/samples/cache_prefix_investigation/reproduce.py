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

"""Offline ADK request-boundary investigation; NOT a live cache benchmark.

Run with the pinned checkout on PYTHONPATH. Only the model/provider boundary
is simulated. ADK builds the requests and evaluates cache compatibility.
The synthetic cache handle never represents a provider-created resource.

From the checkout root:
  uv venv --python 3.11
  uv pip install -e . -r contributing/samples/cache_prefix_investigation/requirements.txt
  PYTHONPATH=src .venv/bin/python contributing/samples/cache_prefix_investigation/reproduce.py --output-dir /tmp/adk-prefix-results
  PYTHONPATH=src .venv/bin/python -m pytest contributing/samples/cache_prefix_investigation/test_reproduce.py -q

All data and responses are synthetic. No provider API is called.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from typing import AsyncGenerator

import google.adk
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.agents.invocation_context import InvocationContext
from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.run_config import RunConfig
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.apps.app import App
from google.adk.events.event import Event
from google.adk.models.base_llm import BaseLlm
from google.adk.models.cache_metadata import CacheMetadata
from google.adk.models.gemini_context_cache_manager import GeminiContextCacheManager
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.runners import Runner
from google.adk.sessions.in_memory_session_service import InMemorySessionService
from google.adk.tools.agent_tool import AgentTool
from google.adk.tools.tool_context import ToolContext
from google.adk.workflow import START
from google.adk.workflow import Workflow
from google.genai import types
from pydantic import PrivateAttr

PINNED_COMMIT = '6854b88272f84a6ca9b1e4dec979a6476a00e0d7'
MODEL = 'gemini-2.5-flash'
SHARED_INSTRUCTION = 'Use the supplied evidence. Do not take external actions.'
SHARED_CONTEXT = '\n'.join(
    f'Record {n}: synthetic research evidence for a read-only review.'
    for n in range(1600)
)
CACHE_CONFIG = ContextCacheConfig(min_tokens=0, ttl_seconds=300)


def content(role: str, text: str) -> types.Content:
  return types.Content(role=role, parts=[types.Part.from_text(text=text)])


def manager() -> GeminiContextCacheManager:
  # No real client or credentials. These are the only client properties read
  # by the local fingerprint/validity paths. Any network path would fail.
  return GeminiContextCacheManager(
      SimpleNamespace(vertexai=False, _api_client=None)
  )


class RecordingModel(BaseLlm):
  """Records real ADK requests and returns deterministic synthetic responses."""

  _requests: list[LlmRequest] = PrivateAttr(default_factory=list)

  async def generate_content_async(
      self, llm_request: LlmRequest, stream: bool = False
  ) -> AsyncGenerator[LlmResponse, None]:
    del stream
    self._requests.append(llm_request.model_copy(deep=True))
    yield LlmResponse(content=content('model', 'Synthetic model response.'))


def agent(name: str, model: BaseLlm, *, mode: str | None = 'chat') -> LlmAgent:
  return LlmAgent(
      name=name,
      model=model,
      mode=mode,
      static_instruction=SHARED_INSTRUCTION,
      tools=[lookup_record],
      disallow_transfer_to_parent=True,
      disallow_transfer_to_peers=True,
  )


def lookup_record(index: int) -> str:
  """Read one synthetic evidence record without external side effects."""
  return f'Record {index}: synthetic research evidence.'


async def seeded_session(service: InMemorySessionService):
  session = await service.create_session(
      app_name='prefix_repro', user_id='user'
  )
  for author, message in (
      ('user', content('user', SHARED_CONTEXT)),
      ('researcher', content('model', 'Prior research result.')),
  ):
    await service.append_event(
        session=session,
        event=Event(invocation_id='previous', author=author, content=message),
    )
  return session


def request_payload(request: LlmRequest) -> dict:
  return {
      'model': request.model,
      'config': request.config.model_dump(mode='json', exclude_none=True),
      'contents': [
          item.model_dump(mode='json', exclude_none=True)
          for item in request.contents
      ],
  }


def compare(parent: LlmRequest, child: LlmRequest) -> dict:
  """Compare structured fields; this does not claim provider token ordering."""
  left, right = request_payload(parent), request_payload(child)
  differences = []
  for field in ('model', 'system_instruction', 'tools', 'tool_config'):
    a = left['model'] if field == 'model' else left['config'].get(field)
    b = right['model'] if field == 'model' else right['config'].get(field)
    if a != b:
      differences.append(field)
  common = 0
  for a, b in zip(left['contents'], right['contents']):
    if a != b:
      break
    common += 1
  return {
      'different_prefix_fields': differences,
      'equal_leading_contents': common,
      'parent_contents': len(parent.contents),
      'child_contents': len(child.contents),
      'parent_roles': [c.role for c in parent.contents],
      'child_roles': [c.role for c in child.contents],
      'child_cache_config_present': child.cache_config is not None,
      'child_cache_metadata_present': child.cache_metadata is not None,
      'shared_context_present': any(
          SHARED_CONTEXT in (part.text or '')
          for item in child.contents
          for part in item.parts or []
      ),
  }


async def run_investigation() -> tuple[dict, dict]:
  model = RecordingModel(model=MODEL)
  service = InMemorySessionService()
  workflow = SequentialAgent(
      name='research_then_review',
      sub_agents=[agent('researcher', model), agent('reviewer', model)],
  )
  runner = Runner(
      app=App(
          name='prefix_repro',
          root_agent=workflow,
          context_cache_config=CACHE_CONFIG,
      ),
      session_service=service,
  )
  session = await seeded_session(service)
  try:
    async for _ in runner.run_async(
        user_id='user',
        session_id=session.id,
        new_message=content('user', 'Continue research, then review.'),
    ):
      pass
  finally:
    await runner.close()
  assert len(model._requests) == 2, len(model._requests)
  parent, sequential_child = model._requests

  # Current graph API in its default single-turn mode, separately from the
  # still-supported but deprecated SequentialAgent chat-mode reproduction.
  graph_model = RecordingModel(model=MODEL)
  graph_service = InMemorySessionService()
  graph = Workflow(
      name='research_then_review',
      edges=[(
          START,
          agent('researcher', graph_model, mode=None),
          agent('reviewer', graph_model, mode=None),
      )],
  )
  graph_runner = Runner(
      app=App(
          name='prefix_repro',
          root_agent=graph,
          context_cache_config=CACHE_CONFIG,
      ),
      session_service=graph_service,
  )
  graph_session = await seeded_session(graph_service)
  try:
    async for _ in graph_runner.run_async(
        user_id='user',
        session_id=graph_session.id,
        new_message=content('user', SHARED_CONTEXT + '\nResearch this.'),
    ):
      pass
  finally:
    await graph_runner.close()
  assert len(graph_model._requests) == 2
  graph_parent, graph_child = graph_model._requests

  # Explicitly seeded metadata is a fixture, not an observed cache hit.
  cache = manager()
  boundary = len(parent.contents) - 1
  assert boundary >= 2
  metadata = CacheMetadata(
      cache_name='cachedContents/synthetic-offline-fixture',
      expire_time=time.time() + 300,
      invocations_used=1,
      contents_count=boundary,
      fingerprint=cache._generate_cache_fingerprint(parent, boundary),
      created_at=time.time(),
  )
  session = await service.get_session(
      app_name='prefix_repro', user_id='user', session_id=session.id
  )
  await service.append_event(
      session=session,
      event=Event(
          invocation_id='parent_cache_fixture',
          author='researcher',
          content=content('model', 'Synthetic cache metadata fixture.'),
          cache_metadata=metadata,
      ),
  )

  # Real AgentTool -> real nested Runner; no child history is manually seeded.
  tool_model = RecordingModel(model=MODEL)
  tool_parent = agent('researcher', RecordingModel(model=MODEL))
  invocation = InvocationContext(
      invocation_id='agent_tool_repro',
      agent=tool_parent,
      session=session,
      session_service=service,
      run_config=RunConfig(max_llm_calls=3),
      context_cache_config=CACHE_CONFIG,
  )
  await AgentTool(agent=agent('reviewer', tool_model)).run_async(
      args={'request': 'Review the research.'},
      tool_context=ToolContext(invocation, function_call_id='review_call'),
  )
  assert len(tool_model._requests) == 1
  tool_child = tool_model._requests[0]

  # Parent metadata is not selected for another agent even in the same session.
  from google.adk.flows.llm_flows.context._cache import ContextCacheRequestProcessor

  processor = ContextCacheRequestProcessor()
  lookup_parent, _ = processor._find_cache_info_from_events(
      invocation, 'researcher', 'new_invocation'
  )
  lookup_child, _ = processor._find_cache_info_from_events(
      invocation, 'reviewer', 'new_invocation'
  )
  assert lookup_parent is not None and lookup_child is None

  # Handing over metadata alone does not make the ordinary child compatible.
  ordinary = sequential_child.model_copy(deep=True)
  ordinary.cache_metadata = metadata.model_copy(deep=True)
  ordinary_valid = await cache._is_cache_valid(ordinary)

  # Request-level positive control ONLY: not a new AgentTool/API and not proof
  # that cross-agent history can safely retain the parent's authority/roles.
  fork = parent.model_copy(deep=True)
  fork.cache_metadata = metadata.model_copy(deep=True)
  # Complete the simulated parent's current turn before adding the child task.
  fork.contents.append(content('model', 'Synthetic model response.'))
  fork.contents.append(content('user', 'Review the preceding research.'))
  fork_valid = await cache._is_cache_valid(fork)
  applied = fork.model_copy(deep=True)
  await cache.handle_context_caching(applied)
  assert applied.config.cached_content == metadata.cache_name
  assert len(applied.contents) == len(fork.contents) - boundary
  assert ordinary_valid is False and fork_valid is True

  return {
      'adk_commit': PINNED_COMMIT,
      'google_genai_version': importlib.metadata.version('google-genai'),
      'pydantic_version': importlib.metadata.version('pydantic'),
      'evidence_type': 'offline real ADK assembly; simulated model boundary',
      'shared_context_characters': len(SHARED_CONTEXT),
      'actual_cached_tokens': None,
      'provider_latency_seconds': None,
      'provider_cost': None,
      'task_quality_measured': False,
      'synthetic_cache_fixture': True,
      'sequential_agents': compare(parent, sequential_child),
      'workflow_default_single_turn': compare(graph_parent, graph_child),
      'agent_tool': compare(parent, tool_child),
      'metadata_lookup': {
          'parent_found': lookup_parent is not None,
          'child_found_parent_metadata': lookup_child is not None,
      },
      'ordinary_child_with_parent_metadata_locally_valid': ordinary_valid,
      'request_snapshot_control_locally_valid': fork_valid,
      'cached_prefix_contents_count': boundary,
      'limitations': [
          (
              'No provider requests, tokenization, timing, cost or quality'
              ' benchmark.'
          ),
          'No blanket claim about all ADK providers, modes or handoff paths.',
          'Snapshot control is not an implemented safe agent fork.',
          (
              'History fencing/role changes must not be removed to improve'
              ' caching.'
          ),
      ],
  }, {
      'parent': request_payload(parent),
      'sequential_child': request_payload(sequential_child),
      'agent_tool_child': request_payload(tool_child),
      'workflow_parent': request_payload(graph_parent),
      'workflow_child': request_payload(graph_child),
      'request_snapshot_control': request_payload(fork),
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--output-dir', type=Path, required=True)
  args = parser.parse_args()
  checkout = Path(__file__).resolve().parents[3]
  imported_checkout = Path(google.adk.__file__).resolve().parents[3]
  if imported_checkout != checkout:
    raise SystemExit('Set PYTHONPATH=src to import the inspected checkout.')
  head = subprocess.check_output(
      ['git', 'rev-parse', 'HEAD'], cwd=checkout, text=True
  ).strip()
  # A research commit can add this script without changing upstream runtime.
  changed = subprocess.check_output(
      ['git', 'diff', PINNED_COMMIT, '--', 'src/google/adk'],
      cwd=checkout,
      text=True,
  )
  if changed:
    raise SystemExit(
        'Runtime differs from pinned ADK; refuse mislabeled output.'
    )
  result, requests = asyncio.run(run_investigation())
  result['checkout_head'] = head
  result['adk_version'] = google.adk.__version__
  result['python_version'] = sys.version.split()[0]
  args.output_dir.mkdir(parents=True, exist_ok=True)
  raw = json.dumps(requests, indent=2, sort_keys=True) + '\n'
  result['requests_sha256'] = hashlib.sha256(raw.encode()).hexdigest()
  (args.output_dir / 'requests.json').write_text(raw)
  (args.output_dir / 'results.json').write_text(
      json.dumps(result, indent=2, sort_keys=True) + '\n'
  )
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
  main()
