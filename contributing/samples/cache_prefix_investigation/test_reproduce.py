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

"""Checks for the offline reproduction, not tests of an implemented fork API."""

from __future__ import annotations

import asyncio
import time

from google.adk.models.cache_metadata import CacheMetadata
from google.adk.models.llm_request import LlmRequest
from google.genai import types
import pytest
from reproduce import CACHE_CONFIG
from reproduce import content
from reproduce import manager
from reproduce import run_investigation


@pytest.fixture
def cached_request():
  parent = LlmRequest(
      model='gemini-2.5-flash',
      cache_config=CACHE_CONFIG,
      config=types.GenerateContentConfig(
          system_instruction='Read-only shared instructions.',
          tools=[
              types.Tool(
                  function_declarations=[
                      types.FunctionDeclaration(
                          name='read_evidence', description='Read'
                      )
                  ]
              )
          ],
      ),
      contents=[
          content('user', 'Shared evidence.'),
          content('model', 'Research.'),
      ],
  )
  cache = manager()
  parent.cache_metadata = CacheMetadata(
      cache_name='cachedContents/synthetic-fixture',
      fingerprint=cache._generate_cache_fingerprint(parent, 2),
      contents_count=2,
      expire_time=time.time() + 300,
      invocations_used=1,
      created_at=time.time(),
  )
  return parent


@pytest.mark.asyncio
async def test_real_handoffs_do_not_preserve_the_tested_parent_prefix():
  """Real Runner/AgentTool requests expose the specific configuration gap."""
  result, _ = await run_investigation()
  sequential = result['sequential_agents']
  tool = result['agent_tool']
  assert sequential['shared_context_present']
  assert sequential['different_prefix_fields'] == ['system_instruction']
  assert sequential['parent_roles'][1] == 'model'
  assert sequential['child_roles'][1] == 'user'
  assert sequential['child_cache_config_present']
  assert not tool['shared_context_present']
  assert not tool['child_cache_config_present']
  assert result['metadata_lookup']['parent_found']
  assert not result['metadata_lookup']['child_found_parent_metadata']
  assert not result['ordinary_child_with_parent_metadata_locally_valid']
  assert result['request_snapshot_control_locally_valid']
  assert result['actual_cached_tokens'] is None
  graph = result['workflow_default_single_turn']
  assert not graph['shared_context_present']
  assert graph['different_prefix_fields'] == []
  assert graph['equal_leading_contents'] == 0
  assert graph['child_cache_config_present']


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'change', ['model', 'instruction', 'tool', 'role', 'text']
)
async def test_incompatible_prefix_is_rejected_locally(cached_request, change):
  """Changing a cached field rejects the seeded parent's explicit cache."""
  child = cached_request.model_copy(deep=True)
  if change == 'model':
    child.model = 'gemini-2.5-pro'
  elif change == 'instruction':
    child.config.system_instruction = 'Different instructions.'
  elif change == 'tool':
    child.config.tools[0].function_declarations[0].name = 'different_tool'
  elif change == 'role':
    child.contents[1].role = 'user'
  else:
    child.contents[0].parts[0].text = 'Different evidence.'

  assert not await manager()._is_cache_valid(child)


@pytest.mark.asyncio
async def test_expired_cache_is_not_reusable(cached_request):
  """An expired cache cannot satisfy the local compatibility check."""
  cached_request.cache_metadata = cached_request.cache_metadata.model_copy(
      update={'expire_time': time.time() - 1}
  )

  assert not await manager()._is_cache_valid(cached_request)


@pytest.mark.asyncio
async def test_parallel_request_controls_do_not_mutate_parent_or_sibling(
    cached_request,
):
  """Separate deep request copies isolate mutation by cache application."""
  before = cached_request.model_dump(mode='json')
  children = [cached_request.model_copy(deep=True) for _ in range(2)]
  for index, child in enumerate(children):
    child.contents.append(content('user', f'Review task {index}.'))

  await asyncio.gather(
      *[manager().handle_context_caching(child) for child in children]
  )

  assert cached_request.model_dump(mode='json') == before
  assert children[0].contents[0].parts[0].text == 'Review task 0.'
  assert children[1].contents[0].parts[0].text == 'Review task 1.'
  children[0].contents[0].parts[0].text = 'Changed child.'
  assert children[1].contents[0].parts[0].text == 'Review task 1.'


@pytest.mark.asyncio
async def test_completed_tool_pair_survives_request_snapshot_control(
    cached_request,
):
  """The local positive control preserves a complete call/response pair."""
  cached_request.contents.extend([
      types.Content(
          role='model',
          parts=[types.Part.from_function_call(name='read_evidence', args={})],
      ),
      types.Content(
          role='user',
          parts=[
              types.Part.from_function_response(
                  name='read_evidence', response={'result': 'Evidence.'}
              )
          ],
      ),
  ])
  cached_request.cache_metadata = cached_request.cache_metadata.model_copy(
      update={
          'contents_count': 4,
          'fingerprint': (
              manager()._generate_cache_fingerprint(cached_request, 4)
          ),
      }
  )
  fork = cached_request.model_copy(deep=True)
  fork.contents.append(content('user', 'Review.'))

  assert await manager()._is_cache_valid(fork)
  assert fork.contents[:4] == cached_request.contents
