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

# pylint: disable=protected-access

import asyncio
import contextvars
import logging
import time
from unittest import mock

from google.adk.telemetry import _instrumentation
from google.adk.telemetry import _metrics
from google.adk.telemetry import tracing
from opentelemetry import context as context_api
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
import pytest


def test_get_elapsed_s_span_none():
  """Tests fallback when span is None."""
  start_time = 10.0
  with mock.patch("time.monotonic", return_value=12.0):
    elapsed = _metrics.get_elapsed_s(None, start_time)
  assert elapsed == 2.0  # 12 - 10


def test_get_elapsed_s_span_valid():
  """Tests duration calculation with valid span times."""
  mock_span = mock.MagicMock(spec=trace.Span)
  mock_span.start_time = 1000000000  # 1s in ns
  mock_span.end_time = 2000000000  # 2s in ns
  elapsed = _metrics.get_elapsed_s(mock_span, time.monotonic())
  assert elapsed == 1.0  # (2 - 1) s


def test_get_elapsed_s_span_missing_start():
  """Tests fallback when start_time is missing."""
  mock_span = mock.MagicMock(spec=trace.Span)
  del mock_span.start_time
  mock_span.end_time = 2000000000
  start_time = 10.0
  with mock.patch("time.monotonic", return_value=12.0):
    elapsed = _metrics.get_elapsed_s(mock_span, start_time)
  assert elapsed == 2.0


def test_get_elapsed_s_span_missing_end():
  """Tests fallback when end_time is missing."""
  mock_span = mock.MagicMock(spec=trace.Span)
  mock_span.start_time = 1000000000
  del mock_span.end_time
  start_time = 10.0
  with mock.patch("time.monotonic", return_value=12.0):
    elapsed = _metrics.get_elapsed_s(mock_span, start_time)
  assert elapsed == 2.0


def test_get_elapsed_s_span_non_int_start():
  """Tests fallback when start_time is not an integer."""
  mock_span = mock.MagicMock(spec=trace.Span)
  mock_span.start_time = 1000000000.0
  mock_span.end_time = 2000000000
  start_time = 10.0
  with mock.patch("time.monotonic", return_value=12.0):
    elapsed = _metrics.get_elapsed_s(mock_span, start_time)
  assert elapsed == 2.0


def test_get_elapsed_s_span_non_int_end():
  """Tests fallback when end_time is not an integer."""
  mock_span = mock.MagicMock(spec=trace.Span)
  mock_span.start_time = 1000000000
  mock_span.end_time = 2000000000.0
  start_time = 10.0
  with mock.patch("time.monotonic", return_value=12.0):
    elapsed = _metrics.get_elapsed_s(mock_span, start_time)
  assert elapsed == 2.0


@pytest.mark.asyncio
async def test_record_tool_execution_forwards_detected_error_type():
  """A failure detected in the tool response reaches the duration metric."""
  tool = mock.MagicMock()
  tool.name = "sample_tool"
  agent = mock.MagicMock()
  agent.name = "sample_agent"

  with mock.patch.object(
      _metrics, "record_tool_execution_duration"
  ) as mock_record:
    async with _instrumentation.record_tool_execution(
        tool=tool,
        agent=agent,
        function_args={},
        invocation_context=mock.MagicMock(),
    ) as tel_ctx:
      tel_ctx.error_type = "MCP_TOOL_ERROR"

  mock_record.assert_called_once()
  assert mock_record.call_args.kwargs["error"] is None
  assert mock_record.call_args.kwargs["error_type"] == "MCP_TOOL_ERROR"


@pytest.mark.asyncio
async def test_record_invocation_no_error_on_early_close(monkeypatch, caplog):
  """Early-stop consumers must not trigger an OTel 'Failed to detach' ERROR.

  Regression test for https://github.com/google/adk-python/issues/6559: when
  a caller stops iterating the invocation's async generator early, the
  generator is finalized (GeneratorExit) later, in a different execution
  context than the one where `record_invocation`'s span context was
  attached. Detaching that token from the wrong context raises "Token was
  created in a different Context" -- OTel's `context.detach()` swallows the
  exception but logs it at ERROR via the `opentelemetry.context` logger.
  """
  real_tracer = TracerProvider().get_tracer(__name__)
  monkeypatch.setattr(
      tracing.tracer, "start_as_current_span", real_tracer.start_as_current_span
  )
  monkeypatch.setattr(tracing.tracer, "start_span", real_tracer.start_span)

  async def _agen():
    with _instrumentation.record_invocation(None, "conversation-id"):
      yield 1
      yield 2

  gen = _agen()
  await gen.__anext__()  # Resume (and attach the span context) in this task.

  # Simulate the generator being finalized later from a *different*
  # execution context than the one it was resumed in -- e.g. via garbage
  # collection or `loop.shutdown_asyncgens()` after the caller returns early,
  # which is what actually happens in the reported scenario.
  fresh_context = contextvars.Context()
  close_task = fresh_context.run(asyncio.ensure_future, gen.aclose())

  with caplog.at_level(logging.ERROR, logger="opentelemetry.context"):
    await close_task

  assert "Failed to detach context" not in caplog.text


@pytest.mark.asyncio
async def test_record_invocation_detaches_context_on_same_context_exception(
    monkeypatch,
):
  """An ordinary exception raised in the same context must still detach.

  Regression test: unlike the early-close case above (where the generator is
  abandoned and finalized in a *different* context), an exception raised
  synchronously inside the `with record_invocation(...):` block propagates
  in the *same* context it was attached in, so the span's context token must
  still be detached -- otherwise the span leaks onto the ambient context and
  corrupts parent/child relationships for every later span created in this
  execution context (e.g. a reused asyncio task or thread-pool thread).
  """
  real_tracer = TracerProvider().get_tracer(__name__)
  monkeypatch.setattr(
      tracing.tracer, "start_as_current_span", real_tracer.start_as_current_span
  )
  monkeypatch.setattr(tracing.tracer, "start_span", real_tracer.start_span)

  before = dict(context_api.get_current())

  with pytest.raises(ValueError):
    with _instrumentation.record_invocation(None, "conversation-id"):
      raise ValueError("boom")

  after = dict(context_api.get_current())

  assert after == before
