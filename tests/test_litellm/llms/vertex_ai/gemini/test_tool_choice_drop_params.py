"""VertexAI must honour drop_params for an out-of-range tool_choice.

Same defect and same fix as the Bedrock Converse mapper: the error message tells
the caller to set drop_params, and the branch raised regardless, so a proxy with
drop_params already on still returned 400. Anthropic spells forced tool use as
tool_choice='any', so an Anthropic-shaped client pointed at the OpenAI-compatible
surface hits it on every tool call.
"""

import pytest

import litellm
from litellm.llms.vertex_ai.gemini.vertex_and_google_ai_studio_gemini import (
    VertexGeminiConfig,
)


@pytest.mark.parametrize("tool_choice", ["any", "AUTO", "some-future-mode"])
def test_unsupported_tool_choice_dropped_when_drop_params_set(tool_choice, monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", False)
    config = VertexGeminiConfig()

    assert (
        config.map_tool_choice_values(
            model="gemini-2.0-flash", tool_choice=tool_choice, drop_params=True
        )
        is None
    )

    monkeypatch.setattr(litellm, "drop_params", True)
    assert (
        config.map_tool_choice_values(
            model="gemini-2.0-flash", tool_choice=tool_choice, drop_params=False
        )
        is None
    )


def test_unsupported_tool_choice_still_raises_without_drop_params(monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", False)
    config = VertexGeminiConfig()

    with pytest.raises(litellm.utils.UnsupportedParamsError, match="tool_choice=any"):
        config.map_tool_choice_values(
            model="gemini-2.0-flash", tool_choice="any", drop_params=False
        )


@pytest.mark.parametrize(
    "tool_choice, expected_mode",
    [("none", "NONE"), ("required", "ANY"), ("auto", "AUTO")],
)
def test_supported_tool_choice_values_unaffected(tool_choice, expected_mode, monkeypatch):
    """The supported values must keep mapping, with or without drop_params."""
    monkeypatch.setattr(litellm, "drop_params", True)
    config = VertexGeminiConfig()

    result = config.map_tool_choice_values(
        model="gemini-2.0-flash", tool_choice=tool_choice, drop_params=True
    )

    assert result is not None
    assert result["functionCallingConfig"]["mode"] == expected_mode


def test_map_openai_params_threads_drop_params_through(monkeypatch):
    """The caller must pass drop_params down, not rely on the global alone."""
    monkeypatch.setattr(litellm, "drop_params", False)
    config = VertexGeminiConfig()

    optional_params = config.map_openai_params(
        non_default_params={"tool_choice": "any"},
        optional_params={},
        model="gemini-2.0-flash",
        drop_params=True,
    )

    assert "tool_choice" not in optional_params
