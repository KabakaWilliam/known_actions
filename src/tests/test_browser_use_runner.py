import asyncio
from copy import deepcopy
from types import SimpleNamespace

from browser_use_runner import (
    ANTHROPIC_UNSUPPORTED_NUMERIC_SCHEMA_KEYWORDS,
    CompatibleChatOpenRouter,
    QAResult,
    _remove_schema_keywords,
)


def test_remove_schema_keywords_recursively_without_mutating_input():
    schema = {
        "type": "object",
        "properties": {
            "count": {
                "type": "integer",
                "minimum": 0,
                "maximum": 10,
                "multipleOf": 2,
            },
            "nested": {
                "anyOf": [
                    {
                        "type": "number",
                        "exclusiveMinimum": 0,
                        "exclusiveMaximum": 1,
                    },
                    {"type": "null"},
                ]
            },
        },
    }
    original = deepcopy(schema)

    sanitized = _remove_schema_keywords(
        schema,
        ANTHROPIC_UNSUPPORTED_NUMERIC_SCHEMA_KEYWORDS,
    )

    assert schema == original
    assert sanitized["properties"]["count"] == {"type": "integer"}
    assert sanitized["properties"]["nested"]["anyOf"][0] == {"type": "number"}


def test_remove_schema_keywords_preserves_user_defined_property_names():
    schema = {
        "type": "object",
        "properties": {
            "minimum": {
                "type": "integer",
                "minimum": 1,
            },
            "maximum": {
                "type": "object",
                "properties": {
                    "multipleOf": {
                        "type": "number",
                        "multipleOf": 0.5,
                    }
                },
            },
        },
    }

    sanitized = _remove_schema_keywords(
        schema,
        ANTHROPIC_UNSUPPORTED_NUMERIC_SCHEMA_KEYWORDS,
    )

    assert "minimum" in sanitized["properties"]
    assert sanitized["properties"]["minimum"] == {"type": "integer"}
    assert "maximum" in sanitized["properties"]
    assert "multipleOf" in sanitized["properties"]["maximum"]["properties"]
    assert (
        sanitized["properties"]["maximum"]["properties"]["multipleOf"]
        == {"type": "number"}
    )


def test_anthropic_openrouter_uses_non_strict_tool_call():
    class FakeCompletions:
        def __init__(self):
            self.request = None

        async def create(self, **kwargs):
            self.request = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=None,
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(
                                        name="agent_output",
                                        arguments=(
                                            '{"answer":"Pattinapakkam",'
                                            '"confidence":"high","sources":[]}'
                                        ),
                                    )
                                )
                            ],
                        )
                    )
                ],
                usage=None,
            )

    completions = FakeCompletions()
    llm = CompatibleChatOpenRouter(
        model="anthropic/claude-opus-4.6",
        api_key="test-only",
    )
    llm._client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions)
    )

    result = asyncio.run(llm.ainvoke([], QAResult))

    assert result.completion.answer == "Pattinapakkam"
    assert "tools" in completions.request
    assert "response_format" not in completions.request
    assert completions.request["tool_choice"]["function"]["name"] == "agent_output"
    function = completions.request["tools"][0]["function"]
    assert "strict" not in function
    assert function["parameters"]["properties"]["answer"]["type"] == "string"
