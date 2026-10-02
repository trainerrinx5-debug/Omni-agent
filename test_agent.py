import json
import tempfile
import unittest
from pathlib import Path

from omni_agent.agent import Agent
from omni_agent.memory import MemoryStore
from omni_agent.safety import WriteApproval
from omni_agent.tools import build_tool_registry


class FakeClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def complete(self, messages, tools):
        self.requests.append((messages, tools))
        return self.responses.pop(0)


class AgentTests(unittest.TestCase):
    def test_tool_round_trip_then_final_answer(self):
        first = {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "calculator", "arguments": json.dumps({"expression": "6 * 7"})},
                    }],
                }
            }]
        }
        second = {"choices": [{"message": {"role": "assistant", "content": "The result is 42."}}]}
        client = FakeClient([first, second])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "workspace"
            registry = build_tool_registry(root, WriteApproval())
            memory = MemoryStore(Path(directory) / "state" / "memory.json")
            agent = Agent(client, registry, memory=memory)
            answer, history = agent.run_turn("What is 6 times 7?")

        self.assertEqual(answer, "The result is 42.")
        self.assertEqual(len(client.requests), 2)
        second_request_messages = client.requests[1][0]
        tool_result = next(item for item in second_request_messages if item.get("role") == "tool")
        self.assertIn('"result": 42', tool_result["content"])
        self.assertEqual(history[-1]["content"], "The result is 42.")
        self.assertIn("calculator", [item["function"]["name"] for item in client.requests[0][1]])

    def test_empty_prompt_rejected(self):
        client = FakeClient([])
        with tempfile.TemporaryDirectory() as directory:
            registry = build_tool_registry(Path(directory) / "workspace", WriteApproval())
            agent = Agent(client, registry)
            with self.assertRaises(ValueError):
                agent.run_turn("  ")


if __name__ == "__main__":
    unittest.main()
