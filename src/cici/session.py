"""Conversation state and context management.

This class stores content and nothing else. Unwrapping a provider response into
content is llm.py::Claude.content_from_message — knowing the shape of a Message
is provider knowledge, and letting it leak in here would put an Anthropic type
behind the seam. cici_101/cli_project/core/claude.py did both jobs in one method
only because that repo had no conversation object to hand the second one to.

TODO (roadmap 3): persistence to .cici/sessions/, context window management
(what to keep, what to summarise, what to drop), @file mention injection as
<document id="..."> blocks (ported from cici_101/cli_project/core/cli_chat.py).
"""


class Session:
    def __init__(self, system_prompt=""):
        self.system_prompt = system_prompt
        self.messages = []

    def add_user(self, content):
        """Append a user turn.

        content is whatever the Messages API accepts for a user turn: a plain
        string (raw input from the REPL) or a list of content blocks (the
        tool_result dicts Registry.run_all returns).
        """
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content):
        """Append an assistant turn.

        content is a list of content blocks, already unwrapped by the provider
        — pass Claude.content_from_message(response), not the response itself.
        """
        self.messages.append({"role": "assistant", "content": content})

    def __len__(self):
        return len(self.messages)
