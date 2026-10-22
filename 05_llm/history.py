"""Conversation history: the model's only memory is what WE send it.

Every call is independent. The model doesn't remember the last request,
so each turn we resend everything it should know:

    turn 1 sends   instructions + [user 1]
    turn 2 sends   instructions + [user 1, assistant 1, user 2]
    turn 3 sends   instructions + [user 1, assistant 1, user 2, assistant 2, user 3]

So input tokens (cost and time-to-first-token) grow every turn, until the
context window is full. `Conversation` keeps that growth under control by
forgetting the oldest turns: by count (`max_turns`) and/or by an estimated
token budget (`max_tokens`). It always drops a whole turn (question AND
answer). Half a turn would leave an answer without its question.

Two rules keep the history honest:
  - A turn is saved only after a COMPLETED reply. A failed or cut-off
    reply never enters history, and neither does its question, so the
    next request doesn't carry a question nobody answered.
  - The new user message and the instructions are never dropped. If they
    alone are over budget, they're sent anyway (and the API will say if
    they don't fit).

`max_turns=0` means no memory at all: watch "What about tomorrow?" fail.
"""

from dataclasses import dataclass

from llm import Message, Reply, estimate_tokens


@dataclass(frozen=True)
class Turn:
    user: Message
    assistant: Message


class Conversation:
    def __init__(self, instructions: str = "", max_turns: int | None = None,
                 max_tokens: int | None = None) -> None:
        if max_turns is not None and max_turns < 0:
            raise ValueError("max_turns can't be negative")
        self.instructions = instructions
        self.max_turns = max_turns
        self.max_tokens = max_tokens
        self.turns: list[Turn] = []
        self.forgotten = 0          # turns dropped so far, to make trimming visible

    def tokens(self, messages: list[Message]) -> int:
        """Estimated input tokens for instructions + these messages."""
        return estimate_tokens(self.instructions + "".join(m.text for m in messages))

    def messages_for(self, user_text: str) -> list[Message]:
        """What to send for this turn: the kept history + the new message.
        Trims (forgets) the oldest turns first if over a limit."""
        new = Message("user", user_text)
        while self.turns and self._over_limit(new):
            self.turns.pop(0)
            self.forgotten += 1
        return self.history() + [new]

    def _over_limit(self, new: Message) -> bool:
        if self.max_turns is not None and len(self.turns) > self.max_turns:
            return True
        return (self.max_tokens is not None
                and self.tokens(self.history() + [new]) > self.max_tokens)

    def history(self) -> list[Message]:
        return [m for t in self.turns for m in (t.user, t.assistant)]

    def commit(self, user_text: str, reply: Reply) -> bool:
        """Save the turn if the reply is a real, complete answer. Returns whether it was saved."""
        if not reply.complete or not reply.text:
            return False
        if self.max_turns == 0:
            return False
        self.turns.append(Turn(Message("user", user_text), Message("assistant", reply.text)))
        while self.max_turns is not None and len(self.turns) > self.max_turns:
            self.turns.pop(0)       # what's stored never exceeds the limit either
            self.forgotten += 1
        return True

    def clear(self) -> None:
        self.turns.clear()
        self.forgotten = 0
