"""History is what makes follow-ups work, and only what we send counts."""

from history import Conversation
from llm import FakeLlm, LlmError, Message, Reply


def ask(convo: Conversation, llm: FakeLlm, text: str) -> Reply | None:
    """One turn, the way chat.py does it."""
    try:
        reply = llm.complete(convo.instructions, convo.messages_for(text))
    except LlmError:
        return None
    convo.commit(text, reply)
    return reply


def test_follow_up_is_sent_with_the_earlier_turns():
    llm = FakeLlm("Which city are you in?", "It's usually warm in Pune.", "Tomorrow looks similar.")
    convo = Conversation("Be brief.")
    ask(convo, llm, "What's the weather like?")
    ask(convo, llm, "I'm in Pune.")
    ask(convo, llm, "What about tomorrow?")
    instructions, sent = llm.calls[-1]
    assert instructions == "Be brief."
    assert sent == [Message("user", "What's the weather like?"),
                    Message("assistant", "Which city are you in?"),
                    Message("user", "I'm in Pune."),
                    Message("assistant", "It's usually warm in Pune."),
                    Message("user", "What about tomorrow?")]


def test_without_history_the_follow_up_arrives_alone():
    llm = FakeLlm("It's usually warm in Pune.", "Tomorrow where?")
    convo = Conversation(max_turns=0)
    ask(convo, llm, "What's the weather in Pune?")
    ask(convo, llm, "What about tomorrow?")
    assert llm.calls[-1][1] == [Message("user", "What about tomorrow?")]


def test_failed_reply_is_not_saved_and_neither_is_its_question():
    llm = FakeLlm("First answer.", LlmError("network", "dropped", partial_text="Half an"), "Third.")
    convo = Conversation()
    ask(convo, llm, "one")
    ask(convo, llm, "two")                       # fails mid-stream
    ask(convo, llm, "three")
    assert [m.text for m in llm.calls[-1][1]] == ["one", "First answer.", "three"]


def test_incomplete_reply_is_not_saved():
    cut = Reply("Once upon a", "fake", "m", status="incomplete", reason="max_output_tokens")
    convo = Conversation()
    assert convo.commit("tell me a story", cut) is False
    assert convo.turns == []


def test_trim_by_turn_count_drops_whole_oldest_turns():
    convo = Conversation(max_turns=2)
    for i in range(4):
        convo.messages_for(f"q{i}")
        convo.commit(f"q{i}", Reply(f"a{i}", "fake", "m"))
    sent = convo.messages_for("q4")
    assert [m.text for m in sent] == ["q2", "a2", "q3", "a3", "q4"]
    assert convo.forgotten == 2
    assert sent[0].role == "user"                # never starts with an orphaned answer


def test_trim_by_token_budget():
    convo = Conversation("x" * 40, max_tokens=40)          # instructions alone ~10 tokens
    for i in range(5):
        convo.messages_for("q" * 40)
        convo.commit("q" * 40, Reply("a" * 40, "fake", "m"))   # each turn ~20 tokens
    sent = convo.messages_for("new")
    assert convo.tokens(sent) <= 40
    assert len(sent) == 3 and sent[-1].text == "new"       # one old turn + the new message


def test_new_message_is_sent_even_when_alone_it_is_over_budget():
    convo = Conversation(max_tokens=5)
    convo.commit("hi", Reply("hello", "fake", "m"))
    sent = convo.messages_for("w" * 400)
    assert [m.text for m in sent] == ["w" * 400]


def test_stored_history_never_exceeds_max_turns():
    convo = Conversation(max_turns=1)
    convo.commit("q1", Reply("a1", "fake", "m"))
    convo.commit("q2", Reply("a2", "fake", "m"))
    assert [t.user.text for t in convo.turns] == ["q2"] and convo.forgotten == 1
