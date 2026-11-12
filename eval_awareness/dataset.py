from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser

from scenarios import FRAMINGS, SCENARIOS
from system_prompts import REALISM


def system_message(operator: str | None, framing: str | None) -> ChatMessageSystem | None:
    """Combine operator context and framing cue into one system message.

    Deliberately one message rather than two: some providers collapse or drop a
    second system message, and if that happened on only some endpoints the
    framing manipulation would vanish from the rich arm without any error. The
    framing goes last so it is the most recent thing said before the task.
    """
    parts = [p for p in (operator, framing) if p is not None]
    if not parts:
        return None
    return ChatMessageSystem(content="\n\n".join(parts))


def build_dataset() -> MemoryDataset:
    # A rich arm with no operator prompt would be byte-identical to the thin arm,
    # so the run would cost full price and measure nothing. Fail before spending.
    for realism_name, operator in REALISM.items():
        if realism_name != "thin" and operator is None:
            raise ValueError(
                f"REALISM['{realism_name}'] is None, which would make it identical "
                "to the thin arm. Write OPERATOR_PROMPT in system_prompts.py first."
            )

    samples: list[Sample] = []
    for scenario in SCENARIOS:
        for framing_name, framing in FRAMINGS.items():
            for realism_name, operator in REALISM.items():
                system = system_message(operator, framing)
                messages: list[ChatMessage] = []
                if system is not None:
                    messages.append(system)
                messages.append(ChatMessageUser(content=scenario.prompt))

                samples.append(Sample(
                    id=f"{scenario.id}_{framing_name}_{realism_name}",
                    input=messages,
                    metadata={
                        "scenario_id": scenario.id,
                        "category": scenario.category,
                        "framing_condition": framing_name,
                        "realism": realism_name,
                        **scenario.metadata,
                    },
                ))
    return MemoryDataset(samples)
