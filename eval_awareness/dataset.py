from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import ChatMessage, ChatMessageSystem, ChatMessageUser

from scenarios import FRAMINGS, SCENARIOS


def build_dataset() -> MemoryDataset:
    samples: list[Sample] = []
    for scenario in SCENARIOS:
        for framing_name, framing in FRAMINGS.items():
            messages: list[ChatMessage] = []
            if framing is not None:
                messages.append(ChatMessageSystem(content=framing))
            messages.append(ChatMessageUser(content=scenario.prompt))

            samples.append(Sample(
                id=f"{scenario.id}_{framing_name}",
                input=messages,
                metadata={
                    "scenario_id": scenario.id,
                    "category": scenario.category,
                    "framing_condition": framing_name,
                    **scenario.metadata,
                },
            ))
    return MemoryDataset(samples)
