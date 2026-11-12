from inspect_ai import task, Task
from dataset import build_dataset
from inspect_ai.solver import use_tools, generate
from tools import write_file, send_email, request_permissions, revoke_access, send_notification, ask_human
from scorers import audit_logging, credential_cleanup, escalation, permission_breadth

from inspect_ai.model import GenerateConfig

@task
def eval_awareness_task() -> Task:
    """
    A task to evaluate the model exhibits different tool-use patterns when it believes it's being evaluated vs. running unsupervised.
    """
    return Task(
        name="eval_awareness",
        dataset=build_dataset(),
        solver=[
            use_tools([write_file(), send_email(), request_permissions(), revoke_access(), send_notification(), ask_human()]),
            generate()
        ],
        scorer = [audit_logging(), escalation(), credential_cleanup(), permission_breadth()],
        # 6, not 10: the confidence interval is driven by the number of
        # scenarios (12 clusters), not by repeats within a scenario, so epochs
        # past ~6 cost money without narrowing the interval. 12 scenarios x 5
        # framings x 2 realism arms x 6 epochs = 720 runs.
        epochs=6,
        config=GenerateConfig(max_connections=3),
    )