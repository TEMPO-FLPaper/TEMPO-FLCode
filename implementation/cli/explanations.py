"""Implementation-based terminal explanations with portable ASCII diagrams."""

from rich.panel import Panel
from rich.text import Text
from implementation.cli.presentation import banner, table
from implementation.config.schema import METHODS

DESCRIPTIONS = {
    "flat": (
        "Flat FedAvg / Simple FedAvg",
        "Global (round start)\n  | broadcast to all 8 clients\n  v\nPrivate local SGD updates\n  | sample-weighted average\n  v\nNext global model",
        "Each client starts from the same round-start global weights. All eight clients participate. "
        "Client updates go directly to the server, weighted by private training support. "
        "Initial client test reporting follows first-round local training; initial global reporting follows its first aggregation. "
        "No persistent personalized client state or confidence cascade is used.",
    ),
    "hfl": (
        "Hierarchical FedAvg / Simple HFL",
        "L3 (round start)\n  | broadcast to clients\n  v\n8 local SGD updates\n  | within-context sample weights\n  v\n4 L2 context averages\n  | context training support\n  v\nNext L3 global model",
        "Clients start from L3 each round. Two clients form each context; their updates form L2, "
        "and the four context models form L3. Local outputs remain available as L1 evaluations. "
        "This source has plain sample-weighted aggregation without TEMPO's calibration, validation gates or selective promotion.",
    ),
    "ditto": (
        "Ditto / Global and personalized learning",
        "Round-start global w\n  | same local batch\n  +--> global path: CE update --> upload --> FedAvg\n  |\n  +--> persistent personal v\n       CE + lambda/2 * sum((v - w_start)^2)\n       retained locally; never uploaded",
        "On each local step, the plain global-path update runs first, then the personalized update on the same batch. "
        "The proximal anchor is the round-start global model, over trainable variables. Personalized weights persist between rounds. "
        "Only global-path updates are uploaded. Context names describe data groups; no context aggregation occurs.",
    ),
    "tempo": (
        "TEMPO-FL / Three separate processes",
        "1. FEDERATED CONTRIBUTION\n"
        "Private data --> received L2 shadow update\n"
        "            --> gated L2 --> gated L3\n"
        "            <-- safe top-down L3-to-L2 blend\n"
        "L1 specialist is retained. No routine L1 update.\n\n"
        "2. PER-INPUT INFERENCE\n"
        "Input --> calibrated L1\n"
        "          accept: prediction\n"
        "          reject --> cached L2\n"
        "                     accept: prediction\n"
        "                     reject --> cached L3\n"
        "                                accept: prediction\n"
        "                                reject: UNKNOWN\n\n"
        "3. PERSISTENT KNOWLEDGE TRANSFER\n"
        "Pooled reliance + higher acceptance + recent teachers\n"
        "  --> candidate from L1: replay CE, then teacher KD\n"
        "  --> preservation + KL/acceptance-gain gates\n"
        "      pass: replace L1, recalibrate, rebaseline\n"
        "      fail: retain L1; attempt cooldown",
        "L1 specialists start from generic ImageNet MobileNetV2 weights adapted to TrashNet, then owned-class specialization. "
        "L2/L3 are cached shadows: routing accepts the first calibrated confidence at or above that level's threshold. "
        "Unknown is abstention (-1), not a seventh trained class. Acceptance is distinct from correctness measured using audit labels. "
        "The detector uses a full pooled window plus current reliance, sufficient escalations, pooled higher acceptance, recent teachers and cooldowns. "
        "Teacher logits are unlabeled operational supervision; replay uses private training labels. Replay CE and temperature-squared KD are separate updates. "
        "The Trust-Region mechanism searches safe blending coefficients with recall/ECE and score safeguards; it has no explicit norm-radius constraint. "
        "The offline benchmark constructs shifts using online labels; operational routing and promotion do not use those labels or final-test labels.",
    ),
}


def explain(console, method: str | None = None) -> None:
    banner(console, "How the methods work")
    for key in [method] if method else METHODS:
        title, diagram, description = DESCRIPTIONS[key]
        console.print(Panel(Text(diagram), title=title, expand=True))
        console.print(Text(description))
        console.print()
    table(
        console,
        ["Method", "Aggregation", "Retained client model", "Routing / transfer"],
        [
            ("Flat FedAvg", "Client -> global", "None between rounds", "Single model"),
            (
                "Simple HFL",
                "Client -> L2 -> L3",
                "Local outputs for evaluation",
                "Single-level evaluation",
            ),
            (
                "Ditto",
                "Global paths -> global",
                "Personalized weights",
                "Personal CE + proximal loss",
            ),
            (
                "TEMPO-FL",
                "L2 shadows -> L2 -> L3",
                "L1 specialist + cached shadows",
                "Confidence cascade + gated promotion",
            ),
        ],
    )
    console.print(
        "These are implemented structures, not evidence of general superiority. See README.md for source quirks."
    )
