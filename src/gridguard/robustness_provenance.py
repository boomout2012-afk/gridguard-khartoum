"""Non-circular release trust anchors for robustness evidence.

This module is intentionally outside :mod:`gridguard.robustness`: the source
bundle binds the exact bytes of that scientific implementation, while the
project source manifest independently binds this trust-anchor module.
"""

from typing import Final

ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS: Final[tuple[tuple[str, str, int], ...]] = (
    (
        "src/gridguard/controllers.py",
        "340d74a3f01a887caa6daa9d93aa36680edd8772400cc52c5b0df3390a0035a3",
        3_619,
    ),
    (
        "src/gridguard/domain.py",
        "a8818ed214423601d84018710bdccec984bd37153acdecf1c14ba17d22bddcba",
        10_978,
    ),
    (
        "src/gridguard/evaluation.py",
        "855b48c32c7d773326a2667cb016812ec285cdd82780965f59435cf63af46cd7",
        50_398,
    ),
    (
        "src/gridguard/metrics.py",
        "f1ac6af7209144efeddf06b8c14ce262d3e864d2409d534559ee65f78d8c02a8",
        6_384,
    ),
    (
        "src/gridguard/policy.py",
        "1415e6989c573500247a5323ef64f26e1173fe5eff3ea5205305b855fa128da2",
        19_073,
    ),
    (
        "src/gridguard/robustness.py",
        "f13d34b00a8880d2dbc72ce474b5857eecebad4f6e010280fb2052805c5c7394",
        64_305,
    ),
    (
        "src/gridguard/scenarios.py",
        "e07c7ee9b41ead44991fbf20e64a7c504883175bd3a6a4b6a51ac7ef0305db84",
        12_513,
    ),
    (
        "src/gridguard/simulation.py",
        "0853ed5b0e0b0f0b9667c6e184d52f4a8741c836cd79a45e43757b826c51cfa4",
        12_419,
    ),
    (
        "src/gridguard/statistics.py",
        "1473de613fa19d68dae4a0625333068d74edcd34b18b1ade4c370d33d92391a9",
        7_279,
    ),
)

# Accepted only after two byte-identical CUDA reproductions of the final source bundle.
ROBUSTNESS_SUPPLEMENT_SHA256: Final = (
    "c89e04b1bdca371e1728f6b4cb19f4e899261ab805c2e400c15e835eaf75a30b"
)
