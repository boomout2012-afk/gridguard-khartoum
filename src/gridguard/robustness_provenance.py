"""Non-circular release trust anchors for robustness evidence.

This module is intentionally outside :mod:`gridguard.robustness`: the source
bundle binds the exact bytes of that scientific implementation, while the
project source manifest independently binds this trust-anchor module.
"""

from typing import Final

ROBUSTNESS_LEGACY_IMPLEMENTATION_SOURCE_BINDINGS: Final[tuple[tuple[str, str, int], ...]] = (
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

ROBUSTNESS_IMPLEMENTATION_SOURCE_BINDINGS: Final[tuple[tuple[str, str, int], ...]] = (
    (
        "src/gridguard/controllers.py",
        "340d74a3f01a887caa6daa9d93aa36680edd8772400cc52c5b0df3390a0035a3",
        3_619,
    ),
    (
        "src/gridguard/claim_contract.py",
        "cafc5848930d3115f5909ef25c231cc800385b1b3fdd1f7aff04a10c58327de9",
        1_332,
    ),
    (
        "src/gridguard/domain.py",
        "bd16fb46fd04a8865a6db4498a56f681c8eb77fe3d534497b53a8db4b5c31d90",
        26_417,
    ),
    (
        "src/gridguard/evaluation.py",
        "a9addfee69647f9111122c2c1b29fbe0d1523a2be9c0578427ef0d2b3eeb36b1",
        70_616,
    ),
    (
        "src/gridguard/forecast.py",
        "9d11f80ed6246466da84203138c48439c2a44cbd707dc53c458d307bb16deeea",
        2_829,
    ),
    (
        "src/gridguard/metrics.py",
        "4a0eb676b3f7ffb0759ff107141cd95653faa653115bec7654422c1e45f9534a",
        6_462,
    ),
    (
        "src/gridguard/paired_inference.py",
        "59ee10d34e780f44f91d6cb30abaf4cc1e8862c4718cc5738cd90dd08cc4e9de",
        8_159,
    ),
    (
        "src/gridguard/policy.py",
        "22f17e11822c89104fab9b145ee5749c7ee8ea45f0eebc1a9ee314efc084a6f9",
        19_073,
    ),
    (
        "src/gridguard/robustness.py",
        "68c5f59d81a1cec877bc65b59130bcb0b3aef03ff6d3a8c36c097e483943f309",
        96_166,
    ),
    (
        "src/gridguard/robustness_inference.py",
        "814fa97fd62e08f83fb149fa2afb83382fe3efb9362f178f2d4f1ec536661798",
        37_813,
    ),
    (
        "src/gridguard/scenarios.py",
        "2b875642c7c89af6881fcc6795d5126bf2d439b4bcacd4335daf9ca8d5d34990",
        14_765,
    ),
    (
        "src/gridguard/simulation.py",
        "84e8d902e7c2c728a9c431b47cee38f04193cfdca1ca0054eb188c1c7b2c4199",
        13_461,
    ),
    (
        "src/gridguard/statistics.py",
        "1473de613fa19d68dae4a0625333068d74edcd34b18b1ade4c370d33d92391a9",
        7_279,
    ),
)

# Accepted only after two byte-identical CUDA reproductions of the final source bundle.
ROBUSTNESS_SUPPLEMENT_SHA256: Final = (
    "6b56d149331ee191302cd9e6f22365988db0ae301f9441190b61214acf832e48"
)
