from __future__ import annotations

def load_feature_backend(name: str, *, function_names: list[str] | None = None, namespace: dict | None = None):
    if name == "generated_rules":
        from .generated_rules import GeneratedRulesBackend

        if function_names is None or namespace is None:
            raise ValueError("generated_rules backend requires function_names and namespace")
        return GeneratedRulesBackend(function_names=function_names, namespace=namespace)
    if name == "bbb_martins":
        from .bbb_martins import BBBMartinsBackend

        return BBBMartinsBackend()
    if name == "dili":
        from .dili import DILIBackend

        return DILIBackend()
    if name == "clintox":
        from .clintox import ClinToxBackend

        return ClinToxBackend()
    if name == "pampa_ncats":
        from .pampa_ncats import PAMPANCATSBackend

        return PAMPANCATSBackend()
    if name == "skin_reaction":
        from .skin_reaction import SkinReactionBackend

        return SkinReactionBackend()
    if name == "pgp_broccatelli":
        from .pgp_broccatelli import PgpBroccatelliBackend

        return PgpBroccatelliBackend()
    if name == "carcinogens_lagunin":
        from .carcinogens_lagunin import CarcinogensLaguninBackend

        return CarcinogensLaguninBackend()
    if name == "ames":
        from .ames import AMESBackend

        return AMESBackend()
    raise ValueError(f"Unknown feature backend: {name}")
