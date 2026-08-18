"""
Extractor implementations.

This module contains all concrete implementations of steering vector extractors,
organized by type for better maintainability.
"""

# Dense extractors
from .dense import (
    CAAExtractor,
    COLDExtractor,
    CASTExtractor,
    ManifoldExtractor,
    SphericalExtractor,
    RiemannianExtractor,
)
from .hf_model import HFCASTExtractor
from .nonlinear import (
    AngularExtractor,
    CurveballExtractor,
    FLASExtractor,
    FlowExtractor,
    PIDExtractor,
    ODEExtractor,
    LoReFTExtractor,
    BIPOExtractor,
    CobraExtractor,
    INNExtractor,
    LQRExtractor,
    JSpaceExtractor,
    IDSExtractor,
    FishBackExtractor,
    GINNExtractor,
)
from .transport import (
    ActivationTransportExtractor,
    CHARSExtractor,
    LinNEASExtractor,
)
from .weight import WeightSteerExtractor

# SAE-based extractors
from .sae import (
    SASExtractor,
    SPAREExtractor,
    SREExtractor,
    SRPSExtractor,
    SSVExtractor,
    SAEIOExtractor,
    SAERSVExtractor,
    SAETSSExtractor,
    SAEFreeExtractor,
    SAECoTExtractor,
    CorrSteerExtractor,
    FGAAExtractor,
)


# =============================================================================
# EXTRACTOR REGISTRY
# To add a new method: 1) Add class above  2) Add to this dict
# =============================================================================

EXTRACTOR_MAP = {
    "CAA": CAAExtractor,
    "COLD": COLDExtractor,
    "CAST": CASTExtractor,  # TL-based, ~0.39 cosine with CAST lib
    "CAST_HF": HFCASTExtractor,  # HF-based, exact match with CAST lib
    "WEIGHTSTEER": WeightSteerExtractor,
    "SAS": SASExtractor,
    "SPARE": SPAREExtractor,
    "SRE": SREExtractor,
    "SRPS": SRPSExtractor,
    "SSV": SSVExtractor,
    "SAEIO": SAEIOExtractor,
    "ANGULAR": AngularExtractor,
    "MANIFOLD": ManifoldExtractor,
    "SPHERICAL": SphericalExtractor,
    "FGAA": FGAAExtractor,
    "CORRSTEER": CorrSteerExtractor,
    "SAE-RSV": SAERSVExtractor,
    "SAE-TS": SAETSSExtractor,
    "SAE-FREE": SAEFreeExtractor,
    "SAE-COT": SAECoTExtractor,
    "ACT": ActivationTransportExtractor,
    "CURVEBALL": CurveballExtractor,
    "FLOW": FlowExtractor,
    "PID": PIDExtractor,
    "REFT": LoReFTExtractor,
    "LOREFT": LoReFTExtractor,
    "REPS": LoReFTExtractor,
    "ODE": ODEExtractor,
    "FLAS": FLASExtractor,
    "BIPO": BIPOExtractor,
    "CHARS": CHARSExtractor,
    "COBRA": CobraExtractor,
    "LINNEAS": LinNEASExtractor,
    "INNSTEER": INNExtractor,
    "LQR": LQRExtractor,
    "JSPACE": JSpaceExtractor,
    "IDS": IDSExtractor,
    "FISHBACK": FishBackExtractor,
    "GINN": GINNExtractor,
    "RIEMANNIAN": RiemannianExtractor,
}


__all__ = [
    # Extractors
    "CAAExtractor",
    "COLDExtractor",
    "CASTExtractor",
    "HFCASTExtractor",
    "SASExtractor",
    "SPAREExtractor",
    "SREExtractor",
    "SRPSExtractor",
    "SSVExtractor",
    "SAEIOExtractor",
    "AngularExtractor",
    "ManifoldExtractor",
    "SphericalExtractor",
    "FGAAExtractor",
    "CorrSteerExtractor",
    "SAERSVExtractor",
    "SAETSSExtractor",
    "SAEFreeExtractor",
    "SAECoTExtractor",
    "ActivationTransportExtractor",
    "CurveballExtractor",
    "FlowExtractor",
    "PIDExtractor",
    "LoReFTExtractor",
    "ODEExtractor",
    "FLASExtractor",
    "BIPOExtractor",
    "CHARSExtractor",
    "CobraExtractor",
    "LinNEASExtractor",
    "INNExtractor",
    "LQRExtractor",
    "JSpaceExtractor",
    "WeightSteerExtractor",
    "IDSExtractor",
    "FishBackExtractor",
    "GINNExtractor",
    "RiemannianExtractor",
    # Registry
    "EXTRACTOR_MAP",
]
