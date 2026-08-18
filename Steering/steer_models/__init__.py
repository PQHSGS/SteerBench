"""
Steering model implementations.

This module contains all concrete implementations of steering models,
organized by type for better maintainability.
"""

# Dense steering models
from .dense import (
    DenseSteerModel,
    RiemannianSteerModel,
    SAEFreeSteerModel,
    ConditionalSteerModel,
    ManifoldSteerModel,
    SphericalSteerModel,
)
from .nonlinear import (
    AngularSteerModel,
    CurveballSteerModel,
    FlowSteerModel,
    PIDSteerModel,
    FLASSteerModel,
    LoReFTSteerModel,
    ODESteerModel,
    BIPOSteerModel,
    CobraSteerModel,
    INNSteerModel,
    LQRSteerModel,
    IDSSteerModel,
    FishBackSteerModel,
    GINNSteerModel,
)

from .transport import (
    ActivationTransportSteerModel,
    CHARSSteerModel,
    LinNEASSteerModel,
)
from .weight import WeightSteerModel

# SAE-based steering models
from .sae import (
    SASSteerModel,
    SRESteerModel,
    SSVSteerModel,
    SPARESteerModel,
    SRPSSteerModel,
    SAERSVSteerModel,
    SAETSSteerModel,
    SAEIOSteerModel,
    SAECoTSteerModel,
    FeatSteerModel,
    CorrSteerModel,  # SAE-based (uses SAE for extraction, simple addition for steering)
)


# =============================================================================
# STEER MODEL REGISTRY
# To add a new method: 1) Add class above  2) Add to this dict
# =============================================================================

STEER_MAP = {
    "CAA": DenseSteerModel,
    "COLD": DenseSteerModel,
    "CAST": ConditionalSteerModel,
    "WEIGHTSTEER": WeightSteerModel,
    "SAS": SASSteerModel,
    "SPARE": SPARESteerModel,
    "SRE": SRESteerModel,
    "SRPS": SRPSSteerModel,
    "SSV": SSVSteerModel,
    "ANGULAR": AngularSteerModel,
    "SPHERICAL": SphericalSteerModel,
    "FGAA": DenseSteerModel,
    "MANIFOLD": ManifoldSteerModel,
    "CORRSTEER": CorrSteerModel,
    "SAE-RSV": SAERSVSteerModel,
    "SAE-TS": SAETSSteerModel,
    "SAEIO": SAEIOSteerModel,
    "SAE-FREE": SAEFreeSteerModel,
    "SAE-COT": SAECoTSteerModel,
    "FEAT": FeatSteerModel,
    "CAST_HF": ConditionalSteerModel,  # HF extraction, same steering as CAST
    "ACT": ActivationTransportSteerModel,
    "CURVEBALL": CurveballSteerModel,
    "FLOW": FlowSteerModel,
    "PID": PIDSteerModel,
    "REFT": LoReFTSteerModel,
    "LOREFT": LoReFTSteerModel,
    "REPS": LoReFTSteerModel,
    "ODE": ODESteerModel,
    "FLAS": FLASSteerModel,
    "BIPO": BIPOSteerModel,
    "CHARS": CHARSSteerModel,
    "COBRA": CobraSteerModel,
    "LINNEAS": LinNEASSteerModel,
    "INNSTEER": INNSteerModel,
    "LQR": LQRSteerModel,
    "JSPACE": DenseSteerModel,
    "IDS": IDSSteerModel,
    "FISHBACK": FishBackSteerModel,
    "GINN": GINNSteerModel,
    "RIEMANNIAN": RiemannianSteerModel,
}


__all__ = [
    # Dense
    "DenseSteerModel",
    "RiemannianSteerModel",
    "SAEFreeSteerModel",
    "SphericalSteerModel",
    # SAE-based
    "SRESteerModel",
    "SSVSteerModel",
    "SPARESteerModel",
    "SRPSSteerModel",
    "SAERSVSteerModel",
    "SAETSSteerModel",
    "SAEIOSteerModel",
    "SAECoTSteerModel",
    "FeatSteerModel",
    # Special
    "ConditionalSteerModel",
    "AngularSteerModel",
    "ManifoldSteerModel",
    "CorrSteerModel",
    "ActivationTransportSteerModel",
    "CurveballSteerModel",
    "FlowSteerModel",
    "PIDSteerModel",
    "LoReFTSteerModel",
    "ODESteerModel",
    "FLASSteerModel",
    "BIPOSteerModel",
    "CHARSSteerModel",
    "CobraSteerModel",
    "LinNEASSteerModel",
    "INNSteerModel",
    "LQRSteerModel",
    "WeightSteerModel",
    "IDSSteerModel",
    "FishBackSteerModel",
    
    # Registry
    "STEER_MAP",
]
