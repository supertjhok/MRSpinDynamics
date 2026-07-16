"""Bayesian experiment-design tools.

The generic Phase 1 core provides NumPy grid/particle inference, likelihood-
backed predictive models, finite candidate spaces, information and QoI-risk
utilities, physical costs, constraints, stopping, and replayable sessions.
Phase 0 remains available as an exact-grid inversion-recovery reference.
"""

from spin_dynamics.design.constraints import (
    CallableConstraint,
    ConstraintResult,
    DesignConstraint,
    evaluate_constraints,
)
from spin_dynamics.design.costs import CallableCost, ConstantCost
from spin_dynamics.design.diagnostics import (
    CredibleIntervalStopping,
    PosteriorStandardDeviationStopping,
    PosteriorSummary,
    QuantityOfInterest,
    quantity_values,
    summarize_quantity,
    weighted_mean,
    weighted_quantile,
    weighted_variance,
)
from spin_dynamics.design.io import (
    load_design_state,
    save_design_state,
    state_from_json,
    state_to_json,
)
from spin_dynamics.design.likelihoods import (
    ComplexGaussianLikelihood,
    GaussianLikelihood,
)
from spin_dynamics.design.models import PredictiveModel
from spin_dynamics.design.posterior import (
    GridPosterior,
    ParticlePosterior,
    posterior_from_state,
)
from spin_dynamics.design.priors import (
    DiscretePrior,
    IndependentPrior,
    LogUniformPrior,
    NormalPrior,
    UniformPrior,
)

from spin_dynamics.design.reference_ir import (
    IRAcquisitionCost,
    IRAdaptiveSession,
    IRBenchmarkResult,
    IRDesign,
    IRDesignScore,
    IRGridPosterior,
    IRGridPrior,
    IRMeasurement,
    IRRecommendation,
    IRTruth,
    inversion_recovery_signal,
    recommend_ir_design,
    run_ir_reference_trial,
    simulate_ir_observation,
)
from spin_dynamics.design.session import (
    AdaptiveDesignSession,
    CandidateScore,
    DesignObservation,
    DesignRecommendation,
)
from spin_dynamics.design.spaces import CandidateDesignSpace
from spin_dynamics.design.types import (
    ObservationLikelihood,
    ParameterParticles,
    PhysicalCost,
    Prior,
    StopRule,
)
from spin_dynamics.design.utilities import (
    DesignUtility,
    ExpectedInformationGain,
    ExpectedVarianceReduction,
    UtilityEstimate,
)

__all__ = [
    "AdaptiveDesignSession",
    "CallableConstraint",
    "CallableCost",
    "CandidateDesignSpace",
    "CandidateScore",
    "ComplexGaussianLikelihood",
    "ConstantCost",
    "ConstraintResult",
    "CredibleIntervalStopping",
    "DesignConstraint",
    "DesignObservation",
    "DesignRecommendation",
    "DesignUtility",
    "DiscretePrior",
    "ExpectedInformationGain",
    "ExpectedVarianceReduction",
    "GaussianLikelihood",
    "GridPosterior",
    "IRAcquisitionCost",
    "IRAdaptiveSession",
    "IRBenchmarkResult",
    "IRDesign",
    "IRDesignScore",
    "IRGridPosterior",
    "IRGridPrior",
    "IRMeasurement",
    "IRRecommendation",
    "IRTruth",
    "IndependentPrior",
    "LogUniformPrior",
    "NormalPrior",
    "ObservationLikelihood",
    "ParameterParticles",
    "ParticlePosterior",
    "PhysicalCost",
    "PosteriorStandardDeviationStopping",
    "PosteriorSummary",
    "PredictiveModel",
    "Prior",
    "QuantityOfInterest",
    "StopRule",
    "UniformPrior",
    "UtilityEstimate",
    "evaluate_constraints",
    "inversion_recovery_signal",
    "load_design_state",
    "posterior_from_state",
    "quantity_values",
    "recommend_ir_design",
    "run_ir_reference_trial",
    "save_design_state",
    "simulate_ir_observation",
    "state_from_json",
    "state_to_json",
    "summarize_quantity",
    "weighted_mean",
    "weighted_quantile",
    "weighted_variance",
]
