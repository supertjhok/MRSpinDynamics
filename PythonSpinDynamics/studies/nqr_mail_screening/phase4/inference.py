"""Joint Gaussian feature likelihood, decision-risk utility and physical ledger."""

from dataclasses import dataclass  # noqa: E402
import numpy as np
from scipy.special import logsumexp, expit  # noqa: E402
from spin_dynamics.design.models import PredictiveModel  # noqa: E402
from spin_dynamics.design.likelihoods import GaussianLikelihood  # noqa: E402
from spin_dynamics.design.posterior import ParticlePosterior  # noqa: E402
from spin_dynamics.design.session import AdaptiveDesignSession  # noqa: E402
from spin_dynamics.design.spaces import CandidateDesignSpace  # noqa: E402
from spin_dynamics.design.utilities import UtilityEstimate  # noqa: E402
from features import real_features  # noqa: E402


@dataclass(frozen=True)
class DecisionRisk:
    samples: int = 48

    def estimate(self, model, posterior, design, rng):
        predictions = model.predict(posterior.parameters, design)
        indices = rng.choice(len(predictions), self.samples, p=posterior.weights)
        y = predictions[indices] + rng.normal(
            size=(self.samples, predictions.shape[-1])
        )
        lp = (
            -0.5 * np.sum((y[:, None, :] - predictions[None, :, :]) ** 2, axis=-1)
            + posterior.log_weights
        )
        weights = np.exp(lp - logsumexp(lp, axis=1)[:, None])
        positive = posterior.parameters["positive"].astype(bool)
        p0 = float(posterior.weights[positive].sum())
        p = weights[:, positive].sum(axis=1)
        reduction = min(p0, 1 - p0) - np.minimum(p, 1 - p)
        return UtilityEstimate(
            float(reduction.mean()),
            float(reduction.std(ddof=1) / np.sqrt(self.samples)),
            reduction,
            "classification risk",
        )


class Cost:
    def __init__(self, nodes):
        self.nodes = nodes

    def seconds(self, node):
        return self.nodes[node]["cost_s"]


class Detector:
    def __init__(
        self,
        info,
        arrays,
        mean,
        scale,
        covariance,
        gain=1.0,
        prior=0.5,
        omit_temperature=None,
    ):
        self.nodes = info["nodes"]
        self.sunk = info["sunk_and_exit_cost_s"]
        self.mean, self.scale, self.covariance = mean, scale, covariance
        self.true = arrays["means_v"]
        positive = [0]
        templates = [np.zeros_like(mean)]
        for ti in range(3):
            if ti == omit_temperature:
                continue
            for mass in (0.1, 1.0):
                for phase in (0, np.pi / 2, np.pi, 3 * np.pi / 2):
                    templates.append(
                        real_features(self.true[ti] * mass * gain * np.exp(1j * phase))
                        / scale
                    )
                    positive.append(1)
        self.templates = np.array(templates)
        self.positive = np.array(positive)
        self.prior = prior
        weights = np.r_[
            1 - prior, np.full(len(positive) - 1, prior / (len(positive) - 1))
        ]
        self.initial = ParticlePosterior(
            {"positive": self.positive, "index": np.arange(len(positive))},
            np.log(weights),
        )
        self.lookup = {tuple(node["path"]): node["id"] for node in self.nodes}
        self.conditionals = {}

    def conditional(self, past, node):
        key = (tuple(past), node)
        if key not in self.conditionals:
            ix = np.arange(4 * node, 4 * node + 4)
            previous = (
                np.concatenate([np.arange(4 * p, 4 * p + 4) for p in past])
                if past
                else np.array([], int)
            )
            block = self.covariance[np.ix_(ix, ix)]
            if len(previous):
                cross = self.covariance[np.ix_(ix, previous)]
                b = np.linalg.solve(
                    self.covariance[np.ix_(previous, previous)], cross.T
                ).T
                block = block - b @ cross.T
            else:
                b = np.zeros((4, 0))
            whitening = np.linalg.inv(np.linalg.cholesky(block))
            self.conditionals[key] = (b, whitening)
        return self.conditionals[key]

    def run(self, features, cap, adaptive=True, seed=0, allow_stopping=True):
        standardized = (features - self.mean) / self.scale
        posterior = self.initial
        path = ()
        past = []
        observations = []
        spent = 0.0
        stopped = False
        utilities = []
        for step in range(3):
            children = [
                self.lookup[path + (a,)] for a in range(3) if path + (a,) in self.lookup
            ]
            children = [
                n
                for n in children
                if self.sunk + spent + self.nodes[n]["cost_s"] <= cap + 1e-12
            ]
            if not children:
                break

            def predictor(parameters, node):
                b, w = self.conditional(past, node)
                mu = self.templates[:, node]
                if past:
                    residual = np.concatenate(observations) - self.templates[
                        :, past
                    ].reshape(len(self.templates), -1)
                    mu = mu + residual @ b.T
                return mu @ w.T

            model = PredictiveModel(predictor, GaussianLikelihood(1.0, event_ndim=1))
            session = AdaptiveDesignSession(
                model=model,
                posterior=posterior,
                design_space=CandidateDesignSpace(children),
                utility=DecisionRisk(),
                cost=Cost(self.nodes),
                seed=seed + step,
            )
            if adaptive:
                recommendation = session.ask()
                chosen = recommendation.best.design
                utilities.append(
                    [
                        recommendation.best.utility.value,
                        recommendation.best.utility.standard_error,
                    ]
                )
            else:
                desired = self.lookup[path + ((0, 1, 2)[step],)]
                if desired not in children:
                    break
                chosen = desired
            _, w = self.conditional(past, chosen)
            observation = standardized[chosen]
            session.tell(chosen, w @ observation)
            posterior = session.posterior
            spent += self.nodes[chosen]["cost_s"]
            observations.append(observation)
            past.append(chosen)
            path = tuple(self.nodes[chosen]["path"])
            log_positive = logsumexp(posterior.log_weights[self.positive == 1])
            log_negative = posterior.log_weights[0]
            log_odds = float(log_positive - log_negative)
            bf = log_odds - np.log(self.prior / (1 - self.prior))
            risk = float(expit(-abs(log_odds)))
            if adaptive and allow_stopping and risk <= 0.05 and abs(bf) >= np.log(3):
                stopped = True
                break
        if not past:
            log_odds = float(np.log(self.prior / (1 - self.prior)))
        bf = log_odds - np.log(self.prior / (1 - self.prior))
        return {
            "score": float(bf),
            "seconds": self.sunk + spent,
            "path": list(path),
            "stopped_on_evidence": stopped,
            "posterior_positive": float(expit(log_odds)),
            "utility_estimates": utilities,
        }

    def signal(self, temperature, phase, gain=1.0, mass=1.0, missing_y=False):
        result = real_features(
            self.true[temperature] * gain * mass * np.exp(1j * phase)
        )
        if missing_y:
            for node in self.nodes:
                if node["action"] == 1:
                    result[node["id"]] = 0
        return result
