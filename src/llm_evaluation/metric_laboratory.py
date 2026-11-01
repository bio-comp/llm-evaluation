"""Known-truth metric, calibration, and reference-error laboratory.

Implements the bounded scientific block specified in EXPERIMENTS.md (lines 194-219)
and Research/METRIC_VALIDITY_AND_MEASUREMENT.md.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Literal, Sequence
from pydantic import BaseModel, ConfigDict, Field

KNOWN_P_VALUES: tuple[float, ...] = (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99)


# ==============================================================================
# 1. Scoring Rules & Binary Proper Losses
# ==============================================================================


def brier_score(forecasts: Sequence[float], outcomes: Sequence[int | float]) -> float:
    """Compute empirical Brier score: mean squared error between forecast and outcome."""
    if len(forecasts) != len(outcomes) or not forecasts:
        raise ValueError("forecasts and outcomes must have equal non-zero length")
    return sum((q - y) ** 2 for q, y in zip(forecasts, outcomes, strict=True)) / len(
        forecasts
    )


def expected_brier(p: float, q: float) -> float:
    """Exact analytic expected Brier loss E_p[(q - Y)^2] = (q - p)^2 + p*(1 - p)."""
    if not (0.0 <= p <= 1.0 and 0.0 <= q <= 1.0):
        raise ValueError("p and q must be in [0, 1]")
    return (q - p) ** 2 + p * (1.0 - p)


def log_score(
    forecasts: Sequence[float],
    outcomes: Sequence[int | float],
    eps: float = 1e-12,
) -> float:
    """Compute empirical binary log score (cross-entropy) with clipping guard."""
    if len(forecasts) != len(outcomes) or not forecasts:
        raise ValueError("forecasts and outcomes must have equal non-zero length")
    total = 0.0
    for q, y in zip(forecasts, outcomes, strict=True):
        q_clip = min(max(q, eps), 1.0 - eps)
        total += -(y * math.log(q_clip) + (1.0 - y) * math.log(1.0 - q_clip))
    return total / len(forecasts)


def expected_log_score(p: float, q: float, eps: float = 1e-12) -> float:
    """Exact analytic expected log loss E_p[- Y ln(q) - (1-Y) ln(1-q)] with clipping."""
    if not (0.0 <= p <= 1.0 and 0.0 <= q <= 1.0):
        raise ValueError("p and q must be in [0, 1]")
    q_clip = min(max(q, eps), 1.0 - eps)
    return -(p * math.log(q_clip) + (1.0 - p) * math.log(1.0 - q_clip))


# ==============================================================================
# 2. Calibration Estimators (Fixed-bin, Equal-mass, Debiased, Population)
# ==============================================================================


@dataclass(frozen=True)
class BinDetail:
    bin_index: int
    lower: float
    upper: float
    count: int
    mean_conf: float
    mean_acc: float
    gap: float


def fixed_bin_ece(
    forecasts: Sequence[float],
    outcomes: Sequence[int | float],
    num_bins: int = 10,
) -> tuple[float, list[BinDetail]]:
    """Compute fixed-width Expected Calibration Error (ECE) over [0, 1]."""
    if len(forecasts) != len(outcomes) or not forecasts:
        raise ValueError("forecasts and outcomes must have equal non-zero length")
    if num_bins < 1:
        raise ValueError("num_bins must be positive")

    n = len(forecasts)
    bin_confs: list[list[float]] = [[] for _ in range(num_bins)]
    bin_accs: list[list[float]] = [[] for _ in range(num_bins)]

    for q, y in zip(forecasts, outcomes, strict=True):
        idx = int(q * num_bins)
        if idx >= num_bins:
            idx = num_bins - 1
        bin_confs[idx].append(float(q))
        bin_accs[idx].append(float(y))

    ece = 0.0
    details: list[BinDetail] = []
    for b in range(num_bins):
        cnt = len(bin_confs[b])
        low = b / num_bins
        high = (b + 1) / num_bins
        if cnt > 0:
            m_conf = sum(bin_confs[b]) / cnt
            m_acc = sum(bin_accs[b]) / cnt
            gap = abs(m_acc - m_conf)
            ece += (cnt / n) * gap
        else:
            m_conf = 0.0
            m_acc = 0.0
            gap = 0.0
        details.append(BinDetail(b, low, high, cnt, m_conf, m_acc, gap))

    return ece, details


def equal_mass_ece(
    forecasts: Sequence[float],
    outcomes: Sequence[int | float],
    num_bins: int = 10,
) -> tuple[float, list[BinDetail]]:
    """Compute adaptive quantile equal-mass Expected Calibration Error (ECE)."""
    if len(forecasts) != len(outcomes) or not forecasts:
        raise ValueError("forecasts and outcomes must have equal non-zero length")
    if num_bins < 1:
        raise ValueError("num_bins must be positive")

    n = len(forecasts)
    paired = sorted(zip(forecasts, outcomes, strict=True), key=lambda x: x[0])
    details: list[BinDetail] = []
    ece = 0.0

    chunk_size = n // num_bins
    remainder = n % num_bins

    start = 0
    for b in range(num_bins):
        size = chunk_size + (1 if b < remainder else 0)
        end = start + size
        bin_items = paired[start:end]
        if bin_items:
            m_conf = sum(x[0] for x in bin_items) / size
            m_acc = sum(x[1] for x in bin_items) / size
            gap = abs(m_acc - m_conf)
            low = float(bin_items[0][0])
            high = float(bin_items[-1][0])
            ece += (size / n) * gap
        else:
            m_conf = 0.0
            m_acc = 0.0
            gap = 0.0
            low = 0.0
            high = 0.0
        details.append(BinDetail(b, low, high, size, m_conf, m_acc, gap))
        start = end

    return ece, details


def debiased_ece(
    forecasts: Sequence[float],
    outcomes: Sequence[int | float],
    num_bins: int = 10,
) -> float:
    """Compute finite-sample debiased calibration error adjustment."""
    _, details = fixed_bin_ece(forecasts, outcomes, num_bins=num_bins)
    n = len(forecasts)
    total = 0.0
    for b in details:
        if b.count > 1:
            raw_sq = (b.mean_acc - b.mean_conf) ** 2
            penalty = (b.mean_acc * (1.0 - b.mean_acc)) / (b.count - 1)
            unbiased_sq = max(0.0, raw_sq - penalty)
            total += (b.count / n) * math.sqrt(unbiased_sq)
    return total


def population_ece(
    probs: Sequence[float],
    forecasts: Sequence[float],
    weights: Sequence[float] | None = None,
) -> float:
    """Analytic population ECE: weighted expectation of |p - q|."""
    if len(probs) != len(forecasts) or not probs:
        raise ValueError("probs and forecasts must have equal non-zero length")
    n = len(probs)
    if weights is None:
        w = [1.0 / n] * n
    else:
        if len(weights) != n:
            raise ValueError("weights length must match probs length")
        total_w = sum(weights)
        w = [x / total_w for x in weights]
    return sum(wi * abs(p - q) for p, q, wi in zip(probs, forecasts, w, strict=True))


# ==============================================================================
# 3. Selective Classification & AURC (LE-103 Gate)
# ==============================================================================


def mean_prefix_risk(
    confidences: Sequence[float],
    errors: Sequence[int | float],
) -> float:
    """Direct mean prefix risk: average error rate across prefixes sorted by decreasing confidence.

    Items are sorted strictly by decreasing confidence. If multiple items share
    identical confidence, input order is preserved.
    """
    if len(confidences) != len(errors) or not confidences:
        raise ValueError("confidences and errors must have equal non-zero length")
    paired = sorted(
        zip(confidences, errors, strict=True), key=lambda x: x[0], reverse=True
    )
    n = len(paired)
    cumulative_errors = 0.0
    prefix_risks_sum = 0.0
    for k, (_, err) in enumerate(paired, start=1):
        cumulative_errors += float(err)
        prefix_risks_sum += cumulative_errors / k
    return prefix_risks_sum / n


def empirical_aurc_harmonic(
    confidences: Sequence[float],
    errors: Sequence[int | float],
) -> float:
    """Empirical harmonic-weight AURC formulation.

    For n sorted items, item i receives weight w_i = (H_n - H_{i-1}) / n,
    where H_m = sum_{j=1}^m 1/j is the m-th harmonic number.
    Mathematically equal to mean_prefix_risk for strictly distinct confidences.
    """
    if len(confidences) != len(errors) or not confidences:
        raise ValueError("confidences and errors must have equal non-zero length")
    paired = sorted(
        zip(confidences, errors, strict=True), key=lambda x: x[0], reverse=True
    )
    n = len(paired)

    # Compute harmonic numbers H_0 ... H_n
    h = [0.0] * (n + 1)
    for j in range(1, n + 1):
        h[j] = h[j - 1] + 1.0 / j

    aurc = 0.0
    for i, (_, err) in enumerate(paired, start=1):
        weight = (h[n] - h[i - 1]) / n
        aurc += weight * float(err)
    return aurc


def compute_aurc(
    confidences: Sequence[float],
    errors: Sequence[int | float],
    tie_convention: Literal["average", "optimistic", "pessimistic"] = "average",
) -> float:
    """Compute Area Under Risk-Coverage curve with explicit tie convention.

    Per LE-103:
    - 'optimistic': within tied blocks, sort errors last (minimizing risk).
    - 'pessimistic': within tied blocks, sort errors first (maximizing risk).
    - 'average': exact analytical expectation over uniform permutations of tied blocks.
    """
    if len(confidences) != len(errors) or not confidences:
        raise ValueError("confidences and errors must have equal non-zero length")

    # Group by confidence
    block_map: dict[float, list[float]] = {}
    for c, err in zip(confidences, errors, strict=True):
        block_map.setdefault(float(c), []).append(float(err))

    # Sort distinct confidence values descending
    distinct_confs = sorted(block_map.keys(), reverse=True)
    n = len(confidences)

    if tie_convention == "optimistic":
        # Within each block, sort 0 before 1
        sorted_errors: list[float] = []
        for c in distinct_confs:
            sorted_errors.extend(sorted(block_map[c]))
        cumulative = 0.0
        risk_sum = 0.0
        for k, err in enumerate(sorted_errors, start=1):
            cumulative += err
            risk_sum += cumulative / k
        return risk_sum / n

    elif tie_convention == "pessimistic":
        # Within each block, sort 1 before 0
        sorted_errors = []
        for c in distinct_confs:
            sorted_errors.extend(sorted(block_map[c], reverse=True))
        cumulative = 0.0
        risk_sum = 0.0
        for k, err in enumerate(sorted_errors, start=1):
            cumulative += err
            risk_sum += cumulative / k
        return risk_sum / n

    elif tie_convention == "average":
        # Exact expected prefix risk across all uniform random permutations
        k_prev = 0
        e_prev = 0.0
        expected_risk_sum = 0.0
        for c in distinct_confs:
            b_errors = block_map[c]
            b_size = len(b_errors)
            e_block = sum(b_errors)
            # For each position j = 1..b_size in the block:
            # Expected errors at step (k_prev + j) = e_prev + j * (e_block / b_size)
            for j in range(1, b_size + 1):
                step = k_prev + j
                expected_prefix_errors = e_prev + j * (e_block / b_size)
                expected_risk_sum += expected_prefix_errors / step
            k_prev += b_size
            e_prev += e_block
        return expected_risk_sum / n

    else:
        raise ValueError(f"unknown tie_convention: {tie_convention}")


# ==============================================================================
# 4. Reference Noise & Error Models
# ==============================================================================


def noisy_target_probability(clean_p: float, eta: float) -> float:
    """Exact target probability under symmetric independent binary flip noise eta:

    P(Y_obs = 1) = eta + (1 - 2*eta)*p.
    """
    if not (0.0 <= clean_p <= 1.0):
        raise ValueError("clean_p must be in [0, 1]")
    if not (0.0 <= eta <= 0.5):
        raise ValueError("eta must be in [0, 0.5]")
    return eta + (1.0 - 2.0 * eta) * clean_p


def noisy_expected_brier(clean_p: float, q: float, eta: float) -> float:
    """Expected Brier score evaluated against noisy reference labels."""
    noisy_p = noisy_target_probability(clean_p, eta)
    return expected_brier(noisy_p, q)


def check_reference_ranking_reversal(
    clean_p: float,
    q_clean: float,
    q_noisy: float,
    eta: float,
) -> dict[str, Any]:
    """Check whether a forecast superior on clean truth reverses on noisy reference.

    Implements Microprobe 4.
    """
    clean_b_qclean = expected_brier(clean_p, q_clean)
    clean_b_qnoisy = expected_brier(clean_p, q_noisy)

    noisy_b_qclean = noisy_expected_brier(clean_p, q_clean, eta)
    noisy_b_qnoisy = noisy_expected_brier(clean_p, q_noisy, eta)

    # Reversal occurs when clean_b_qclean < clean_b_qnoisy but noisy_b_qnoisy < noisy_b_qclean
    reversal = (clean_b_qclean < clean_b_qnoisy) and (noisy_b_qnoisy < noisy_b_qclean)
    return {
        "reversal_detected": reversal,
        "clean_brier_q_clean": clean_b_qclean,
        "clean_brier_q_noisy": clean_b_qnoisy,
        "noisy_brier_q_clean": noisy_b_qclean,
        "noisy_brier_q_noisy": noisy_b_qnoisy,
    }


# ==============================================================================
# 5. Pseudoreplication & Cluster Standard Error
# ==============================================================================


class ClusterSEResult(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)
    mean: float
    item_cluster_se: float
    naive_row_se: float
    se_ratio: float
    item_count: int
    row_count: int


def compute_cluster_vs_naive_se(
    item_outcomes: Sequence[Sequence[int | float]],
) -> ClusterSEResult:
    """Compare cluster-level inference (treating items as independent units)

    against naive row-level inference (treating repeated observations as independent).
    Implements Microprobe 5.
    """
    n_items = len(item_outcomes)
    if n_items < 2:
        raise ValueError("require at least 2 items")

    item_means: list[float] = []
    all_rows: list[float] = []
    for outcomes in item_outcomes:
        if not outcomes:
            raise ValueError("item must have at least one outcome")
        m = sum(outcomes) / len(outcomes)
        item_means.append(float(m))
        all_rows.extend(float(x) for x in outcomes)

    grand_mean = sum(all_rows) / len(all_rows)
    n_rows = len(all_rows)

    # Item cluster SE: variance of item means divided by n_items
    mean_of_item_means = sum(item_means) / n_items
    var_items = sum((m - mean_of_item_means) ** 2 for m in item_means) / (n_items - 1)
    item_se = math.sqrt(var_items / n_items)

    # Naive row SE: sample variance of all rows divided by n_rows
    var_rows = sum((y - grand_mean) ** 2 for y in all_rows) / (n_rows - 1)
    naive_se = math.sqrt(var_rows / n_rows)

    ratio = item_se / naive_se if naive_se > 0 else 1.0

    return ClusterSEResult(
        mean=grand_mean,
        item_cluster_se=item_se,
        naive_row_se=naive_se,
        se_ratio=ratio,
        item_count=n_items,
        row_count=n_rows,
    )


# ==============================================================================
# 6. Candidate Simulation Grid & Replicates
# ==============================================================================


def compute_forecast(p: float, kind: str) -> float:
    """Compute forecast q from clean p under the 4 declared forecast models."""
    if kind == "oracle":
        return p
    elif kind == "constant":
        return 0.5
    elif kind == "overconfident":
        # clip(0.5 + 1.5*(p - 0.5), 0.001, 0.999)
        val = 0.5 + 1.5 * (p - 0.5)
        return min(max(val, 0.001), 0.999)
    elif kind == "underconfident":
        # 0.2 + 0.6 * p
        return 0.2 + 0.6 * p
    else:
        raise ValueError(f"unknown forecast kind: {kind}")


class SimulationCellConfig(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)
    p_values: tuple[float, ...] = KNOWN_P_VALUES
    n_base_items: int = Field(default=50, ge=1)
    k_within_item: int = Field(default=1, ge=1)
    shared_within_item: bool = False
    reference_flip_eta: float = Field(default=0.0, ge=0.0, le=0.5)
    forecast_kind: Literal["oracle", "constant", "overconfident", "underconfident"] = (
        "oracle"
    )


class ReplicateEvaluation(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)
    brier_score: float
    log_score: float
    fixed_bin_ece: float
    equal_mass_ece: float
    aurc: float
    item_cluster_se: float
    naive_row_se: float


class CellSummary(BaseModel):
    model_config = ConfigDict(strict=True, frozen=True)
    config: SimulationCellConfig
    replicates_count: int
    mean_brier: float
    mc_se_brier: float
    mean_log_score: float
    mc_se_log_score: float
    mean_ece: float
    mc_se_ece: float
    mean_aurc: float
    mc_se_aurc: float


def generate_replicate_dataset(
    config: SimulationCellConfig,
    seed: int,
) -> tuple[list[float], list[int], list[int], list[list[int]]]:
    """Generate a reproducible synthetic dataset for one cell replicate.

    Returns:
    - flat forecasts q (length n_base_items * k_within_item)
    - flat clean labels y_true
    - flat observed/noisy labels y_obs
    - grouped item outcomes (list of k outcomes per item)
    """
    rng = random.Random(seed)
    n_p = len(config.p_values)

    forecasts: list[float] = []
    clean_labels: list[int] = []
    observed_labels: list[int] = []
    item_groups: list[list[int]] = []

    for i in range(config.n_base_items):
        p_val = config.p_values[i % n_p]
        q_val = compute_forecast(p_val, config.forecast_kind)

        group: list[int] = []
        if config.shared_within_item:
            # Single Bernoulli draw duplicated across k
            clean_draw = 1 if rng.random() < p_val else 0
            for _ in range(config.k_within_item):
                forecasts.append(q_val)
                clean_labels.append(clean_draw)
                obs_draw = clean_draw
                if (
                    config.reference_flip_eta > 0
                    and rng.random() < config.reference_flip_eta
                ):
                    obs_draw = 1 - obs_draw
                observed_labels.append(obs_draw)
                group.append(obs_draw)
        else:
            # Independent Bernoulli draws
            for _ in range(config.k_within_item):
                clean_draw = 1 if rng.random() < p_val else 0
                forecasts.append(q_val)
                clean_labels.append(clean_draw)
                obs_draw = clean_draw
                if (
                    config.reference_flip_eta > 0
                    and rng.random() < config.reference_flip_eta
                ):
                    obs_draw = 1 - obs_draw
                observed_labels.append(obs_draw)
                group.append(obs_draw)
        item_groups.append(group)

    return forecasts, clean_labels, observed_labels, item_groups


def evaluate_replicate(
    config: SimulationCellConfig,
    seed: int,
) -> ReplicateEvaluation:
    """Evaluate one replicate of a simulation cell."""
    forecasts, clean_labels, observed_labels, item_groups = generate_replicate_dataset(
        config, seed
    )
    brier = brier_score(forecasts, observed_labels)
    log_sc = log_score(forecasts, observed_labels)
    ece_fb, _ = fixed_bin_ece(forecasts, observed_labels, num_bins=10)
    ece_em, _ = equal_mass_ece(forecasts, observed_labels, num_bins=10)

    # For AURC, error indicator is |y_obs - round(q)|
    binary_errors = [
        1 if int(round(q)) != y else 0
        for q, y in zip(forecasts, observed_labels, strict=True)
    ]
    # Confidence is max(q, 1-q)
    confidences = [max(q, 1.0 - q) for q in forecasts]
    aurc_val = compute_aurc(confidences, binary_errors, tie_convention="average")

    se_res = compute_cluster_vs_naive_se(item_groups)

    return ReplicateEvaluation(
        brier_score=brier,
        log_score=log_sc,
        fixed_bin_ece=ece_fb,
        equal_mass_ece=ece_em,
        aurc=aurc_val,
        item_cluster_se=se_res.item_cluster_se,
        naive_row_se=se_res.naive_row_se,
    )


def run_simulation_cell(
    config: SimulationCellConfig,
    num_replicates: int = 2000,
    seed_base: int = 1000,
) -> CellSummary:
    """Run Monte Carlo simulation across independent replicates for one cell."""
    briers: list[float] = []
    log_scores: list[float] = []
    eces: list[float] = []
    aurcs: list[float] = []

    for r in range(num_replicates):
        eval_res = evaluate_replicate(config, seed=seed_base + r)
        briers.append(eval_res.brier_score)
        log_scores.append(eval_res.log_score)
        eces.append(eval_res.fixed_bin_ece)
        aurcs.append(eval_res.aurc)

    def _mean_se(vals: list[float]) -> tuple[float, float]:
        m = sum(vals) / len(vals)
        if len(vals) < 2:
            return m, 0.0
        v = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
        return m, math.sqrt(v / len(vals))

    m_brier, se_brier = _mean_se(briers)
    m_log, se_log = _mean_se(log_scores)
    m_ece, se_ece = _mean_se(eces)
    m_aurc, se_aurc = _mean_se(aurcs)

    return CellSummary(
        config=config,
        replicates_count=num_replicates,
        mean_brier=m_brier,
        mc_se_brier=se_brier,
        mean_log_score=m_log,
        mc_se_log_score=se_log,
        mean_ece=m_ece,
        mc_se_ece=se_ece,
        mean_aurc=m_aurc,
        mc_se_aurc=se_aurc,
    )


# ==============================================================================
# 7. Canonical Deterministic Microprobes Suite (Probes 1 - 5)
# ==============================================================================


def probe_1_prevalence_precision() -> dict[str, float]:
    """Microprobe 1: TPR=0.9, FPR=0.01 at prevalence 0.5 vs 0.01."""
    tpr = 0.9
    fpr = 0.01

    prev_50 = 0.5
    prec_50 = (tpr * prev_50) / (tpr * prev_50 + fpr * (1.0 - prev_50))

    prev_01 = 0.01
    prec_01 = (tpr * prev_01) / (tpr * prev_01 + fpr * (1.0 - prev_01))

    return {
        "precision_prev_50": prec_50,  # 90 / 91 ~= 0.989010989
        "precision_prev_01": prec_01,  # 10 / 21 ~= 0.476190476
    }


def probe_2_coarse_bin_cancellation() -> dict[str, float]:
    """Microprobe 2: Coarse-bin cancellation."""
    forecasts = [0.1] * 10 + [0.9] * 10
    outcomes = [0] * 10 + [1] * 10
    ece_1_bin, _ = fixed_bin_ece(forecasts, outcomes, num_bins=1)
    ece_2_bin, _ = fixed_bin_ece(forecasts, outcomes, num_bins=2)
    return {
        "one_bin_ece": ece_1_bin,  # ~ 0.0
        "two_bin_ece": ece_2_bin,  # = 0.1
    }


def probe_3_constant_uninformative_predictor() -> dict[str, float]:
    """Microprobe 3: Constant q=0.5 predictor on p=0.5."""
    pop_ece = population_ece([0.5], [0.5])
    exp_brier_const = expected_brier(0.5, 0.5)
    exp_brier_oracle = expected_brier(1.0, 1.0)
    return {
        "population_ece": pop_ece,
        "expected_brier_constant": exp_brier_const,
        "expected_brier_oracle": exp_brier_oracle,
    }


def probe_4_reference_noise_ranking_reversal() -> dict[str, Any]:
    """Microprobe 4: Ranking reversal under reference flip noise."""
    return check_reference_ranking_reversal(
        clean_p=0.9, q_clean=0.9, q_noisy=0.74, eta=0.2
    )


def probe_5_pseudoreplication_cluster_se() -> dict[str, float]:
    """Microprobe 5: Pseudoreplication artificially deflates naive row SE."""
    items = [[0] * 25, [0] * 25, [1] * 25, [1] * 25]
    res = compute_cluster_vs_naive_se(items)
    return {
        "item_cluster_se": res.item_cluster_se,
        "naive_row_se": res.naive_row_se,
        "se_ratio": res.se_ratio,
    }


def run_all_deterministic_microprobes() -> dict[str, Any]:
    """Execute and collect all canonical deterministic microprobes."""
    return {
        "probe_1_prevalence_precision": probe_1_prevalence_precision(),
        "probe_2_coarse_bin_cancellation": probe_2_coarse_bin_cancellation(),
        "probe_3_constant_uninformative_predictor": probe_3_constant_uninformative_predictor(),
        "probe_4_reference_noise_ranking_reversal": probe_4_reference_noise_ranking_reversal(),
        "probe_5_pseudoreplication_cluster_se": probe_5_pseudoreplication_cluster_se(),
    }
