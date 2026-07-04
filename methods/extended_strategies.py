from __future__ import annotations

from methods.catalog_methods import run_catalog_variant
from methods.registry import OPTIMIZERS, register_optimizer


def register_strategy(key: str, name: str, short_name: str, description: str, mode: str) -> None:
    if key in OPTIMIZERS:
        return
    register_optimizer(
        key,
        {
            "name": name,
            "shortName": short_name,
            "description": description,
        },
        lambda demo, batch_size, selected_mode=mode, method_name=name: run_catalog_variant(
            demo,
            batch_size,
            selected_mode,
            method_name,
        ),
    )


STRATEGIES = [
    ("adaptive_grid_archive", "Adaptive Grid Archive", "AGA", "Grid archive search that emphasizes sparse objective-space boxes.", "grid_moea"),
    ("epsilon_box_moea", "Epsilon-box MOEA", "epsMOEA", "Epsilon-box archive optimizer for diversity-preserving Pareto search.", "epsilon_archive"),
    ("crowding_archive_ea", "Crowding Archive EA", "CA-EA", "Archive-based evolutionary search with crowding pressure.", "two_archive"),
    ("decomposition_archive", "Decomposition Archive", "DA", "Decomposition-guided Pareto search using archive leaders.", "moea_dd"),
    ("reference_vector_archive", "Reference Vector Archive", "RVA", "Reference-vector optimizer for three or more objectives.", "ar_moea"),
    ("indicator_archive", "Indicator Archive", "IA", "Indicator-guided optimizer using preference and dominance proxies.", "r2_ibea"),
    ("angle_distance_moea", "Angle-distance MOEA", "AD-MOEA", "Angle-distance selection for many-objective diversity.", "vaea"),
    ("knee_point_moea", "Knee-point MOEA", "KP-MOEA", "Preference-friendly search biased toward knee regions.", "r2_ibea"),
    ("dual_archive_search", "Dual Archive Search", "DAS", "Two-archive optimizer balancing convergence and diversity.", "two_archive"),
    ("hypervolume_search", "Hypervolume Search", "HV-EA", "Hypervolume-contribution style evolutionary search.", "hype"),
    ("sparse_variable_search", "Sparse Variable Search", "SVS", "Large-scale sparse mutation strategy for high-dimensional tasks.", "sparse_moea"),
    ("competitive_swarm_mo", "Competitive Swarm MO", "CS-MO", "Competitive swarm optimizer for multi-objective search.", "lmocso"),
    ("leader_swarm_mo", "Leader Swarm MO", "LS-MO", "Leader-attracted swarm search with archive guidance.", "mopso"),
    ("turbulent_swarm_mo", "Turbulent Swarm MO", "TS-MO", "Swarm search with turbulence for escaping local fronts.", "omopso"),
    ("bounded_velocity_swarm", "Bounded Velocity Swarm", "BVS", "Speed-constrained swarm optimizer for stable local refinement.", "smpso"),
    ("covariance_pareto_es", "Covariance Pareto ES", "CP-ES", "Covariance-guided Pareto evolution strategy.", "mo_cma_es"),
    ("operator_portfolio_ea", "Operator Portfolio EA", "OP-EA", "Adaptive portfolio of mutation, crossover, and differential operators.", "operator_moea"),
    ("differential_archive_mo", "Differential Archive MO", "DA-DE", "Differential-evolution style archive search.", "operator_moea"),
    ("constraint_repair_moea", "Constraint Repair MOEA", "CR-MOEA", "Constrained optimizer with repair-oriented archive sampling.", "ctaea"),
    ("push_pull_archive", "Push-pull Archive", "PPA", "Two-stage constrained search balancing feasibility and Pareto pressure.", "ctaea"),
    ("preference_decomposition", "Preference Decomposition", "PD-MO", "Decomposition search guided by current preference weights.", "moea_dd"),
    ("transfer_archive_mo", "Transfer Archive MO", "TA-MO", "Transfer-biased multi-objective search using archive structure.", "momfea"),
    ("adaptive_operator_moea", "Adaptive Operator MOEA", "AO-MOEA", "Operator portfolio search with archive feedback.", "borg"),
    ("surrogate_reference_mo", "Surrogate Reference MO", "SR-MO", "Surrogate-assisted reference-vector optimizer for expensive tasks.", "krvea"),
    ("trust_region_mobo", "Trust-region MOBO", "TR-MOBO", "Expensive multi-objective BO with local trust-region sampling.", "turbo_mobo"),
    ("hypervolume_ei_mobo", "Hypervolume EI MOBO", "HVEI", "Expected-improvement style expensive optimizer for Pareto progress.", "ego_mobo"),
    ("preference_ei_mobo", "Preference EI MOBO", "PEI", "Preference-weighted expected-improvement style expensive optimizer.", "mobo_ei"),
    ("ucb_pareto_mobo", "UCB Pareto MOBO", "UCB-P", "Uncertainty-guided expensive optimizer using UCB-style ranking.", "mobo_ucb"),
    ("entropy_pareto_mobo", "Entropy Pareto MOBO", "ENT-P", "Entropy-seeking expensive optimizer for uncertain Pareto regions.", "pesmo_bo"),
    ("max_value_mobo", "Max-value MOBO", "MV-MOBO", "Max-value entropy style expensive multi-objective optimizer.", "mesmo_bo"),
    ("knowledge_gradient_mobo", "Knowledge Gradient MOBO", "KG-MOBO", "Knowledge-gradient style expensive optimizer.", "kg_mobo"),
    ("robust_archive_moea", "Robust Archive MOEA", "RA-MOEA", "Archive search biased toward stable objective-space improvement.", "two_archive"),
    ("many_objective_regularized", "Many-objective Regularized EA", "MR-EA", "Regularized many-objective search with reference alignment.", "ar_moea"),
    ("large_scale_cooperative_mo", "Large-scale Cooperative MO", "LSC-MO", "Cooperative sparse search for high-dimensional multi-objective tasks.", "lmocso"),
    ("random_weight_archive", "Random Weight Archive", "RWA", "Randomized scalar preference search with Pareto archive retention.", "r2_ibea"),
    ("local_pareto_pattern", "Local Pareto Pattern Search", "LPPS", "Local Pareto refinement around non-dominated archive points.", "paes"),
]


for item in STRATEGIES:
    register_strategy(*item)
