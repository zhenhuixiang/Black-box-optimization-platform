from __future__ import annotations

from methods.catalog_methods import run_catalog_variant
from methods.registry import OPTIMIZERS, register_optimizer


def register_catalog_method(key: str, name: str, short_name: str, category: str, mode: str) -> None:
    if key in OPTIMIZERS:
        return
    register_optimizer(
        key,
        {
            "name": name,
            "shortName": short_name,
            "description": f"BlackBoxLab catalog entry for {category}; mapped to a runnable {mode} ask-tell strategy.",
        },
        lambda demo, batch_size, selected_mode=mode, method_name=name: run_catalog_variant(
            demo,
            batch_size,
            selected_mode,
            method_name,
        ),
    )


METHOD_CATALOG = [
    # Reference-vector and many-objective algorithms
    ("a_nsga3", "A-NSGA-III", "A-NSGA-III", "adaptive reference-vector many-objective optimization", "ar_moea"),
    ("ae_nsga2", "AE-NSGA-II", "AE-NSGA-II", "adaptive evolutionary Pareto search", "operator_moea"),
    ("age2", "AGE-II", "AGE-II", "geometry estimation based many-objective optimization", "vaea"),
    ("age_moea2", "AGE-MOEA-II", "AGE-II", "geometry estimation based many-objective optimization", "vaea"),
    ("adaw", "AdaW", "AdaW", "adaptive weight-vector many-objective optimization", "ar_moea"),
    ("bi_ge", "BiGE", "BiGE", "bi-goal evolution for many-objective optimization", "vaea"),
    ("dcnsga3", "DCNSGA-III", "DCNSGA-III", "dual-control reference-vector search", "ar_moea"),
    ("maoea_hap", "MaOEA-HAP", "HAP", "many-objective adaptive preference search", "vaea"),
    ("maoea_igd", "MaOEA-IGD", "IGD-MaOEA", "many-objective IGD-guided search", "r2_ibea"),
    ("maoea_it", "MaOEA-IT", "IT-MaOEA", "many-objective indicator transformation search", "r2_ibea"),
    ("maoea_rd", "MaOEA-RD", "RD-MaOEA", "many-objective region-division search", "ar_moea"),
    ("nsga3_wa", "NSGA-III-WA", "NSGA-WA", "weighted-adaptive NSGA-III style search", "ar_moea"),
    ("rvea_a", "RVEAa", "RVEAa", "adaptive reference-vector evolutionary search", "ar_moea"),
    ("tdea", "tDEA", "tDEA", "theta-dominance evolutionary search", "moea_dd"),
    ("one_by_one_ea", "One-by-one EA", "1by1EA", "incremental many-objective evolutionary search", "vaea"),
    ("kn_ea", "KnEA", "KnEA", "knee-point driven evolutionary search", "r2_ibea"),

    # Decomposition and weight-vector algorithms
    ("bce_moead", "BCE-MOEA/D", "BCE-D", "bi-criterion evolution with decomposition", "moea_dd"),
    ("c_moead", "C-MOEA/D", "C-D", "constrained decomposition-based optimization", "ctaea"),
    ("eag_moead", "EAG-MOEA/D", "EAG-D", "external archive guided decomposition", "moea_dd"),
    ("ens_moead", "ENS-MOEA/D", "ENS-D", "efficient non-dominated sorting with decomposition", "moea_dd"),
    ("moead_de", "MOEA/D-DE", "D-DE", "differential evolution decomposition baseline", "moea_dd"),
    ("moead_dra", "MOEA/D-DRA", "D-DRA", "dynamic resource allocation decomposition", "moea_dd"),
    ("moead_m2m", "MOEA/D-M2M", "D-M2M", "many-to-many decomposition search", "moea_dd"),
    ("moead_pas", "MOEA/D-PaS", "D-PaS", "Pareto adaptive scalarization decomposition", "moea_dd"),
    ("moead_stm", "MOEA/D-STM", "D-STM", "stable matching decomposition", "moea_dd"),
    ("moead_uraw", "MOEA/D-URAW", "D-URAW", "utility-ratio adaptive weighting decomposition", "moea_dd"),
    ("moea_ib", "MOEA-IB", "MOEA-IB", "indicator-based decomposition search", "moea_dd"),
    ("moea_psl", "MOEA-PSL", "PSL", "Pareto set learning decomposition", "moea_dd"),

    # Indicator, grid, and archive algorithms
    ("bce_ibea", "BCE-IBEA", "BCE-IBEA", "bi-criterion indicator-based evolution", "r2_ibea"),
    ("c3m", "C3M", "C3M", "clustering and convergence archive search", "two_archive"),
    ("ca_moea", "CA-MOEA", "CA-MOEA", "clustering archive evolutionary search", "two_archive"),
    ("caead", "CAEAD", "CAEAD", "archive-assisted decomposition search", "grid_moea"),
    ("csea", "CSEA", "CSEA", "clustering selection evolutionary algorithm", "grid_moea"),
    ("daea", "DAEA", "DAEA", "diversity archive evolutionary algorithm", "two_archive"),
    ("dea_gng", "DEA-GNG", "DEA-GNG", "growing-neural-gas diversity archive search", "grid_moea"),
    ("edn_armoea", "EDN-ARMOEA", "EDN-AR", "edge-detection angle-regularity search", "ar_moea"),
    ("picea_g", "PICEA-g", "PICEA-g", "preference-inspired co-evolutionary archive search", "r2_ibea"),
    ("rsea", "RSEA", "RSEA", "reference set evolutionary search", "r2_ibea"),
    ("smea", "SMEA", "SMEA", "shift-based density evolutionary search", "grid_moea"),
    ("spea2_sde", "SPEA2-SDE", "SPEA-SDE", "strength Pareto search with shift-based density", "r2_ibea"),
    ("spea_r", "SPEA/R", "SPEA/R", "reference-direction strength Pareto search", "r2_ibea"),

    # Constrained multi-objective algorithms
    ("bi_co", "BiCo", "BiCo", "bilevel constrained multi-objective search", "ctaea"),
    ("ccmo", "CCMO", "CCMO", "cooperative constrained multi-objective optimization", "ctaea"),
    ("cglp", "CGLP", "CGLP", "constraint-guided learning population search", "ctaea"),
    ("clia", "CLIA", "CLIA", "constraint learning immune archive search", "ctaea"),
    ("cmdeipcm", "CMDEIPCM", "CMDE", "constrained DE with infeasible population management", "ctaea"),
    ("cmegl", "CMEGL", "CMEGL", "constraint manifold evolutionary guidance", "ctaea"),
    ("emcmo", "EMCMO", "EMCMO", "evolutionary multitasking constrained optimization", "ctaea"),
    ("cmoea_2s", "CMOEA-2S", "2S-CMOEA", "two-stage constrained multi-objective optimization", "ctaea"),
    ("cmoea_aop", "CMOEA-AOP", "AOP", "adaptive operator constrained optimization", "ctaea"),
    ("cmoea_cd", "CMOEA-CD", "CD-CMOEA", "constraint-dominance evolutionary search", "ctaea"),
    ("cmoea_ms", "CMOEA-MS", "MS-CMOEA", "multi-stage constrained search", "ctaea"),
    ("cmoea_msg", "CMOEA-MSG", "MSG", "multi-strategy constrained search", "ctaea"),
    ("cmoes", "CMOES", "CMOES", "constrained multi-objective evolution strategy", "ctaea"),
    ("cmopso", "CMOPSO", "CMOPSO", "constrained multi-objective particle swarm optimization", "ctaea"),
    ("cps_moea", "CPS-MOEA", "CPS", "push-pull constrained evolutionary search", "ctaea"),
    ("ilcmo", "ILCMO", "ILCMO", "infeasible learning constrained multi-objective search", "ctaea"),
    ("prcea", "PRCEA", "PRCEA", "preference-region constrained evolutionary search", "ctaea"),

    # Large-scale and sparse algorithms
    ("afsea", "AFSEA", "AFSEA", "adaptive feature selection evolutionary search", "sparse_moea"),
    ("amg_psl", "AMG-PSL", "AMG-PSL", "adaptive manifold-guided Pareto set learning", "sparse_moea"),
    ("ccgde3", "CCGDE3", "CCGDE3", "cooperative coevolutionary differential search", "sparse_moea"),
    ("dgea", "DGEA", "DGEA", "directional grouping evolutionary search", "sparse_moea"),
    ("dpcpra", "DPCPRA", "DPCPRA", "decision-variable clustering and Pareto region adaptation", "sparse_moea"),
    ("dpvaps", "DPVAPS", "DPVAPS", "decision-variable analysis Pareto search", "sparse_moea"),
    ("dssea", "DSSEA", "DSSEA", "decision-space sparse evolutionary algorithm", "sparse_moea"),
    ("dvcea", "DVCEA", "DVCEA", "decision variable clustering evolutionary algorithm", "sparse_moea"),
    ("fdv", "FDV", "FDV", "feature decision-variable search", "sparse_moea"),
    ("gdvtsf", "GDVTSF", "GDVTSF", "grouped decision-variable two-stage framework", "sparse_moea"),
    ("glmo", "GLMO", "GLMO", "large-scale global-local multi-objective search", "lmocso"),
    ("lmea", "LMEA", "LMEA", "large-scale multi-objective evolutionary algorithm", "lmocso"),
    ("lsmo_f", "LSMOF", "LSMOF", "large-scale multi-objective framework", "lmocso"),
    ("moea_vd", "MOEA-VD", "MOEA-VD", "variable-decomposition large-scale search", "sparse_moea"),
    ("sparse_emt", "SparseEMT", "SparseEMT", "sparse evolutionary multitasking search", "sparse_moea"),

    # Particle swarm and differential evolution variants
    ("adsapso", "ADSAPSO", "ADSAPSO", "adaptive dynamic strategy PSO", "mopso"),
    ("cmocso", "CMOCSO", "CMOCSO", "competitive swarm constrained multi-objective optimization", "mopso"),
    ("cmode_ftr", "CMODE-FTR", "CMODE", "constrained multi-objective differential evolution", "operator_moea"),
    ("dnsga2", "DNSGA-II", "DNSGA-II", "dynamic non-dominated sorting genetic algorithm", "operator_moea"),
    ("dn_nsga2", "DN-NSGA-II", "DN-NSGA", "dual-neighborhood NSGA-II style search", "operator_moea"),
    ("dspcmde", "DSPCMDE", "DSPCMDE", "dynamic selection pressure constrained DE", "operator_moea"),
    ("heia", "HEIA", "HEIA", "hybrid evolutionary immune algorithm", "operator_moea"),
    ("mopso_cdr", "MOPSO-CDR", "MOPSO-CDR", "crowding-distance PSO with roulette leader selection", "mopso"),
    ("mpsod", "MPSOD", "MPSOD", "decomposition-based multi-objective PSO", "mopso"),
    ("s_nsga2", "S-NSGA-II", "S-NSGA-II", "steady-state NSGA-II style search", "operator_moea"),

    # Multimodal, multitask, dynamic, and preference algorithms
    ("ac_mmea", "AC-MMEA", "AC-MMEA", "adaptive clustering multimodal multi-objective search", "two_archive"),
    ("cmm_e", "CMME", "CMME", "constrained multimodal multi-objective evolution", "two_archive"),
    ("cmmo", "CMMO", "CMMO", "constrained multimodal multi-objective optimization", "two_archive"),
    ("commea", "CoMMEA", "CoMMEA", "cooperative multimodal multi-objective evolution", "two_archive"),
    ("dbemto", "DBEMTO", "DBEMTO", "dynamic bilevel evolutionary multitasking optimization", "momfea"),
    ("dm_moea", "DM-MOEA", "DM-MOEA", "dynamic multi-objective evolutionary algorithm", "grid_moea"),
    ("dmoea_ec", "DMOEA-eC", "DMOEA-eC", "dynamic multi-objective search with environment classification", "grid_moea"),
    ("dp_pps", "DP-PPS", "DP-PPS", "dual-population push-pull search", "two_archive"),
    ("emcmms", "EMCMMS", "EMCMMS", "evolutionary multitasking for multimodal search", "momfea"),
    ("emmoea", "EMMOEA", "EMMOEA", "evolutionary multimodal multi-objective search", "two_archive"),
    ("mmea_arm", "MMEA-ARM", "MMEA-ARM", "adaptive region multimodal multi-objective search", "two_archive"),
    ("mmoeabh", "MMOEABH", "MMOEABH", "bi-hierarchy multimodal multi-objective search", "two_archive"),
    ("momfea2", "MOMFEA-II", "MOMFEA-II", "multifactorial multi-objective evolutionary optimization", "momfea"),
    ("nsga2_sdr", "NSGA-II-SDR", "SDR", "special crowding-distance NSGA-II search", "two_archive"),
    ("nsga2_tor", "NSGA-II-ToR", "ToR", "target-of-region preference NSGA-II search", "r2_ibea"),

    # Surrogate-assisted and expensive optimization algorithms
    ("ab_saea", "AB-SAEA", "AB-SAEA", "adaptive Bayesian surrogate-assisted evolutionary algorithm", "krvea"),
    ("agsea", "AGSEA", "AGSEA", "adaptive Gaussian surrogate evolutionary algorithm", "krvea"),
    ("apsea", "APSEA", "APSEA", "adaptive preference surrogate evolutionary algorithm", "krvea"),
    ("avg_saea", "AVG-SAEA", "AVG-SAEA", "adaptive vector-guided surrogate-assisted search", "krvea"),
    ("bl_saea", "BL-SAEA", "BL-SAEA", "bilevel surrogate-assisted evolutionary algorithm", "krvea"),
    ("ci_emo", "CI-EMO", "CI-EMO", "confidence-interval expensive multi-objective optimization", "mobo_ucb"),
    ("disk", "DISK", "DISK", "surrogate-assisted dominance and knowledge search", "krvea"),
    ("disk_plus", "DISKplus", "DISK+", "enhanced surrogate-assisted knowledge search", "krvea"),
    ("dkca", "DKCA", "DKCA", "data-driven knowledge clustering algorithm", "krvea"),
    ("drl_saea", "DRL-SAEA", "DRL-SAEA", "deep reinforcement learning assisted expensive optimization", "turbo_mobo"),
    ("dirhv_ei", "DirHV-EI", "DirHV-EI", "directed hypervolume expected-improvement optimization", "ego_mobo"),
    ("eges", "EGES", "EGES", "efficient global expensive search", "ego_mobo"),
    ("eim_ego", "EIM-EGO", "EIM-EGO", "expected improvement matrix efficient global optimization", "mobo_ei"),
    ("em_saea", "EM-SAEA", "EM-SAEA", "ensemble-model surrogate-assisted evolutionary search", "krvea"),
    ("emoskt", "EMOSKT", "EMOSKT", "surrogate knowledge transfer expensive optimization", "pesmo_bo"),
    ("esb_ceo", "ESB-CEO", "ESB-CEO", "expensive surrogate-assisted constrained optimization", "mobo_ucb"),
    ("gcnmoea", "GCNMOEA", "GCNMOEA", "graph-convolution surrogate multi-objective search", "krvea"),
    ("gfm_moea", "GFM-MOEA", "GFM", "Gaussian-field model assisted evolutionary search", "krvea"),
]


for item in METHOD_CATALOG:
    register_catalog_method(*item)
