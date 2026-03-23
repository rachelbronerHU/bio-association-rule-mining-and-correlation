import logging

logger = logging.getLogger(__name__)

def validate_config(config, algo, methods):
    """
    Validates config keys at pipeline startup and fills in defaults for optional parameters.
    Mutates config in-place.
    """
    # 1. Fill defaults for optional parameters and log warnings
    defaults = {
        "MAX_NEGATIVE_LEVERAGE": -0.001,
        "MAX_NEGATIVE_LIFT": 0.7,
        "MIN_CONVICTION": 1.0,
        "MIN_REDUNDANCY_LIFT_IMPROVEMENT": 1.1,
    }
    
    for key, default_val in defaults.items():
        if key not in config:
            config[key] = default_val
            logger.warning(f"Optional config key {key!r} missing. Using default: {default_val}")

    # 2. Strict validation for required parameters
    required = [
        "RADIUS", "K_NEIGHBORS", "MIN_SUPPORT", "MIN_CONFIDENCE", "MIN_LIFT",
        "MIN_LEVERAGE", "MAX_RULE_LENGTH", "N_PERMUTATIONS",
        "N_TOP_RULES", "MIN_CELL_TYPE_FREQUENCY",
    ]
    for key in required:
        if key not in config:
            raise ValueError(f"CONFIG missing required key: {key!r}")

    # 3. Value range validation
    if not (0 < config["MIN_SUPPORT"] < 1):
        raise ValueError(f"MIN_SUPPORT must be in (0, 1), got {config['MIN_SUPPORT']}")
    if not (0 <= config["MIN_CONFIDENCE"] <= 1):
        raise ValueError(f"MIN_CONFIDENCE must be in [0, 1], got {config['MIN_CONFIDENCE']}")
    if config["MIN_LIFT"] <= 0:
        raise ValueError(f"MIN_LIFT must be > 0, got {config['MIN_LIFT']}")
    if config["RADIUS"] <= 0:
        raise ValueError(f"RADIUS must be > 0, got {config['RADIUS']}")
    if config["K_NEIGHBORS"] < 1:
        raise ValueError(f"K_NEIGHBORS must be >= 1, got {config['K_NEIGHBORS']}")
    if config["MAX_RULE_LENGTH"] < 2:
        raise ValueError(f"MAX_RULE_LENGTH must be >= 2, got {config['MAX_RULE_LENGTH']}")
    if config["N_PERMUTATIONS"] < 0:
        raise ValueError(f"N_PERMUTATIONS must be >= 0, got {config['N_PERMUTATIONS']}")
    if config.get("MIN_ABS_SUPPORT", 0) < 0:
        raise ValueError(f"MIN_ABS_SUPPORT must be >= 0, got {config['MIN_ABS_SUPPORT']}")

    if algo == "weighted_fpgrowth":
        if "BANDWIDTH" not in config:
            raise ValueError("CONFIG missing 'BANDWIDTH' required by weighted_fpgrowth")
        if config["BANDWIDTH"] <= 0:
            raise ValueError(f"BANDWIDTH must be > 0, got {config['BANDWIDTH']}")
        invalid_methods = set(methods) - {"CN", "KNN_R"}
        if invalid_methods:
            raise ValueError(f"weighted_fpgrowth does not support methods: {invalid_methods}")
