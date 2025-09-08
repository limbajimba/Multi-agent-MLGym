"""
Copyright (c) Meta Platforms, Inc. and affiliates.

Constants for MLGym framework.
"""

# Default step limits
DEFAULT_MAX_STEPS = 300
DEFAULT_MAX_SUPERVISOR_STEPS = 10
DEFAULT_MAX_AGENTS_PER_WORKFLOW = 10
DEFAULT_MAX_STEPS_PER_AGENT = 30

# Default model parameters
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 0.95  # Default to 1.0 to avoid issues with models that don't support it

# Default cost limits
DEFAULT_TOTAL_COST_LIMIT = 3.0
DEFAULT_PER_INSTANCE_COST_LIMIT = 1.0

# Default check-in interval for supervisor agents
DEFAULT_CHECK_IN_INTERVAL = 10

# Default container settings
DEFAULT_CONTAINER_TYPE = "podman"
DEFAULT_SEED = 42


