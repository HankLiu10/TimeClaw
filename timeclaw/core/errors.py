"""Errors that invalidate execution evidence rather than ordinary tool failures."""

class ExecutionContractError(RuntimeError):
    """Fatal execution-contract failure; never turn it into a repair observation."""
