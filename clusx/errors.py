"""Errors for the clusx package."""


class ClusxError(Exception):
    """Base class for all Clusx errors."""


class EvaluationError(ClusxError):
    """Error raised when evaluation fails."""
