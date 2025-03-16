"""Errors for the clusx package."""


class ClusxError(Exception):
    """Base class for all Clusx errors."""


class EvaluationError(ClusxError):
    """Error raised when evaluation fails."""


class ClusterIntegrityError(ClusxError):
    """
    Error raised when a cluster assignments file has integrity issues.

    This error indicates that the cluster assignments file is corrupted,
    was created with errors, or is missing critical information needed
    for further processing.

    See Also
    --------
    :class:`MissingClusterColumnError`, :class:`MissingParametersError`
    """


class MissingClusterColumnError(ClusterIntegrityError):
    """
    Error raised when a cluster assignments file is missing the cluster column.

    This error indicates that the file does not contain a column that starts with
    ``Cluster_`` (such as Cluster_PYP or Cluster_DP), which is required for identifying
    cluster assignments.

    See Also
    --------
    ClusterIntegrityError : Parent class for integrity errors
    MissingParametersError : Related error for missing parameters
    """

    def __init__(self, file_path: str):
        """
        Initialize the error with the path to the problematic file.

        Parameters
        ----------
        file_path : str
            Path to the file missing the cluster column
        """
        self.file_path = file_path
        message = (
            f"Integrity error: No cluster column (Cluster_PYP or Cluster_DP) "
            f"found in {file_path}. The file appears to be corrupted or was "
            "not created by the clustering algorithm."
        )
        super().__init__(message)


class MissingParametersError(ClusterIntegrityError):
    """
    Error raised when a cluster assignments file is missing required parameters.

    This error indicates that the file is missing one or more of the required
    parameters (alpha, sigma, variance) needed for further processing.

    Parameters
    ----------
    file_path : str
        Path to the file with missing parameters
    missing_params : list[str]
        List of parameter names that are missing

    See Also
    --------
    ClusterIntegrityError : Parent class for integrity errors
    MissingClusterColumnError : Related error for missing cluster columns
    """

    def __init__(self, file_path: str, missing_params: list[str]):
        """
        Initialize the error with the path to the problematic file and missing parameters.

        Parameters
        ----------
        file_path : str
            Path to the file with missing parameters
        missing_params : list[str]
            List of parameter names that are missing
        """  # noqa: E501
        self.file_path = file_path
        self.missing_params = missing_params
        message = (
            f"Integrity error: Required parameters {', '.join(missing_params)} "
            f"are missing in {file_path}. The file appears to be corrupted "
            "or was created with errors. No further processing is possible."
        )
        super().__init__(message)


class VisualizationError(ClusxError):
    """Error raised when a visualization fails."""
