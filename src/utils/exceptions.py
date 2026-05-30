class FraudDetectionException(Exception):
    """
    Custom exception class for project-specific errors.
    """

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message