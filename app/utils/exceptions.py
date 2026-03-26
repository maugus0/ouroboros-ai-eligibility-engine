"""Custom exception classes for the Eligibility Engine."""


class EligibilityEngineBaseError(Exception):
    """Base exception for all Eligibility Engine errors."""

    def __init__(self, message: str = "An unexpected error occurred", status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class DatabaseError(EligibilityEngineBaseError):
    """Raised when a database operation fails."""

    def __init__(self, message: str = "Database operation failed"):
        super().__init__(message=message, status_code=500)


class NotFoundError(EligibilityEngineBaseError):
    """Raised when a requested resource is not found."""

    def __init__(self, resource: str = "Resource"):
        super().__init__(message=f"{resource} not found", status_code=404)


class ValidationError(EligibilityEngineBaseError):
    """Raised when request validation fails beyond Pydantic checks."""

    def __init__(self, message: str = "Validation failed"):
        super().__init__(message=message, status_code=422)


class ServiceAuthError(EligibilityEngineBaseError):
    """Raised when inter-service authentication fails."""

    def __init__(self, message: str = "Service authentication failed"):
        super().__init__(message=message, status_code=401)


class LLMReasoningError(EligibilityEngineBaseError):
    """Raised when LLM reasoning/explanation generation fails after retries."""

    def __init__(self, message: str = "LLM reasoning failed"):
        super().__init__(message=message, status_code=502)


class ScoringError(EligibilityEngineBaseError):
    """Raised when match scoring computation fails."""

    def __init__(self, message: str = "Scoring computation failed"):
        super().__init__(message=message, status_code=500)


class EmbeddingError(EligibilityEngineBaseError):
    """Raised when vector embedding generation or search fails."""

    def __init__(self, message: str = "Embedding operation failed"):
        super().__init__(message=message, status_code=502)
