class SpaceTrackAuthenticationError(Exception):
    """
    Exception raised when Space-Track.org authentication fails.
    """

    def __init__(self, message: str):
        super().__init__(message)
