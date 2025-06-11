class AegeanError(Exception):
    pass


class AegeanNaNModelError(AegeanError):
    """
    In some rare instances `lmfit` can pass through parameters whose
    values are NaNs while optimising.
    This can in turn return `lmfit` errors indicating that there may
    be NaNs in the data. It is a little
    misleading.
    """
