"""Re-export OffspringCreationScheduler for compatible import paths.

The implementation and its validation and persistence logic reside in
offspring_registration_scheduler."""

from modules.offspring_registration_scheduler import (
    OffspringCreationResult,
    OffspringCreationScheduler,
    OffspringRegistrationError,
    OffspringRegistrationResult,
)


__all__ = (
    "OffspringCreationResult",
    "OffspringCreationScheduler",
    "OffspringRegistrationError",
    "OffspringRegistrationResult",
)
