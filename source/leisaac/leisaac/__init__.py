from pkgutil import extend_path
from importlib import import_module


__path__ = extend_path(__path__, __name__)

# Preserve leisaac's package side effects while allowing local task overrides.
from .tasks import *  # noqa: F401,F403
monkey_patch = import_module(f"{__name__}.utils").monkey_patch
