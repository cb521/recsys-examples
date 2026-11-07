"""Entry point for ``VERL_USE_EXTERNAL_MODULES=ai_search_plugin.verl_plugin``.

verl imports this module from ``verl/__init__.py`` in every process.  It refuses deprecated switches and installs the
features listed in ``ASRT_VERL`` (written by ``ai_search_plugin.install(framework="verl")`` /
``ai_search_plugin.frameworks.verl.plan``).  The features are post-import hooks, so verl modules imported later are
patched when they load.  ``ASRT_VERL`` unset / ``off`` = plain verl.
"""
import os

from . import registry as _registry

_bad = _registry.deprecated_env_in(os.environ)
if _bad:
    raise RuntimeError(f"ai_search_plugin: deprecated switches set: {_bad}")

from .adapters import verl as _verl  # noqa: E402

installed = _verl.install()
if installed and os.environ.get("ASRT_VERL_QUIET", "0") != "1":
    print(f"[asrt] ai_search_plugin verl pid={os.getpid()} features={sorted(installed)}", flush=True)
