"""Selection des providers par nom. Un seul endroit a modifier pour changer de
fournisseur ; le pipeline ne connait jamais un SDK."""

from __future__ import annotations

import os

from babarug.providers.base import ProviderError

_VISION = {}
_GENERATION = {}
_SEGMENTATION = {}


def _lazy():
    """Import tardif : ne pas exiger une cle Gemini pour lancer les tests."""
    if _VISION:
        return
    from babarug.providers.generation import (
        FluxKontextProvider, GeminiImageProvider, OfflineSceneProvider,
    )
    from babarug.providers.scene_library import SceneLibraryProvider
    from babarug.providers.segmentation import LocalGrabCutProvider, RemoteMattingProvider
    from babarug.providers.vision_claude import ClaudeVisionProvider

    _VISION.update(claude=ClaudeVisionProvider)
    _GENERATION.update(gemini=GeminiImageProvider, flux=FluxKontextProvider,
                       library=SceneLibraryProvider, offline=OfflineSceneProvider)
    _SEGMENTATION.update(local=LocalGrabCutProvider, remote=RemoteMattingProvider)


def get_vision(name: str | None = None, **kw):
    _lazy()
    name = name or os.environ.get("BABARUG_VISION", "claude")
    if name not in _VISION:
        raise ProviderError(f"VisionProvider inconnu : {name} (dispo : {list(_VISION)})")
    return _VISION[name](**kw)


def get_generation(name: str | None = None, **kw):
    _lazy()
    name = name or os.environ.get("BABARUG_GENERATION", "offline")
    if name not in _GENERATION:
        raise ProviderError(f"ImageGenerationProvider inconnu : {name} (dispo : {list(_GENERATION)})")
    return _GENERATION[name](**kw)


def get_segmentation(name: str | None = None, **kw):
    _lazy()
    name = name or os.environ.get("BABARUG_SEGMENTATION", "local")
    if name not in _SEGMENTATION:
        raise ProviderError(f"SegmentationProvider inconnu : {name} (dispo : {list(_SEGMENTATION)})")
    return _SEGMENTATION[name](**kw)


def available() -> dict:
    _lazy()
    return {"vision": list(_VISION), "generation": list(_GENERATION),
            "segmentation": list(_SEGMENTATION)}
