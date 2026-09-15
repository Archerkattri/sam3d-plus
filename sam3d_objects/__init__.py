# Copyright (c) Meta Platforms, Inc. and affiliates.
import os

# Allow skipping initialization for lightweight tools
if not os.environ.get('LIDRA_SKIP_INIT'):
    try:
        import sam3d_objects.init
    except ModuleNotFoundError as exc:
        # The acceleration-only source snapshot does not carry Meta's optional
        # environment/bootstrap module. Keep lightweight solver imports usable,
        # but do not hide unrelated missing dependencies; a complete upstream
        # checkout still follows the normal initialization path above.
        if exc.name != "sam3d_objects.init":
            raise
        import warnings

        warnings.warn(
            "sam3d_objects.init is absent; running the source-only acceleration "
            "surface without upstream environment initialization",
            RuntimeWarning,
            stacklevel=2,
        )
