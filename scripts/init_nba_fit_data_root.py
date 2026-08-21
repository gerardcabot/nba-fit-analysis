"""Create the external NBA_FIT_DATA_ROOT directory layout (raw, interim, features, models)."""

from __future__ import annotations

import argparse
import sys

from nba_fit.config.settings import ENV_NBA_FIT_DATA_ROOT, ensure_data_root_layout, get_settings


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            f"Create {', '.join(('raw', 'interim', 'features', 'models'))} under "
            f"{ENV_NBA_FIT_DATA_ROOT} or an explicit path."
        ),
    )
    parser.add_argument(
        "path",
        nargs="?",
        help=f"Data root (default: read {ENV_NBA_FIT_DATA_ROOT} from the environment)",
    )
    args = parser.parse_args()
    try:
        if args.path:
            root = ensure_data_root_layout(args.path)
        else:
            settings = get_settings()
            if settings.data_root is None:
                print(
                    f"Set {ENV_NBA_FIT_DATA_ROOT} or pass a path argument.",
                    file=sys.stderr,
                )
                return 1
            root = ensure_data_root_layout(settings.data_root)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"Data root ready: {root}")
    for name in ("raw", "interim", "features", "models"):
        print(f"  {name}/ -> {root / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
