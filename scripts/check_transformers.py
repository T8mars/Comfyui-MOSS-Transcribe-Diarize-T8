from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version


try:
    installed = Version(version("transformers"))
except PackageNotFoundError:
    installed = None

if installed is not None and Version("4.52.1") <= installed < Version("6"):
    print(f"Transformers {installed} is supported; no environment change is needed.")
elif installed is None or installed < Version("4.52.1"):
    print(f"Transformers {installed or 'not installed'} is too old.")
    print("Install the conservative v4 repair: pip install -r requirements-transformers-v4.txt")
    raise SystemExit(1)
else:
    print(f"Transformers {installed} is newer than the validated range (<6).")
    raise SystemExit(1)
