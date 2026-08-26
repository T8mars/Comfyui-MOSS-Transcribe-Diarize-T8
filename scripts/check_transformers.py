from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from packaging.version import Version


try:
    installed = Version(version("transformers"))
except PackageNotFoundError:
    installed = None

if installed is not None and Version("5.5.0") <= installed < Version("6"):
    print(f"Transformers {installed} is supported and includes the required security fixes.")
elif installed is not None and Version("4.52.1") <= installed < Version("5.5.0"):
    print(f"Transformers {installed} is runtime-compatible but affected by published security advisories.")
    print("After checking other custom nodes, upgrade with: pip install -r requirements-transformers-v5.txt")
    raise SystemExit(1)
elif installed is None or installed < Version("4.52.1"):
    print(f"Transformers {installed or 'not installed'} is too old.")
    print("Install the security-patched repair: pip install -r requirements-transformers-v5.txt")
    raise SystemExit(1)
else:
    print(f"Transformers {installed} is newer than the validated range (<6).")
    raise SystemExit(1)
