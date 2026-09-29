"""
Test session setup.

Expose 8 simulated CPU devices so the multi-device code paths
(OceanJAX.parallel) are exercised on a single machine.  XLA reads the flag
when the backend is first initialised, so it has to be set before any test
module touches a device; conftest.py is imported before the test modules.
Single-device code is unaffected: arrays live on device 0 unless they are
explicitly sharded.
"""

import os

_FLAG = "--xla_force_host_platform_device_count=8"

if "xla_force_host_platform_device_count" not in os.environ.get("XLA_FLAGS", ""):
    os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "") + " " + _FLAG).strip()
