"""
Auto-MLOps Python SDK

A Python client library for interacting with the Auto-MLOps API.

Usage:
    from sdk import MLOpsClient

    client = MLOpsClient(api_key="your-api-key")
    result = client.run("Set up MLOps pipeline for my project")
    print(result.status)
"""

from sdk.async_client import AsyncMLOpsClient
from sdk.client import MLOpsClient

__all__ = ["MLOpsClient", "AsyncMLOpsClient"]
__version__ = "1.0.0"
