import os

# Centralized configuration for runtime variables.
# Set BACKEND_URL to override the default remote backend endpoint.
BACKEND_URL = os.environ.get(
    "BACKEND_URL",
    "https://trustchain-backend-qihp.onrender.com",
)

def get_backend_url() -> str:
    """Return the configured backend URL."""
    return BACKEND_URL
