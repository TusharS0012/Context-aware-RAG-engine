import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import httpx

# Setup logging to see retries in the console
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def retry_on_api_limit():
    """
    Decorator to handle Hugging Face rate limits and transient errors.
    It will retry up to 5 times with exponential wait times.
    """
    return retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        # Retry if we get a 429 (Rate Limit) or 503 (Service Unavailable/Loading)
        retry=retry_if_exception_type((httpx.HTTPStatusError, httpx.ConnectError)),
        before_sleep=lambda retry_state: logger.info(
            f"Retrying API call... Attempt {retry_state.attempt_number}"
        )
    )