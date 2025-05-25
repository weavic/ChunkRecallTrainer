import logging
import sys

# Create a logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Add a console handler only if the logger has no handlers
if not logger.hasHandlers():
    # Create a console handler and set its format
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)

    # Add the console handler to the logger
    logger.addHandler(console_handler)

if __name__ == "__main__":
    # Example usage (will only run if this script is executed directly)
    logger.info("Logger initialized and configured.")
    logger.warning("This is a warning message.")
    logger.error("This is an error message.")
