import os
import logging
from datetime import datetime
import structlog


class CustomLogger:
    def __init__(self, log_dir="logs"):
        """
        Custom structured logger for FloatChat project.
        Logs to both console (stdout) and timestamped file in JSON format.
        """
        # Ensure logs directory exists
        self.logs_dir = os.path.join(os.getcwd(), log_dir)
        os.makedirs(self.logs_dir, exist_ok=True)

        # Timestamped log file (for persistence)
        log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
        self.log_file_path = os.path.join(self.logs_dir, log_file)

    def get_logger(self, name=__file__):
        """
        Returns a structlog logger configured for console + file JSON logging.
        """
        logger_name = os.path.basename(name)

        # File handler for persistent logs
        file_handler = logging.FileHandler(self.log_file_path)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("%(message)s"))  # Raw JSON

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(message)s"))

        # Configure Python logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",  # Structlog renders JSON
            handlers=[console_handler, file_handler],
            force=True  # Important: prevents duplicate handlers when re-importing
        )

        # Configure structlog
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                structlog.processors.add_log_level,
                structlog.processors.EventRenamer(to="event"),
                structlog.processors.JSONRenderer()
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )

        return structlog.get_logger(logger_name)


# # --- Usage Example ---
# if __name__ == "__main__":
#     logger = CustomLogger().get_logger(__file__)

#     # Data ingestion logs
#     logger.info("NetCDF file loaded", file="argo_float_123.nc", rows=1500)

#     # Database logs
#     logger.info("Inserted profiles into PostgreSQL", table="argo_profiles", rows=1500)

#     # RAG pipeline logs
#     logger.info("User query processed", query="Show me salinity profiles near equator", sql="SELECT ...")

#     # Visualization logs
#     logger.info("Generated profile plot", float_id="2903350", variable="salinity")

#     # API logs
#     logger.error("API request failed", endpoint="/query", error="Timeout")
