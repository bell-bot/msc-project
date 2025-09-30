from contextlib import contextmanager
import logging
import time

from tqdm import tqdm


class TqdmLoggingHandler(logging.Handler):
    """A logging handler that uses tqdm.write() to prevent conflicts with progress bars."""

    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)


@contextmanager
def timed_operation(description: str, logger: logging.Logger, log_details: dict | None = {}):
    """A context manager to time operations, showing a temporary progress bar
    and a final summary message."""

    start_time = time.time()
    logger.info(f"{description}...")

    pbar = tqdm(total=None, bar_format="{desc}", desc=f"  ⏳ {description}...")

    try:
        yield
    finally:
        elapsed_time = time.time() - start_time
        pbar.close()

        summary = f"✓ {description} finished in {elapsed_time:.2f}s."
        if log_details:
            details_str = ", ".join([f"{k}: {v}" for k, v in log_details.items()])
            summary += f" ({details_str})"

        logger.info(summary)
        tqdm.write(summary)


class TimedLogger(logging.Logger):
    """A custom logger that includes a .time() context manager."""

    @contextmanager
    def time(
        self,
        description: str,
        log_details: dict | None = None,
        pbar: tqdm | None = None,
        step_info: str = "",
        show_pbar: bool = True,
    ):

        start_time = time.time()
        self.info(f"{step_info}{description}...")

        active_pbar = None

        if show_pbar:
            pbar_desc = f"{step_info}{description}"
            if pbar is None:
                active_pbar = tqdm(total=None, bar_format="{desc}", desc=f"  ⏳ {pbar_desc}...")
            else:
                active_pbar = pbar
                active_pbar.set_description(pbar_desc)

        try:
            yield
        finally:
            elapsed_time = time.time() - start_time

            if active_pbar is not None and pbar is None:
                active_pbar.close()

            summary = f"✓ {description} finished in {elapsed_time:.2f}s."
            if log_details:
                details_str = ", ".join([f"{k}: {v}" for k, v in log_details.items()])
                summary += f" ({details_str})"

            self.info(f"{step_info}{summary}")

            if active_pbar is not None and pbar is None:
                tqdm.write(summary)
