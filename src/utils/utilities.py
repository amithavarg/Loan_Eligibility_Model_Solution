import logging
import os

def configure_logging():
    """Configure logging for the application."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger

def save_figure(fig, filename, directory='figures'):
    """Save the matplotlib figure to a file."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, filename)
    fig.savefig(file_path)
    logging.info(f"Figure saved to {file_path}.")
