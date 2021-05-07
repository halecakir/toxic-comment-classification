import logging
import os

LOG_ROOT_DIR = "logs"
if not os.path.exists(LOG_ROOT_DIR):
    os.mkdir(LOG_ROOT_DIR)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


COLORS = {
    "WARNING": bcolors.WARNING,
    "INFO": bcolors.OKGREEN,
    "DEBUG": bcolors.OKBLUE,
    "ERROR": bcolors.FAIL,
    "END": bcolors.ENDC,
}


class ColorFilter(logging.Filter):
    def filter(self, record):
        record.color = COLORS[record.levelname]
        record.end = COLORS["END"]
        return True


class ColoredLogger(logging.Logger):
    """Custom logger class with multiple destinations"""

    COLOR_FORMAT = (
        "%(color)s%(asctime)s %(name)-12s %(levelname)-8s %(message)s %(end)s"
    )
    FORMAT = "%(asctime)s %(name)-12s %(levelname)-8s %(message)s"

    def __init__(self, name):
        """Init function for Logger class.
        Args:
            name (str): Logger name
        """
        logging.Logger.__init__(self, name)

        # handlers
        console = logging.StreamHandler()
        filename = os.path.join(LOG_ROOT_DIR, "{}.log".format(name))
        filehandler = logging.FileHandler(filename)

        # formatters
        color_formatter = logging.Formatter(self.COLOR_FORMAT)
        formatter = logging.Formatter(self.FORMAT)

        # set formatters and filters
        console.setFormatter(color_formatter)
        console.addFilter(ColorFilter())
        filehandler.setFormatter(formatter)

        # add handlers
        self.addHandler(console)
        self.addHandler(filehandler)
