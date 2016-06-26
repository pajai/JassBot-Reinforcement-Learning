import logging
import sys

reload(logging)

logger = logging.getLogger('')
logger.setLevel(logging.WARNING)
format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(format)
ch.setLevel(logging.INFO)
logger.addHandler(ch)

fh = logging.FileHandler('./logs/jassbot.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(format)
logger.addHandler(fh)
