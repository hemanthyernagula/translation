"""
    This is logging module
"""
import sys
from loguru import logger

logger.add(sys.stdout, format="{time} - {level} - {message}", filter="sub.module")
logger.add("log_files/file_{time}.log", level="ERROR", rotation="1000 MB")
