#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jul. 20 09:37 a.m. 2024

@author: AI group, Department of Information Engineering, University of Pisa
"""

import sys
import logging
from typing import Callable

logging_formatter_str: str = "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"

logger = logging.getLogger()
logFormatter: logging.Formatter = logging.Formatter(logging_formatter_str)

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)
logger.setLevel(logging.INFO)

logger_info: Callable = logger.info
logger_error: Callable = logger.error
