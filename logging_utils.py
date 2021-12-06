#########################
# Utilities for logging #
#########################

import logging
import sys
import wandb
from logging.handlers import TimedRotatingFileHandler
import torch

from wandb.util import WandBHistoryJSONEncoder
FORMATTER = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
LOG_FILE = "exp.log"

def get_console_handler():
   console_handler = logging.StreamHandler(sys.stdout) # log to stdout
   # console_handler = logging.StreamHandler() # log to stderr
   console_handler.setFormatter(FORMATTER)
   return console_handler

def get_file_handler():
   file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
   file_handler.setFormatter(FORMATTER)
   return file_handler

def print_gpu_info(m=''):
   print(m)
   for i in range(torch.cuda.device_count()):
      d = 1024 ** 3
      t = torch.cuda.get_device_properties(i).total_memory / d
      r = torch.cuda.memory_reserved(i) / d 
      a = torch.cuda.memory_allocated(i) / d
      f = r-a  # free inside reserved
      print(f'Device: {i}\tTotal: {t:.2f}G\tReserved: {r*100/t:.1f}%\tAllocated: {a*100/t:.1f}%')

def get_logger(logger_name):
   logger = logging.getLogger(logger_name)
   if not logger.hasHandlers():
      logger.setLevel(logging.DEBUG)
      logger.addHandler(get_console_handler())
      logger.propagate = False
      logger.gpu_usage = print_gpu_info
   return logger
