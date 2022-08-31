#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Logging module
@author: Saloua Litayem
"""
import os
import traceback
import logging
import logging.config

import utils


class Logger:
    """
    Set up logging
    """
    def __init__(self):
        self.config_path = os.path.join(
            utils.get_script_folder(), "logging.yml")

    def setup_logging(self):
        """
        setup logging
        """
        try:
            config_ = utils.parse_yaml_config(path=self.config_path)
            logging.config.dictConfig(config_)
            logging.debug(f"Logging Config path {self.config_path}")
            # logging.getLogger().setLevel(default_level)
        except Exception:
            logging.exception(
                "Error while loading logging configuration. Using default configs")
            logging.basicConfig(level=logging.INFO)

def log_exception(exp):
    """
    Log exceptions
    :param exp: Exception object
    """
    logging.error(
        f"Exception: {exp.__class__} ({exp.__doc__}): {traceback.format_exc()}")
