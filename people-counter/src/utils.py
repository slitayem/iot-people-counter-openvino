"""
Helper functions module
@author: Saloua Litayem
"""

import logging
import re
import os
from time import time
import json
import yaml

import paho.mqtt.client as mqtt

import logger

def connect_mqtt(host, port, keepalive_interval=60):
    """ Connect to the MQTT client
    """
    client = None
    try:
        client = mqtt.Client()
        client.connect(host, port, keepalive_interval)
    except Exception as exp:
        logging.error("Error while connecting to the MQTT server")
        logger.log_exception(exp)

    return client


def human_readable_size(size, decimal_places=2):
    """Convert size in bytes to a human readable value"""
    for unit in ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB']:
        if size < 1024.0 or unit == 'PiB':
            break
        size /= 1024.0
    return f"{size:.{decimal_places}f} {unit}"


def publish_messages(mqtt_client, topics_messages):
    """Publish messages to the provided topics
    :param mqtt_client: MQTT client object
    :param topics_messages: Dictionary of topics messages
    mappping
    """
    for topic, message in topics_messages.items():
        mqtt_client.publish(topic, json.dumps(message))

def timer(function):
    """Decorator for timing functions"""
    def wrapper(*args, **kwargs):
        start = time()
        function(*args, **kwargs)
        end = time()
        logging.debug(f'It took {(end-start):.3f} seconds')

    return wrapper


def constructor_env_variables(loader, node):
    """
    Extracts the environment variable from the node's value
    :param yaml.Loader loader: the yaml loader
    :param node: the current node in the yaml
    :return: the parsed string that contains the value of the environment
    variable
    """
    pattern = re.compile(r'.*?\${(\w+)}.*?')
    value = loader.construct_scalar(node)
    match = pattern.findall(value)  # to find all env variables in line
    if match:
        full_value = value
        for g in match:
            full_value = full_value.replace(
                f'${{{g}}}', os.environ.get(g, g)
            )
        return full_value
    return value

def parse_yaml_config(path=None, data=None, tag='!ENV'):
    """
    Load a yaml configuration file and resolve any environment variables
    The environment variables must have !ENV before them and be in this format
    to be parsed: ${VAR_NAME}.
    https://dev.to/mkaranasou/python-yaml-configuration-with-environment-variables-parsing-2ha6
    E.g.:
    database:
        host: !ENV ${HOST}
        port: !ENV ${PORT}
    app:
        log_path: !ENV '/var/${LOG_PATH}'
        something_else: !ENV '${AWESOME_ENV_VAR}/var/${A_SECOND_AWESOME_VAR}'
    :param str path: the path to the yaml file
    :param str data: the yaml data itself as a stream
    :param str tag: the tag to look for
    :return: the dict configuration
    :rtype: dict[str, T]
    """
    # pattern for global vars: look for ${word}
    pattern = re.compile(r'.*?\${(\w+)}.*?')
    loader = yaml.SafeLoader

    # the tag will be used to mark where to start searching for the pattern
    # e.g. somekey: !ENV somestring${MYENVVAR}blah blah blah
    loader.add_implicit_resolver(tag, pattern, None)

    loader.add_constructor(tag, constructor_env_variables)

    if path:
        with open(path) as conf_data:
            return yaml.load(conf_data, Loader=loader)
    elif data:
        return yaml.load(data, Loader=loader)
    else:
        raise ValueError('Either a path or data should be defined as input')

def get_script_folder():
    """ Get the folder path of the module
    calling the function
    """
    return os.path.dirname(os.path.abspath(__file__))