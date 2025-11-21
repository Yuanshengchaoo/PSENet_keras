import json
import os

from utils.bunch import Bunch


def get_config_from_json(json_file):
    """
    Get the config from a json file
    :param json_file:
    :return: config(namespace), config(dictionary)
    """
    # parse the configurations from the config json file provided
    print("loading json file ... ...")
    print(json_file)
    with open(json_file, 'r') as config_file:
        print("json opened")
        config_dict = json.load(config_file)
        print("json loaded")

    # convert the dictionary to a namespace using bunch lib
    config = Bunch(config_dict)
    config.config_path = os.path.abspath(json_file)
    config.config_dir = os.path.dirname(config.config_path)

    return config, config_dict


def process_config(json_file):

    # 在参数信息的bunch中加入summary dir和checkpoint dir的信息
    config, _ = get_config_from_json(json_file)
    # config.summary_dir = os.path.join("..\\experiments", config.exp_name, "summary")
    # config.checkpoint_dir = os.path.join("..\\experiments", config.exp_name, "checkpoints")
    return config
