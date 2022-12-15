"""TB-Net configuration for examples."""

import json


class TBNetConfig:
    """
    TBNet config file parser and holder.

    Args:
        config_path (str): json config file path.
    """

    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            json_dict = json.load(f)
        self.num_items = int(json_dict['num_items'])
        self.num_relations = int(json_dict['num_relations'])
        self.num_references = int(json_dict['num_references'])
        self.per_item_paths = int(json_dict['per_item_paths'])
        self.embedding_dim = int(json_dict['embedding_dim'])
        self.batch_size = int(json_dict['batch_size'])
        self.lr = float(json_dict['lr'])
        self.kge_weight = float(json_dict['kge_weight'])
        self.node_weight = float(json_dict['node_weight'])
        self.l2_weight = float(json_dict['l2_weight'])

    def save(self, config_path):
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
