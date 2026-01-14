
config_instance = None


class Config:

    champollion_data_root_dir = "/neurospin/dico/data/deep_folding/current"

    def get_champollion_data_root_dir(self):
        """ get directory of model / regions data.

        For now it's hard-coded. A config system should be done for this
        (amongst others), but at least we centralize it here.
        """
        return self.champollion_data_root_dir


def config():
    """ get the global unique instance of the Config
    """
    global config_instance
    if config_instance is None:
        config_instance = Config()
    return config_instance
