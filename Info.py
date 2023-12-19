import os
from config_parser import Parser

class Info:
    def __init__(self):
        config = Parser()
        config.get_args()
        self.conf = config