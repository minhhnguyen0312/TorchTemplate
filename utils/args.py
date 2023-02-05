import os
import sys
import argparse
import yaml

class Parser3rdParty:
    def __init__(self):
        self.ARGS = {}

    def add_args(self, args):
        self.ARGS.update(args)
    
    def parse_args(self):
        idx = sys.argv.index("--")
        argv = sys.argv[idx+1:]
        for i, (arg, t) in enumerate(self.args.items()):
            try:
                self.args[arg] = t(argv[i])
            except IndexError:
                raise IndexError(f"Required argument {arg} not found!")
            except ValueError:
                raise ValueError(f"Required argument '{arg}' is a {t}, got {type(argv[i])}.")
        return self.args

class PythonParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
    
    def parse_args(self):
        self.parser.parse_args()
        return self.parser

class YmlParser:
    def __init__(self):
        pass

    def read(self, filename):
        with open(filename, 'r') as f:
            args = yaml.safe_load(f)
        return args