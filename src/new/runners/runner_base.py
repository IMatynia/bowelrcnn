from argparse import ArgumentParser, Namespace


class RunnerBase:
    def __init__(self, args: Namespace) -> None:
        self.args = args

    @classmethod
    def add_arguments(self, parser: ArgumentParser):
        NotImplemented

    def run(self):
        NotImplemented
