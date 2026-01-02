from Classification.Parameter.Parameter import Parameter

class IREPParameter(Parameter):
    """
    Parameter class for the IREP Algorithm.
    It manages settings like pruning ratio and minimum coverage.
    """

    def __init__(self, seed: int = None, pruning_ratio: float = 0.66, min_coverage: int = 1):
        """
        Constructor for IREP parameters.
        :param seed: Random seed for reproducible results.
        :param pruning_ratio: Percentage of data used for growing rules (default 0.66).
        :param min_coverage: Minimum number of instances a rule must cover to be valid.
        """
        super().__init__(seed)
        self.pruning_ratio = pruning_ratio
        self.min_coverage = min_coverage

    def getPruningRatio(self) -> float:
        return self.pruning_ratio
        
    def getMinCoverage(self) -> int:
        return self.min_coverage