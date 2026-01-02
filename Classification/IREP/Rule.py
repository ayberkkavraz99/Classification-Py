from .Condition import Condition

class Rule:
    """
    Represents a classification rule consisting of multiple conditions.
    Example: IF (Age > 18) AND (Income > 5000) THEN Class = Approved.
    """
    
    # Use __slots__ to optimize memory usage.
    __slots__ = ['class_label', 'conditions']

    def __init__(self, class_label):
        self.class_label = class_label
        self.conditions = []

    def add_condition(self, condition):
        """ Adds a new condition to the rule. """
        self.conditions.append(condition)

    def remove_condition(self, index):
        """ Removes a condition at a specific index. """
        if 0 <= index < len(self.conditions):
            self.conditions.pop(index)

    def covers(self, instance):
        """ Checks if the rule covers (applies to) the given instance. """
        for condition in self.conditions:
            if not condition.satisfies(instance):
                return False
        return True

    def get_length(self):
        return len(self.conditions)
    
    def copy(self):
        """ Creates a copy of the rule. """
        new_rule = Rule(self.class_label)
        new_rule.conditions = self.conditions[:] 
        return new_rule

    def __str__(self):
        condition_str = " AND ".join([str(c) for c in self.conditions])
        return f"IF {condition_str} THEN {self.class_label}"
    
    def __repr__(self):
        return self.__str__()