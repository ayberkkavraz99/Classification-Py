from Classification.Instance.Instance import Instance
from Classification.Attribute.Attribute import Attribute

class Condition:
    """
    Represents a single condition in a rule (e.g., Age > 18).
    """
    
    # Use __slots__ to save memory because we create many conditions.
    __slots__ = ['attribute_index', 'operator', 'value']

    def __init__(self, attribute_index: int, operator: str, value):
        self.attribute_index = attribute_index
        self.operator = operator
        self.value = value

    def satisfies(self, instance: Instance) -> bool:
        """
        Checks if the instance meets this condition.
        """
        # Get the attribute value from the instance
        attribute_obj: Attribute = instance.getAttribute(self.attribute_index)
        val = attribute_obj.getValue()

        # Compare based on the operator
        if self.operator == "<=":
            return val <= self.value
        elif self.operator == ">":
            return val > self.value
        elif self.operator == "==":
            return val == self.value
        else:
            return False

    def __str__(self):
        return f"Att-{self.attribute_index} {self.operator} {self.value}"

    def __repr__(self):
        return self.__str__()