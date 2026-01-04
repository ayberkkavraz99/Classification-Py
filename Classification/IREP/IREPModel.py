import math
import random

from Classification.Attribute.ContinuousAttribute import ContinuousAttribute
from Classification.Attribute.DiscreteAttribute import DiscreteAttribute
from Classification.Model.Model import Model
from Classification.Parameter.IREPParameter import IREPParameter
from .Rule import Rule
from .Condition import Condition

class IREPModel(Model):
    """
    IREP (Incremental Reduced Error Pruning) Algorithm Implementation.
    Supports multiclass classification, continuous/discrete attributes, and model saving.
    """

    def __init__(self):
        super().__init__()
        self.rules = []
        self.default_class = None

    def train(self, trainSet, parameters=None):
        """
        Trains the IREP model using the given dataset.
        """
        # 1. Handle Parameters
        min_coverage = 1
        if parameters is None:
            pruning_ratio = 0.66
        else:
            pruning_ratio = parameters.getPruningRatio()
            min_coverage = parameters.getMinCoverage()
            if parameters.seed is not None:
                random.seed(parameters.seed)

        # 2. Sort Classes by Frequency
        # We learn rare classes first to handle imbalanced datasets better.
        instances = trainSet.getInstances()[:] 
        
        counts = {}
        for inst in instances:
            lbl = inst.getClassLabel()
            counts[lbl] = counts.get(lbl, 0) + 1
            
        distinct_classes = list(counts.keys())
        # Sort from least frequent to most frequent
        classes = sorted(distinct_classes, key=lambda c: counts[c])
        
        if len(classes) < 2:
            if len(classes) == 1:
                self.default_class = classes[0]
            return

        random.shuffle(instances)
        self.rules = []

        # 3. Multiclass Training Loop (One-vs-Rest)
        # Iterate through classes, treating the current one as Positive and others as Negative.
        for i in range(len(classes) - 1):
            pos_class = classes[i]
            
            while self.has_positive_instances(instances, pos_class):
                # Split data into growing and pruning sets
                split_idx = int(len(instances) * pruning_ratio)
                grow_set = instances[:split_idx]
                prune_set = instances[split_idx:]
                
                if len(prune_set) == 0: 
                    prune_set = grow_set

                # Grow a rule using the growing set
                rule = self.grow_rule(grow_set, trainSet, pos_class, min_coverage)
                
                # Prune the rule using the pruning set
                self.prune_rule(rule, prune_set, pos_class)

                # Check stopping criteria
                rule_acc = self.calculate_pruning_metric(rule, prune_set, pos_class)
                empty_acc = self.calculate_empty_clause_metric(prune_set, pos_class)

                if rule.get_length() == 0 or rule_acc <= empty_acc:
                    break
                
                self.rules.append(rule)
                
                # Remove instances covered by the new rule
                instances = [inst for inst in instances if not rule.covers(inst)]
        
        # The last remaining class becomes the default class
        self.default_class = classes[-1]

    def predict(self, instance):
        """ Predicts the class label for a new instance. """
        for rule in self.rules:
            if rule.covers(instance):
                return rule.class_label
        return self.default_class

    def saveModel(self, fileName):
        """ Saves the model rules to a file. """
        try:
            with open(fileName, "w", encoding="utf-8") as file:
                file.write(f"DefaultClass:{self.default_class}\n")
                for rule in self.rules:
                    file.write(str(rule) + "\n")
        except Exception as e:
            print(f"Error saving model: {e}")

    def loadModel(self, fileName):
        """ Loads the model rules from a file. """
        self.rules = []
        try:
            with open(fileName, "r", encoding="utf-8") as file:
                lines = file.readlines()
                for line in lines:
                    line = line.strip()
                    if not line: continue
                    
                    if line.startswith("DefaultClass:"):
                        self.default_class = line.split(":")[1]
                    elif line.startswith("IF "):
                        self.parse_and_add_rule(line)
        except Exception as e:
            print(f"Error loading model: {e}")

    def parse_and_add_rule(self, line):
        """ Helper to parse a rule string from the file. """
        try:
            parts = line.split(" THEN ")
            class_label = parts[1]
            condition_part = parts[0][3:] 

            rule = Rule(class_label)
            conditions = condition_part.split(" AND ")
            for cond_str in conditions:
                terms = cond_str.split(" ")
                att_index = int(terms[0].split("-")[1])
                operator = terms[1]
                value_str = terms[2]

                try:
                    value = float(value_str)
                except ValueError:
                    value = value_str 

                rule.add_condition(Condition(att_index, operator, value))
            
            self.rules.append(rule)
        except:
            pass

    # --- Helper Methods ---

    def grow_rule(self, instances, trainSet, pos_class, min_coverage=1):
        """ Grows a rule by adding conditions that maximize Foil Gain. """
        rule = Rule(pos_class)
        current_instances = instances[:]
        
        if len(instances) > 0:
            num_attributes = instances[0].attributeSize()
            sample_instance = instances[0]
        elif trainSet.size() > 0:
            num_attributes = trainSet.get(0).attributeSize()
            sample_instance = trainSet.get(0)
        else:
            return rule

        while True:
            # Stop if no negative instances are left
            negatives = [inst for inst in current_instances if inst.getClassLabel() != pos_class]
            if len(negatives) == 0:
                break

            best_cond = None
            best_gain = -float('inf')

            for i in range(num_attributes):
                attribute_obj = sample_instance.getAttribute(i)
                possible_values = self.get_distinct_values(current_instances, i)
                
                # Handle Continuous Attributes (Numbers)
                if isinstance(attribute_obj, ContinuousAttribute):
                    for value in possible_values:
                        for op in ["<=", ">"]:
                            cond = Condition(i, op, value)
                            gain = self.calculate_foil_gain(current_instances, cond, pos_class, min_coverage)
                            if gain > best_gain:
                                best_gain = gain
                                best_cond = cond
                # Handle Discrete Attributes (Categories)
                else:
                    for value in possible_values:
                        cond = Condition(i, "==", value)
                        gain = self.calculate_foil_gain(current_instances, cond, pos_class, min_coverage)
                        if gain > best_gain:
                            best_gain = gain
                            best_cond = cond

            if best_cond is None or best_gain <= 0:
                break

            rule.add_condition(best_cond)
            current_instances = [inst for inst in current_instances if best_cond.satisfies(inst)]
            
        return rule

    def prune_rule(self, rule, prune_set, pos_class):
        """ Prunes the rule to avoid overfitting using the pruning metric. """
        best_metric = self.calculate_pruning_metric(rule, prune_set, pos_class)
        while rule.get_length() > 0:
            best_temp_rule = None
            improved = False
            for i in range(rule.get_length()):
                temp_rule = rule.copy()
                temp_rule.remove_condition(i)
                metric = self.calculate_pruning_metric(temp_rule, prune_set, pos_class)
                if metric > best_metric:
                    best_metric = metric
                    best_temp_rule = temp_rule
                    improved = True
            if improved:
                rule.conditions = best_temp_rule.conditions
                best_metric = best_metric
            else:
                break

    def calculate_pruning_metric(self, rule, instances, pos_class):
        """ Calculates accuracy metric: (p + (N - n)) / (P + N). """
        if len(instances) == 0: return 0.0
        P = sum(1 for inst in instances if inst.getClassLabel() == pos_class)
        N = len(instances) - P
        covered = [inst for inst in instances if rule.covers(inst)]
        p = sum(1 for inst in covered if inst.getClassLabel() == pos_class)
        n = sum(1 for inst in covered if inst.getClassLabel() != pos_class)
        return (p + (N - n)) / (P + N)

    def calculate_empty_clause_metric(self, instances, pos_class):
        """ Calculates accuracy if no rule existed. """
        if len(instances) == 0: return 0.0
        P = sum(1 for inst in instances if inst.getClassLabel() == pos_class)
        N = len(instances) - P
        return N / (P + N)

    def calculate_foil_gain(self, instances, cond, pos_class, min_coverage=1):
        """ Calculates Foil Gain to select the best condition. """
        pos_before = sum(1 for inst in instances if inst.getClassLabel() == pos_class)
        neg_before = len(instances) - pos_before
        
        subset = [inst for inst in instances if cond.satisfies(inst)]
        
        # Check minimum coverage to prevent overfitting
        if len(subset) < min_coverage:
            return -float('inf')
            
        pos_after = sum(1 for inst in subset if inst.getClassLabel() == pos_class)
        neg_after = len(subset) - pos_after

        if pos_after == 0: return -float('inf')
        
        info_before = -math.log2(pos_before / (pos_before + neg_before + 1e-9))
        info_after = -math.log2(pos_after / (pos_after + neg_after + 1e-9))
        
        return pos_after * (info_before - info_after)

    def has_positive_instances(self, instances, pos_class):
        for inst in instances:
            if inst.getClassLabel() == pos_class: return True
        return False

    def get_distinct_values(self, instances, attribute_index):
        values = set()
        for inst in instances:
            values.add(inst.getAttribute(attribute_index).getValue())
        return sorted(list(values))