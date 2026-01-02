import unittest
import os

from Classification.Parameter.IREPParameter import IREPParameter
from Classification.IREP.IREPModel import IREPModel
from test.Classifier.ClassifierTest import ClassifierTest
from Classification.Attribute.AttributeType import AttributeType
from Classification.DataSet.DataDefinition import DataDefinition
from Classification.DataSet.DataSet import DataSet

class IREPTest(ClassifierTest):
    """
    Test suite for the IREP Algorithm.
    Checks performance on various datasets and verifies save/load functionality.
    """

    def setUp(self) -> None:
        # Load datasets with correct attribute definitions
        
        # 1. Iris (Continuous)
        attributeTypes = 4 * [AttributeType.CONTINUOUS]
        dataDefinition = DataDefinition(attributeTypes)
        self.iris = DataSet(dataDefinition, ",", "datasets/iris.data")
        
        # 2. Bupa (Continuous)
        attributeTypes = 6 * [AttributeType.CONTINUOUS]
        dataDefinition = DataDefinition(attributeTypes)
        self.bupa = DataSet(dataDefinition, ",", "datasets/bupa.data")

        # 3. Dermatology (Continuous)
        attributeTypes = 34 * [AttributeType.CONTINUOUS]
        dataDefinition = DataDefinition(attributeTypes)
        self.dermatology = DataSet(dataDefinition, ",", "datasets/dermatology.data")

        # 4. Car (Categorical/Discrete)
        attributeTypes = 6 * [AttributeType.DISCRETE]
        dataDefinition = DataDefinition(attributeTypes)
        self.car = DataSet(dataDefinition, ",", "datasets/car.data")

        # 5. TicTacToe (Categorical/Discrete)
        attributeTypes = 9 * [AttributeType.DISCRETE]
        dataDefinition = DataDefinition(attributeTypes)
        self.tictactoe = DataSet(dataDefinition, ",", "datasets/tictactoe.data")

        # Ensure directory exists for model saving tests
        if not os.path.exists("models"):
            os.makedirs("models")

    def test_Train(self):
        print("\n--- IREPModel Performance Tests ---")
        
        # 1. Iris
        # Using default parameters
        irep = IREPModel()
        irep.train(self.iris.getInstanceList()) 
        error = 100 * irep.test(self.iris.getInstanceList()).getErrorRate()
        print(f"Iris Error       : %{error:.2f}")
        self.assertLess(error, 50.0)

        # 2. Bupa
        # Using min_coverage=5 to reduce overfitting on noisy data
        params_bupa = IREPParameter(min_coverage=5)
        irep.train(self.bupa.getInstanceList(), params_bupa)
        error = 100 * irep.test(self.bupa.getInstanceList()).getErrorRate()
        print(f"Bupa Error       : %{error:.2f}")

        # 3. Dermatology
        irep.train(self.dermatology.getInstanceList())
        error = 100 * irep.test(self.dermatology.getInstanceList()).getErrorRate()
        print(f"Dermatology Error: %{error:.2f}")

        # 4. Car
        # Using higher pruning ratio for better generalization
        params_car = IREPParameter(pruning_ratio=0.80)
        irep.train(self.car.getInstanceList(), params_car)
        error = 100 * irep.test(self.car.getInstanceList()).getErrorRate()
        print(f"Car Error        : %{error:.2f}")
        
        # 5. TicTacToe
        irep.train(self.tictactoe.getInstanceList())
        error = 100 * irep.test(self.tictactoe.getInstanceList()).getErrorRate()
        print(f"TicTacToe Error  : %{error:.2f}")

    def test_Load(self):
        print("\n--- IREPModel Save/Load Test ---")
        filename = "models/irep-iris-test.txt"
        
        # Train and save the model
        model_original = IREPModel()
        model_original.train(self.iris.getInstanceList())
        model_original.saveModel(filename)
        error_original = 100 * model_original.test(self.iris.getInstanceList()).getErrorRate()

        # Load the model back
        model_loaded = IREPModel()
        model_loaded.loadModel(filename)
        error_loaded = 100 * model_loaded.test(self.iris.getInstanceList()).getErrorRate()
        
        # Compare results
        self.assertEqual(error_original, error_loaded, "Model load failed: Results do not match!")
        print("Save/Load Test Passed: OK")

if __name__ == '__main__':
    unittest.main()