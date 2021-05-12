from levels.zelda_action_interpreter import interpret_action
import unittest


class TestZeldaActionInterpreter(unittest.TestCase):

    def test_interpret_action(self):
        test_cases = {
            0: [0, 0],
            1: [0, 1],
            2: [0, 2],
            3: [0, 3],
            4: [0, 4],
            5: [1, 0],
            6: [1, 1],
            7: [0, 0],
            -1: [0, 0],
            "cat": [0, 0]
        }
        for key, val in test_cases.items():
            res = interpret_action(key)
            assert res == val, f"input {key} yielded {res}, but expected {val}"
