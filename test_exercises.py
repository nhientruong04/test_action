# import pytest
import numpy as np
import pandas as pd

from exercises import exercise_1, exercise_2, exercise_3, exercise_4, exercise_5

def test_exercise_1():
    assert exercise_1(2, 3) == 5  # Example test case

def test_exercise_2():
    assert exercise_2("hello") == "HELLO"  # Modify accordingly

def test_exercise_3():
    assert exercise_3([1, 2, 3]) == [3, 2, 1]  # Example case

def test_exercise_4():
    data = pd.DataFrame({
        "Col1" : [1, 2, 3],
        "Col2" : [3, 2, 1]
    })
    res = pd.DataFrame({
        "Col1" : [6],
        "Col2" : [6]
    })
    assert exercise_4(data).equals(res) # Modify accordingly

def test_exercise_5():
    res = exercise_5([1, 2, 3])
    assert isinstance(res, np.ndarray)
    assert (res==np.ndarray([1,2,3])).all() # Modify accordingly
