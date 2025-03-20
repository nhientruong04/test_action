#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
import pytest

# Set up random generators using an integer seed.
random.seed(42)
np.random.seed(42)

# --- Import student functions from your exercises module ---
from exercises import (
    question_1, question_2, question_3, question_4, question_5,
    question_6, question_7, question_8, question_9, question_10,
    question_11, question_12, question_13, question_14, question_15,
    question_16, question_17, question_18, question_19, question_20
)

# --- Optional: Fixture for a random generator using only integers (if needed) ---
@pytest.fixture
def int_random():
    # Returns a NumPy random generator seeded with an integer.
    rng = np.random.default_rng(42)
    return rng

# --- Expected Solutions ---

# Q1: One-dimensional array from 5 to 905 counting by 10.
EXPECTED_Q1 = np.arange(5, 905, 10)

# Q2: Same array via list comprehension.
EXPECTED_Q2 = np.array([i for i in range(5, 905, 10)])

# Q3: Array of capital letters A-Z.
EXPECTED_Q3 = np.array([chr(i) for i in range(ord('A'), ord('Z') + 1)])

# Q4: Tuple: ten zeros and six ones.
EXPECTED_Q4_1 = np.zeros(10)
EXPECTED_Q4_2 = np.ones(6)

# Q5: Extract third column from 2D array.
input_q5 = np.random.randint(10, 100, size=(3, 3))
EXPECTED_Q5 = input_q5[:, 2]

# Q6: Array of shape (3,4) filled with 64-bit integer zeros.
EXPECTED_Q6 = np.zeros((3, 4), dtype=np.int64)

# Q7: Split array into 3 equal sub-arrays with randomized integers.
arr_q7 = np.random.randint(10, 100, size=(9, 3))
EXPECTED_Q7 = np.split(arr_q7, 3)

# Q8: Return last two elements of all but the last row of a (3,4) array.
input_q8 = np.random.randint(1, 100, size=(3, 4))
EXPECTED_Q8 = input_q8[:-1, -2:]

# Q9: Count elements greater than 5.
input_q9 = np.random.randint(1, 20, size=(3, 4))
EXPECTED_Q9 = (input_q9 > 5).sum()

# Q10: Delete second column and insert a new column.
input_q10 = np.random.randint(10, 100, size=(3, 3))
insert_data = np.random.randint(10, 100, size=(1, 3))
temp_q10 = np.delete(input_q10, 1, axis=1)
EXPECTED_Q10 = np.insert(temp_q10, 1, insert_data, axis=1)

# Q11: Compute Euclidean distances between consecutive points and append as a new column.
input_q11 = np.random.randint(1, 20, size=(4, 2))
diffs = np.diff(input_q11, axis=0)
distance = np.sqrt(np.sum(diffs ** 2, axis=1))
distance = np.append(distance, np.nan)  # Pad the last row with NaN.
EXPECTED_Q11 = np.column_stack((input_q11, distance))

# Q12: Remove consecutive duplicate rows.
input_q12 = np.random.randint(1, 5, size=(6, 2))
mask = np.insert(np.any(input_q12[1:] != input_q12[:-1], axis=1), 0, True)
EXPECTED_Q12 = input_q12[mask]

# Q13: Normalize a 2D array (avoid division by zero).
input_q13 = np.random.randint(10, 50, size=(4, 3))
EXPECTED_Q13 = (input_q13 - np.mean(input_q13, axis=0)) / np.std(input_q13, axis=0)

# Q14: Delete second column and insert a new column with row sums.
input_q14 = np.random.randint(1, 10, size=(3, 3))
row_sums = np.sum(input_q14, axis=1)
temp_q14 = np.delete(input_q14, 1, axis=1)
EXPECTED_Q14 = np.insert(temp_q14, 1, row_sums, axis=1)

# Q15: Extract unique characters from a NumPy array of strings.
input_q15 = np.array(["python", "data", "science", "rocks"])
EXPECTED_Q15 = set(sorted("".join(input_q15)))

# Q16: Map unique characters to unique indices sorted by ASCII.
input_q16 = np.array([chr(i) for i in np.random.randint(97, 123, size=10)])
EXPECTED_Q16 = {k: v for v, k in enumerate(sorted(set(input_q16)))}

# Q17: Stack a list of 2D traces into a single array.
input_q17 = [np.random.randint(1, 10, size=(2, 2)), np.random.randint(1, 10, size=(3, 2))]
EXPECTED_Q17 = np.vstack(input_q17)

# Q18: Convert a list of text labels into an array of integer encodings.
vocab = {'apple': 0, 'banana': 1, 'cherry': 2, 'date': 3, 'elderberry': 4, 'fig': 5, 'grape': 6}
labels = "apple cherry grape"
EXPECTED_Q18 = np.array([vocab[label] for label in labels.split()])

# Q19: Extract non-zero differences from an array.
input_q19 = np.array([[3, 4], [0, 0], [-2, 1]])
mask_q19 = np.all(input_q19 == 0, axis=1)
EXPECTED_Q19 = input_q19[~mask_q19]

# Q20: Time Series Data Transformation and Feature Engineering.
input_q20 = np.random.uniform(1.0, 100.0, size=(5, 3))
input_q20[2][2] = 10000.0
mean_data = np.mean(input_q20, axis=0)
std_data = np.std(input_q20, axis=0)
std_data[std_data == 0] = 1
normalized_data = (input_q20 - mean_data) / std_data
anomalies = np.abs(normalized_data) > 2.5
data_with_nan = input_q20.copy()
data_with_nan[anomalies] = np.nan
interpolated_data = np.where(np.isnan(data_with_nan),
                             np.nanmean(data_with_nan, axis=0),
                             data_with_nan)
first_order_diff = np.diff(interpolated_data, axis=0)
first_order_diff_padded = np.vstack((first_order_diff, np.zeros((1, first_order_diff.shape[1]))))
missing_flag = np.isnan(data_with_nan).astype(int)
EXPECTED_Q20 = np.hstack((interpolated_data, first_order_diff_padded, missing_flag))


# --- Pytest Test Functions ---

@pytest.mark.xfail(raises=NotImplementedError, reason="Feature not implemented yet")
def test_question_1():
    sol = question_1()
    np.testing.assert_array_equal(sol, EXPECTED_Q1)

@pytest.mark.xfail(raises=NotImplementedError, reason="Feature not implemented yet")
def test_question_2():
    sol = question_2()
    np.testing.assert_array_equal(sol, EXPECTED_Q2)

@pytest.mark.xfail(raises=NotImplementedError, reason="Feature not implemented yet")
def test_question_3():
    sol = question_3()
    np.testing.assert_array_equal(sol, EXPECTED_Q3)

@pytest.mark.xfail(raises=NotImplementedError, reason="Feature not implemented yet")
def test_question_4():
    sol1, sol2 = question_4()
    np.testing.assert_array_equal(sol1, EXPECTED_Q4_1)
    np.testing.assert_array_equal(sol2, EXPECTED_Q4_2)

@pytest.mark.xfail(raises=NotImplementedError, reason="Feature not implemented yet")
def test_question_5():
    sol = question_5(input_q5)
    np.testing.assert_array_equal(sol, EXPECTED_Q5)

@pytest.mark.xfail(raises=NotImplementedError, reason="Feature not implemented yet")
def test_question_6():
    sol = question_6()
    np.testing.assert_array_equal(sol, EXPECTED_Q6)

@pytest.mark.xfail(raises=NotImplementedError, reason="Feature not implemented yet")
def test_question_7():
    sol = question_7(arr_q7)
    # Ensure the output is a list of three arrays.
    assert isinstance(sol, list)
    assert len(sol) == len(EXPECTED_Q7)
    for a, b in zip(sol, EXPECTED_Q7):
        np.testing.assert_array_equal(a, b)

@pytest.mark.xfail(raises=NotImplementedError, reason="Feature not implemented yet")
def test_question_8():
    sol = question_8(input_q8)
    np.testing.assert_array_equal(sol, EXPECTED_Q8)

@pytest.mark.xfail(raises=NotImplementedError, reason="Feature not implemented yet")
def test_question_9():
    sol = question_9(input_q9)
    assert sol == EXPECTED_Q9

@pytest.mark.xfail(raises=NotImplementedError, reason="Feature not implemented yet")
def test_question_10():
    sol = question_10(input_q10, insert_data)
    np.testing.assert_array_equal(sol, EXPECTED_Q10)

@pytest.mark.xfail(raises=NotImplementedError, reason="Feature not implemented yet")
def test_question_11():
    sol = question_11(input_q11)
    np.testing.assert_allclose(sol, EXPECTED_Q11, equal_nan=True)

@pytest.mark.xfail(raises=NotImplementedError, reason="Feature not implemented yet")
def test_question_12():
    sol = question_12(input_q12)
    np.testing.assert_array_equal(sol, EXPECTED_Q12)

@pytest.mark.xfail(raises=NotImplementedError, reason="Feature not implemented yet")
def test_question_13():
    sol = question_13(input_q13)
    np.testing.assert_allclose(sol, EXPECTED_Q13)

@pytest.mark.xfail(raises=NotImplementedError, reason="Feature not implemented yet")
def test_question_14():
    sol = question_14(input_q14)
    np.testing.assert_array_equal(sol, EXPECTED_Q14)

@pytest.mark.xfail(raises=NotImplementedError, reason="Feature not implemented yet")
def test_question_15():
    sol = question_15(input_q15)
    assert sol == EXPECTED_Q15

@pytest.mark.xfail(raises=NotImplementedError, reason="Feature not implemented yet")
def test_question_16():
    sol = question_16(input_q16)
    assert sol == EXPECTED_Q16

@pytest.mark.xfail(raises=NotImplementedError, reason="Feature not implemented yet")
def test_question_17():
    sol = question_17(input_q17)
    np.testing.assert_array_equal(sol, EXPECTED_Q17)

@pytest.mark.xfail(raises=NotImplementedError, reason="Feature not implemented yet")
def test_question_18():
    sol = question_18(vocab)
    np.testing.assert_array_equal(sol, EXPECTED_Q18)

@pytest.mark.xfail(raises=NotImplementedError, reason="Feature not implemented yet")
def test_question_19():
    sol = question_19(input_q19)
    np.testing.assert_array_equal(sol, EXPECTED_Q19)

@pytest.mark.xfail(raises=NotImplementedError, reason="Feature not implemented yet")
def test_question_20():
    sol = question_20(input_q20)
    np.testing.assert_allclose(sol, EXPECTED_Q20, equal_nan=True)
