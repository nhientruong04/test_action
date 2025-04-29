#!/usr/bin/env python
# coding: utf-8

import torch
import re

def clean_exercise_file(input_path="exercises.py", output_path="exercises_cleaned.py"):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    patterns = [
        re.compile(r"^\s*ex\d+_sol\s*.*$"),  # ex1_sol = exercise_1(...)
        re.compile(r"^\s*assert.*$"),        # asserts
        re.compile(r"^\s*exercise_\d+(.*)\s*")
    ]

    cleaned_lines = [line for line in lines if not any(p.match(line) for p in patterns)]

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

    print(f"Cleaned file written to: {output_path}")


clean_exercise_file()


# --- Import student functions from your exercises module ---
from exercises_cleaned import (
    exercise_1, exercise_2, exercise_3, exercise_4
)

# --- Expected Solutions ---

# Q1
EXPECTED_Q1 = torch.tensor(7.0)

# Q2
Q2_INPUT_STUD = torch.randint(0, 100, (5, ))
Q2_INPUT_SOL = Q2_INPUT_STUD.clone()

torch.manual_seed(0)
weight = torch.randn([5], requires_grad=True)
bias = torch.tensor(5.0, requires_grad=True)
y = (Q2_INPUT_SOL * weight).sum() + bias
y.backward()
EXPECTED_Q2  = (y, weight.grad)

# Q3
Q3_INPUT_STUD = torch.randint(0, 100, (5, ))
Q3_INPUT_SOL = Q3_INPUT_STUD.clone()

torch.manual_seed(0)
weight = torch.randn([5], requires_grad=True)
bias = torch.arange(1,6, dtype=torch.float32)
y = (Q3_INPUT_SOL * weight) + bias
y.backward(torch.ones(5))
EXPECTED_Q3 = (y, weight.grad)

# Q4
Q4_DATA_STUD = torch.randint(1, 100, (10, )).float()
Q4_DATA_SOL = Q4_DATA_STUD.clone()
Q4_TARGET = torch.randint(30, 70, (1,)).float()

torch.manual_seed(0)
model_weight = torch.randn(10, requires_grad=True)
torch.manual_seed(0)
model_bias = torch.randn(1, requires_grad=True)
output = (Q4_DATA_SOL * model_weight).sum() + model_bias
loss = (Q4_TARGET- output) ** 2
loss.backward()
EXPECTED_Q4 = (model_weight.grad, model_bias.grad)

# --- Pytest Test Functions ---

def test_exercise_1():
    sol = exercise_1()
    assert torch.equal(sol, EXPECTED_Q1)

def test_exercise_2():
    sol = exercise_2(Q2_INPUT_STUD)
    assert torch.all((sol[0] - EXPECTED_Q2[0] < 1e-4)).item()
    assert torch.equal(sol[1], EXPECTED_Q2[1])


def test_exercise_3():
    sol = exercise_3(Q3_INPUT_STUD)

    assert torch.all((sol[0] - EXPECTED_Q3[0]) < 5e-4).item()
    assert torch.equal(sol[1], EXPECTED_Q3[1])

def test_exercise_4():
    sol = exercise_4(data=Q4_DATA_STUD, target=Q4_TARGET)
    assert torch.all((sol[0] - EXPECTED_Q4[0]) < 5e-4)
    assert torch.all((sol[1] - EXPECTED_Q4[1]) < 5e-4)