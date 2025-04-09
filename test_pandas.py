#!/usr/bin/env python
# coding: utf-8

import random
import numpy as np
import pandas as pd
import pytest
import re

def clean_exercise_file(input_path="exercises.py", output_path="exercises_cleaned.py"):
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    patterns = [
        re.compile(r"^\s*ex\d+_sol\s*=\s*exercise_\d+\(.*?\)\s*$"),  # ex1_sol = exercise_1(...)
        re.compile(r"^\s*q\d+\.check\(\)\s*$"),                      # q1.check()
        re.compile(r"^\s*\w+\s*=\s*pd\.read_csv\(.*?\)\s*$"),        # df = pd.read_csv(...)
    ]

    cleaned_lines = [line for line in lines if not any(p.match(line) for p in patterns)]

    with open(output_path, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)

    print(f"Cleaned file written to: {output_path}")


clean_exercise_file()

# Set up random generators using an integer seed.
random.seed(42)
np.random.seed(42)

# --- Import student functions from your exercises module ---
from exercises_cleaned import (
    exercise_1, exercise_2, exercise_3, exercise_4, exercise_5,
    exercise_6, exercise_7, exercise_8, exercise_9, exercise_10,
    exercise_11, exercise_12, exercise_13, exercise_14, exercise_15,
    exercise_16, exercise_17, exercise_18, exercise_19, exercise_20
)

@pytest.fixture
def int_random():
    # Returns a NumPy random generator seeded with an integer.
    rng = np.random.default_rng(42)
    return rng

# --- Expected Solutions ---

# Q1: Form a DataFrame with values and column names given below. Use the variable names as column names.
Q1_A = np.random.randint(low=1, high=10, size=(10))
Q1_B = np.random.randint(low=10, high=15, size=(10))
EXPECTED_Q1 = pd.DataFrame({
        'A': Q1_A,
        'B': Q1_B
    })

# Q2: Form a DataFrame with the given Series. Use the variable names as column names.
Q2_SER1 = pd.Series(list('youandi'))
Q2_SER2 = pd.Series(range(7))
EXPECTED_Q2 = pd.DataFrame({'ser1': Q2_SER1, 'ser2': Q2_SER2})

# Q3: How to get the items not common to both series A and series B?
Q3_A = pd.Series(np.random.randint(low=50, high=70, size=(20)))
Q3_B = pd.Series(np.random.randint(low=40, high=60, size=(20)))
ser_u = pd.Series(np.union1d(Q3_A, Q3_B))  # union
ser_i = pd.Series(np.intersect1d(Q3_A, Q3_B))  # intersect
EXPECTED_Q3 = ser_u[~ser_u.isin(ser_i)]

dataset1 = pd.DataFrame({ 
    'ID': [10, np.nan, 20, 30, np.nan, 50, np.nan, 
           150, 200, np.nan, 75, 130], 
  
    'Sale': [10, 20, 14, 11, 90, np.nan, 
             55, 14, 28, 25, 75, 35], 
  
    'Date': ['2019-10-05', '2020-09-10', np.nan, 
             '2020-08-17', '2020-09-10', '2020-07-27', 
             '2020-09-10', np.nan, '2020-10-10', 
             '2020-06-27', '2020-08-17', '2020-04-25'], 
})

# Q4: Fill any NaN value with 0.
EXPECTED_Q4 = dataset1['Sale'].fillna(0)

# Q5: Fill any NaN value with the mean value of the column "Sale". Round the mean to the nearest integer.
EXPECTED_Q5 = dataset1['Sale'].fillna(round(dataset1['Sale'].mean())) 

# Q6: How many products are there which can be identified with "ID"?
EXPECTED_Q6 = dataset1['ID'].nunique()

dataset2 = pd.read_csv("dataset/dataset2.csv")

# Q7: How many years of crime data are collected in the dataset?
EXPECTED_Q7 = dataset2['Year'].nunique()

# Q8: Find the year with the highest number of property crimes.
EXPECTED_Q8 = dataset2.loc[dataset2['Property'].idxmax(), 'Year']

# Q9: Count elements greater than 5.
input_q9 = dataset2.copy()

input_q9['Year'] = pd.to_datetime(input_q9['Year'], format='%Y')
EXPECTED_Q9 = input_q9.set_index('Year', drop = True)

# Q10: Delete any column having "theft" in its name (case insensitive). Show the first 10 rows.
input_q10 = dataset2.copy()
input_q10 = input_q10.loc[:, ~input_q10.columns.str.contains('theft', case=False)]
EXPECTED_Q10 = input_q10.head(10)

# Q11: Find the year with the largest percentage increase in total crimes compared to the previous year.
input_q11 = dataset2.copy()
input_q11['Total_change'] = input_q11['Total'].pct_change()
EXPECTED_Q11 = input_q11.loc[input_q11['Total_change'].idxmax(), 'Year']

# Q12: Calculate the percentage change in robbery crimes from 1960 to 2014. Round it to the nearest 2 decimal places. 
input_q12 = dataset2.copy()
start = input_q12.loc[input_q12['Year'] == 1960, 'Robbery'].values[0]
end = input_q12.loc[input_q12['Year'] == 2014, 'Robbery'].values[0]
percentage_change = ((end - start) / start) * 100
EXPECTED_Q12 = round(percentage_change,2)

dataset3 = pd.read_csv('dataset/open-food-facts-sample.csv', dtype={"code": str}, index_col="code")

# Q13: How many unique products are available in the dataset?
EXPECTED_Q13 = dataset3['product_name'].nunique()

# Q14: Find the percentage of missing values in each column. Round your results to the nearest 2 decimal places.
EXPECTED_Q14 = ((dataset3.isnull().sum() / len(dataset3)) * 100).round(2)

# Q15: Identify the top 5 manufacturers based on the number of unique products.
EXPECTED_Q15 = dataset3.groupby(by="brands", dropna=True)["product_name"].nunique().sort_values(ascending=False).head(5)

# Q16: Find the most common food additives used in products across all countries.
EXPECTED_Q16 = dataset3['additives'].str.split(',').explode().value_counts().head(10)

# Q17: Calculate the average "saturated-fat_100g" of all products containing "palm oil" and all products that don't. Return a DataFrame with 2 columns "contains_palm_oil" and "no_palm_oil" listing your results. Round your result to the nearest 2 decimal places.
input_q17 = dataset3.copy()
palm_oil = input_q17[input_q17['ingredients_text'].str.contains('palm oil', na=False, case=False)]
no_palm_oil = input_q17[~input_q17['ingredients_text'].str.contains('palm oil', na=False, case=False)]

EXPECTED_Q17 = pd.DataFrame ({
    "contains_palm_oil": [palm_oil['saturated-fat_100g'].mean().round(2)],
    "no_palm_oil": [no_palm_oil['saturated-fat_100g'].mean().round(2)]
})

# Q18: Create a pivot table showing the average carbohydrate content ('carbohydrates_100g') for each food category across different countries.
EXPECTED_Q18 = dataset3.pivot_table(index='categories', columns='countries', values='carbohydrates_100g', aggfunc='mean')

# Q19: Compare the average sodium levels ('sodium_100g') between processed and non-processed foods. Processed foods are those with "categories" matching one or more in the list given below. Round your results to the nearest 3 decimal places.
processed_keywords = ['snack', 'processed', 'packaged']
input_q19 = dataset3.copy()
input_q19['is_processed'] = input_q19['categories'].str.contains('|'.join(processed_keywords), na=False, case=False)
# Compute the average sodium content for processed vs. non-processed foods
sodium_comparison = input_q19.groupby('is_processed')['sodium_100g'].mean().rename({True: 'Processed', False: 'Non-Processed'})

EXPECTED_Q19 = (sodium_comparison['Non-Processed'].round(3), sodium_comparison['Processed'].round(3))

# Q20: Extract the first ingredient from the 'ingredients_text' for each row. NaN values should be saved as 'Unknown'. Show the first 100 values.
input_q20 = dataset3.copy()

def get_first_ingredient(ingredient_list):
    if pd.isna(ingredient_list):  # Handle NaN values
        return 'Unknown'
    return ingredient_list.split(',')[0].strip()  # Split by comma and get first ingredient

input_q20['main_ingredient'] = input_q20['ingredients_text'].apply(get_first_ingredient)
EXPECTED_Q20 = input_q20.head(100)['main_ingredient']


# --- Pytest Test Functions ---

def test_exercise_1():
    sol = exercise_1(Q1_A, Q1_B)
    pd.testing.assert_frame_equal(sol, EXPECTED_Q1)

def test_exercise_2():
    sol = exercise_2(Q2_SER1, Q2_SER2)
    pd.testing.assert_frame_equal(sol, EXPECTED_Q2)

def test_exercise_3():
    sol = exercise_3(Q3_A, Q3_B)
    np.testing.assert_array_equal(np.sort(sol.values), np.sort(EXPECTED_Q3.values))

def test_exercise_4():
    sol = exercise_4(dataset1)
    pd.testing.assert_series_equal(sol, EXPECTED_Q4, check_names=False, check_exact=True)

def test_exercise_5():
    sol = exercise_5(dataset1)
    pd.testing.assert_series_equal(sol, EXPECTED_Q5, check_names=False, check_exact=True)

def test_exercise_6():
    sol = exercise_6(dataset1)
    assert sol == EXPECTED_Q6

def test_exercise_7():
    sol = exercise_7(dataset2)
    
    assert sol == EXPECTED_Q7

def test_exercise_8():
    sol = exercise_8(dataset2)
    np.testing.assert_array_equal(sol, EXPECTED_Q8)

def test_exercise_9():
    sol = exercise_9(dataset2)
    
    assert sol.index.name == 'Year', "The index column must be Year"
    assert sol.index.dtype == 'datetime64[ns]', "Not datetime64"
    np.testing.assert_array_equal(sol.index.values, EXPECTED_Q9.index.values)

def test_exercise_10():
    sol = exercise_10(dataset2)
    pd.testing.assert_frame_equal(sol, EXPECTED_Q10)

def test_exercise_11():
    sol = exercise_11(dataset2)
    
    assert sol == EXPECTED_Q11

def test_exercise_12():
    sol = exercise_12(dataset2)
    
    assert abs(sol-EXPECTED_Q12) <= 1e-2

def test_exercise_13():
    sol = exercise_13(dataset3)
    
    assert sol == EXPECTED_Q13

def test_exercise_14():
    sol = exercise_14(dataset3)

    pd.testing.assert_series_equal(sol, EXPECTED_Q14, check_names=False, atol=2e-2)

def test_exercise_15():
    sol = exercise_15(dataset3)

    pd.testing.assert_series_equal(sol, EXPECTED_Q15, check_names=False)

def test_exercise_16():
    sol = exercise_16(dataset3)
    
    pd.testing.assert_series_equal(sol, EXPECTED_Q16, check_names=False)

def test_exercise_17():
    sol = exercise_17(dataset3)
    pd.testing.assert_frame_equal(sol, EXPECTED_Q17)

def test_exercise_18():
    sol = exercise_18(dataset3)
    
    assert len(sol.columns) == len(EXPECTED_Q18.columns), "Column mismatch"
    assert sol.index.name == 'categories', "Index column must be 'categories'"

def test_exercise_19():
    sol = exercise_19(dataset3, processed_keywords)
    
    assert abs(sol[0] - EXPECTED_Q19[0]) <= 1e-3
    assert abs(sol[1] - EXPECTED_Q19[1]) <= 1e-3

def test_exercise_20():
    sol = exercise_20(dataset3)
    pd.testing.assert_series_equal(sol, EXPECTED_Q20, check_names=False)
