# Advanced Linear System Analyser & 3D Visualizer

## Overview

This Python project provides a comprehensive tool for analysing systems of linear equations. It implements core linear algebra algorithms to solve systems, determine the nature of their solution sets, and offer insightful 3D visualizations for systems with three variables. The primary goal is to bridge the gap between theoretical linear algebra concepts and their practical, geometric interpretations, making the subject more intuitive.

This project was inspired by and aligns with concepts typically found in linear algebra texts such as Gilbert Strang's "Introduction to Linear Algebra."

## Features

### Core Functionality:
* **User Input:** Allows users to define a system of linear equations by specifying the number of equations and variables, and then entering the coefficients and constant terms.
* **Augmented Matrix Representation:** Internally represents the system $A\mathbf{x} = \mathbf{b}$ as an augmented matrix $[A|\mathbf{b}]$ using NumPy.
* **Reduced Row Echelon Form (RREF):** Implements Gauss-Jordan elimination to transform the augmented matrix into its RREF ($R_0$).
* **Solution Analysis:**
    * Determines the rank of the coefficient matrix $A$ and the augmented matrix $[A|\mathbf{b}]$.
    * Classifies the solution set as:
        * **Unique Solution:** Provides the specific solution vector.
        * **Infinite Solutions:** Identifies free variables and provides a parameterized solution in the form $\mathbf{x} = \mathbf{x}_p + c_1\mathbf{s}_1 + c_2\mathbf{s}_2 + \dots$ (particular solution plus a linear combination of special solutions forming the nullspace basis).
        * **No Solution:** Detects inconsistent systems.
* **Textual Output:** Clearly presents the original matrix, RREF matrix, rank information, and the determined solution.

### Advanced & Rigorous Extensions:
* **Partial Pivoting:** The RREF computation includes partial pivoting (selecting the largest absolute value pivot) to enhance numerical stability.
* **3D Visualization (for 3-variable systems):**
    * Plots equations as planes in $\mathbb{R}^3$ for both the original system and its RREF.
    * Handles various plane orientations, including those parallel to coordinate axes.
    * Visually represents the solution set:
        * A distinct point for unique solutions.
        * A line segment for infinite solutions with one free variable.
        * The defining plane(s) from RREF for infinite solutions with two free variables.
    * Offers visualization using both **Matplotlib** (for static plots) and **Plotly** (for interactive 3D exploration).
    * Extends 1-variable and 2-variable systems into 3D for a consistent visualization approach.
* **Interactive CLI:** Provides a command-line interface to analyse multiple systems in a session.

## Core Linear Algebra Concepts Utilized
* Systems of Linear Equations ($A\mathbf{x}=\mathbf{b}$)
* Augmented Matrices
* Elementary Row Operations
* Gauss-Jordan Elimination
* Reduced Row Echelon Form (RREF / $R_0$)
* Pivots, Pivot Columns, Free Variables
* Rank of a Matrix
* Consistency of Systems
* Structure of Solution Sets (Particular Solution $\mathbf{x}_p$ + Nullspace Solution $\mathbf{x}_n$)
* Special Solutions (Basis for Nullspace $N(A)$)
* Geometric Interpretation of Linear Equations as Planes
* Intersection of Planes (representing solution sets: point, line, plane, or empty)

## Requirements/Dependencies
* Python 3.x
* NumPy (`pip install numpy`)
* Matplotlib (`pip install matplotlib`)
* Plotly (`pip install plotly`)

## How to Run
1.  **Ensure Dependencies are Installed:**
    ```bash
    pip install numpy matplotlib plotly
    ```
2.  **Save the Script:** Save the Python code as `linear_system_analyser.py` (or your preferred filename).
3.  **Execute from the Command Line:**
    ```bash
    python linear_system_analyser.py
    ```
4.  **Follow Prompts:**
    * The script will first offer to run an interactive session or predefined examples.
    * If running interactively, it will prompt you to enter the number of equations and variables.
    * Then, enter the coefficients for each equation, including the constant term on the right-hand side, separated by spaces.

## Example Usage

Upon running, you can choose to enter a system interactively:

=== Linear System Input ===Enter the number of equations (e.g., 3): 3Enter the number of variables (e.g., 3): 3Enter coefficients for 3 equations with 3 variables.Format: a1 a2 ... an b (space-separated coefficients and constant term)Example for 2x + 3y - z = 5: 2 3 -1 5Equation 1: 1 2 -1 3Equation 2: 2 -1 1 1Equation 3: 1 -1 2 -1
The analyser will then output:
* The original augmented matrix.
* The RREF matrix.
* Rank information.
* A detailed analysis of the solution type (unique, infinite, or no solution).
* If applicable (for 3-variable systems), it will generate and display interactive 3D plots of the planes and the solution set.

The script also includes a `run_examples()` function (commented out by default in `if __name__ == "__main__":`) that demonstrates various types of systems.


## Acknowledgements
This project draws heavily on the foundational concepts and pedagogical approaches to linear algebra as presented in "Introduction to Linear Algebra" by Gilbert Strang and "Linear Algebra done right" by Sheldon Axler. The aim was to bring these concepts to life through computation and visualization for those interested in Linear Algebra and/or Machine Learning.
