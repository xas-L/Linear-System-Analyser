import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.subplots as sp
from typing import Tuple, List, Optional, Union
import warnings

# Ignore common NumPy warnings that might arise from RREF operations with zeros,
# as these are handled by the tolerance checks.
warnings.filterwarnings('ignore', category=RuntimeWarning)

class LinearSystemAnalyser:
    """
    Advanced Linear System Analyser & 3D Visualizer.
    This class takes a system of linear equations, performs Gauss-Jordan
    elimination with partial pivoting to find the Reduced Row Echelon Form (RREF),
    analyses the RREF to determine the nature of the solution set (unique,
    infinite, or no solution), and provides 3D visualizations of the system's
    geometric interpretation for 3-variable systems.
    """
    
    def __init__(self, tolerance: float = 1e-10):
        """
        Initializes the analyser.
        Args:
            tolerance (float): A small value to handle floating-point comparisons.
                               Numbers smaller than this are treated as zero.
        """
        self.tolerance = tolerance
        self.original_matrix: Optional[np.ndarray] = None
        self.rref_matrix: Optional[np.ndarray] = None
        self.solution_type: Optional[str] = None
        self.solution: Optional[Union[np.ndarray, dict, str]] = None
        self.rank_a: Optional[int] = None
        self.rank_ab: Optional[int] = None
        self.pivot_cols: Optional[List[int]] = None
        self.free_vars: Optional[List[int]] = None
        
    def get_user_input(self) -> np.ndarray:
        """
        Phase 1: Prompts the user to input the linear system and returns
        the augmented matrix [A|b].
        """
        print("=== Linear System Input ===")
        
        while True:
            try:
                m = int(input("Enter the number of equations (e.g., 3): "))
                n_vars = int(input("Enter the number of variables (e.g., 3): "))
                if m > 0 and n_vars > 0:
                    break
                else:
                    print("Number of equations and variables must be positive integers.")
            except ValueError:
                print("Invalid input. Please enter integers for dimensions.")
        
        augmented_matrix = np.zeros((m, n_vars + 1), dtype=float)
        
        print(f"\nEnter coefficients for {m} equations with {n_vars} variables.")
        print("Format: a1 a2 ... an b (space-separated coefficients and constant term)")
        print("Example for 2x + 3y - z = 5: 2 3 -1 5")
        
        for i in range(m):
            while True:
                try:
                    coeffs_str = input(f"Equation {i+1}: ").strip().split()
                    if len(coeffs_str) != n_vars + 1:
                        print(f"Please enter exactly {n_vars + 1} numbers for Equation {i+1}.")
                        continue
                    
                    row = [float(x) for x in coeffs_str]
                    augmented_matrix[i] = row
                    break
                except ValueError:
                    print("Invalid number format. Please enter space-separated numbers.")
        
        self.original_matrix = augmented_matrix.copy()
        return augmented_matrix
    
    def gauss_jordan_with_partial_pivoting(self, matrix: np.ndarray) -> np.ndarray:
        """
        Phase 2: Performs Gauss-Jordan elimination with partial pivoting to
        transform the input matrix into its Reduced Row Echelon Form (RREF).
        Args:
            matrix (np.ndarray): The augmented matrix [A|b] to be reduced.
        Returns:
            np.ndarray: The RREF of the input matrix.
        """
        rref = matrix.copy()
        num_rows, num_cols_aug = rref.shape
        
        lead = 0  # Index of the current pivot column being processed
        for r in range(num_rows): # Iterate through rows
            if lead >= num_cols_aug -1: # num_cols_aug-1 is the number of variable columns
                break
                
            # Partial pivoting: find the row with largest absolute value in current pivot column
            i = r
            for k in range(r + 1, num_rows):
                if abs(rref[k, lead]) > abs(rref[i, lead]):
                    i = k
            
            # Swap current row with the row having the largest pivot element
            if i != r:
                rref[[r, i]] = rref[[i, r]] # Note: NumPy fancy indexing for row swap
            
            # If pivot element is (close to) zero, move to the next potential pivot column
            if abs(rref[r, lead]) < self.tolerance:
                lead += 1
                # We need to re-process this row 'r' with the new 'lead' column
                # This requires a way to repeat the current row iteration or adjust loop.
                # A simpler way for now is to skip this column for this row,
                # but a more robust RREF might try to find another pivot in the same row further right.
                # For now, if the max pivot in column 'lead' (at or below row 'r') is zero,
                # we increment 'lead' and effectively skip this column for pivoting with row 'r'.
                # The outer loop will then process row 'r+1'.
                # A more standard RREF continues with the current row 'r' if lead is incremented.
                # Let's adjust to re-evaluate the current row if lead is incremented due to zero pivot.
                # This can be complex. A simpler approach:
                # If after pivoting, rref[r, lead] is still zero, this column has no pivot from this row down.
                # So we just increment lead and try the next column for the *same* row `r` in the next iteration
                # of the *outer* loop (which is not how it works).
                # The current structure: if rref[r,lead] is 0, this row `r` won't produce a pivot in col `lead`.
                # The `lead` variable should only advance if a pivot is successfully processed in that column.
                # Let's refine the lead advancement.
                
                # If rref[r, lead] is zero after pivoting, it means all entries in this column
                # at or below row r are zero. So, move to the next column.
                # The current row `r` will not have a pivot in this `lead` column.
                # The `lead` variable should be advanced, and the outer loop for `r` continues.
                # This means the current row `r` will not be a pivot row.
                # This logic is tricky. Standard RREF:
                if abs(rref[r, lead]) < self.tolerance: # Re-check after swap
                    lead += 1 # Try next column
                    # To ensure we don't skip a row that could become a pivot row later:
                    # we should re-evaluate the current row `r` with the new `lead`.
                    # This is best handled by ensuring `lead` only advances when a pivot is processed.
                    # The current loop structure: `for r in range(num_rows)` and `if lead >= num_cols_aug -1: break`
                    # If `rref[r, lead]` is zero, this row `r` cannot be used to create a pivot in column `lead`.
                    # We should find a row `k > r` to swap. If all are zero, then `lead` must advance.
                    # The pivoting logic above already tries to find a non-zero pivot.
                    # If `abs(rref[r, lead]) < self.tolerance` even after that, it means
                    # column `lead` (from row `r` downwards) is all zeros.
                    # So, we must advance `lead` and the current row `r` cannot be a pivot row.
                    # This means we should effectively skip this row `r` for creating a pivot.
                    # The `pivot_rows.append(r)` should only happen if a pivot is made.
                    # The loop for `r` will naturally go to `r+1`.
                    # The problem is if `lead` advances, `r` also advances, potentially missing a pivot.
                    # Correct RREF:
                    # Initialize lead = 0
                    # For r = 0 to num_rows-1:
                    #   If lead >= num_var_cols: break
                    #   i = r
                    #   while matrix[i, lead] is 0:  (and i < num_rows)
                    #     i++
                    #     if i == num_rows: (all zeros in this column from r down)
                    #       i = r; lead++;
                    #       if lead == num_var_cols: return (or break outer loop)
                    #       goto next_iteration_of_r_loop (or use a flag)
                    #   Swap row i and r
                    #   Normalize row r
                    #   Eliminate other rows
                    #   lead++
                    # This is complex to inject here. The current code is a common Gauss-Jordan variant.
                    # Let's assume the current pivoting correctly handles finding a non-zero pivot if available.
                    # If rref[r, lead] is zero, it means column `lead` from `r` down is zero.
                    # So, we must move to the next column.
                    # The current `r` cannot be a pivot row for this `lead`.
                    # The `lead` should advance, and `r` will advance. This might be okay.
                    # The number of pivots is determined by how many times we *don't* hit this continue.
                    continue # Skip to next column for this row, or effectively next row if lead increments enough

            # Scale current row to make pivot element 1
            rref[r] = rref[r] / rref[r, lead]
            
            # Eliminate other entries in the current pivot column (make them zero)
            for i in range(num_rows):
                if i != r: # For all other rows
                    # Subtract matrix[i, lead] times the pivot row (row r) from row i
                    rref[i] = rref[i] - rref[i, lead] * rref[r]
            
            lead += 1 # Move to the next pivot column
        
        # Clean up very small (near-zero) entries resulting from floating point arithmetic
        rref[np.abs(rref) < self.tolerance] = 0.0
        
        self.rref_matrix = rref
        return rref

    def analyse_solution(self) -> None:
        """
        Phase 3: Analyses the RREF matrix to determine the solution type,
        rank, pivot columns, free variables, and the solution itself.
        Updates attributes of the class.
        """
        if self.rref_matrix is None:
            raise ValueError("RREF matrix not computed. Call gauss_jordan_with_partial_pivoting first.")
            
        rref = self.rref_matrix
        num_rows, num_cols_aug = rref.shape
        num_vars = num_cols_aug - 1
        
        pivot_positions = [] # Store (row, col) of pivots
        for r in range(num_rows):
            pivot_found_in_row = False
            for c in range(num_vars): # Only check variable columns
                if abs(rref[r, c] - 1.0) < self.tolerance: # Potential pivot is 1
                    # Check if it's a true pivot (all other entries in its column are zero)
                    is_pivot_col = True
                    for k in range(num_rows):
                        if k != r and abs(rref[k, c]) > self.tolerance:
                            is_pivot_col = False
                            break
                    if is_pivot_col:
                        pivot_positions.append((r, c))
                        pivot_found_in_row = True
                        break # Move to next row once pivot for this row is found
            # If no pivot found in this row for variable columns, it might be a zero row or inconsistent
            
        self.pivot_cols = sorted(list(set(col for row, col in pivot_positions)))
        self.rank_a = len(self.pivot_cols) # Rank of coefficient matrix A
        
        # Check for inconsistency
        inconsistent_row_found = False
        for r in range(num_rows):
            is_all_zeros_in_coeffs = all(abs(rref[r, c]) < self.tolerance for c in range(num_vars))
            if is_all_zeros_in_coeffs and abs(rref[r, num_vars]) > self.tolerance: # Last col is num_vars
                inconsistent_row_found = True
                break
        
        # Rank of augmented matrix [A|b]
        # Number of non-zero rows in RREF of [A|b]
        # A simpler way: if inconsistent_row_found, rank_ab > rank_a
        # Otherwise, rank_ab = rank_a
        if inconsistent_row_found:
            # Heuristic: if an inconsistent row [0...0 | d] (d!=0) exists,
            # rank_ab is effectively rank_a + 1 for comparison purposes.
            # A more formal way to get rank_ab is to count non-zero rows in rref of [A|b].
            # Let's count non-zero rows in the full RREF matrix
            self.rank_ab = 0
            for r_idx in range(num_rows):
                if np.any(np.abs(rref[r_idx, :]) > self.tolerance):
                    self.rank_ab +=1
        else:
            self.rank_ab = self.rank_a


        if inconsistent_row_found: # Equivalent to self.rank_a < self.rank_ab
            self.solution_type = "No solution"
            self.solution = "System is inconsistent due to a row like [0 0 ... 0 | c] where c != 0."
            return

        self.free_vars = [j for j in range(num_vars) if j not in self.pivot_cols]

        if self.rank_a == num_vars:
            self.solution_type = "Unique solution"
            solution_vector = np.zeros(num_vars)
            # For each variable (which must be a pivot variable)
            for var_idx in range(num_vars): # Iterate 0 to num_vars-1
                # Find the pivot row for this variable column
                found_pivot_for_var = False
                for p_row, p_col in pivot_positions:
                    if p_col == var_idx:
                        solution_vector[var_idx] = rref[p_row, num_vars] # Constant term
                        found_pivot_for_var = True
                        break
                if not found_pivot_for_var:
                    # This should not happen if rank_a == num_vars and consistent
                    # It implies a variable is free, contradicting rank_a == num_vars
                    # Or the pivot finding logic for solution_vector needs adjustment
                    # Let's refine based on pivot_positions directly
                    pass # Handled by the loop below for pivot_cols
            
            # Simpler way for unique solution:
            sol_vec = np.zeros(num_vars)
            for r_pivot, c_pivot in pivot_positions:
                sol_vec[c_pivot] = rref[r_pivot, num_vars]
            self.solution = sol_vec

        else: # rank_a < num_vars, so infinite solutions
            self.solution_type = "Infinite solutions"
            
            # Particular solution (set free variables to 0)
            x_p = np.zeros(num_vars)
            for r_pivot, c_pivot in pivot_positions:
                x_p[c_pivot] = rref[r_pivot, num_vars] # Constant term
            
            special_solutions = []
            for free_var_idx in self.free_vars:
                s_sol = np.zeros(num_vars)
                s_sol[free_var_idx] = 1.0 # Set this free variable to 1
                
                # Solve for pivot variables in terms of this free variable
                for r_pivot, c_pivot in pivot_positions:
                    # The equation for this pivot row is:
                    # 1*x_pivot + sum(coeff_free_var * x_free_var_set_to_0_or_1) = 0 (for homogeneous part)
                    # So, x_pivot = -coeff_of_this_free_var_in_pivot_row
                    s_sol[c_pivot] = -rref[r_pivot, free_var_idx]
                special_solutions.append(s_sol)
            
            self.solution = {
                'particular_solution_xp': x_p,
                'special_solutions_s': special_solutions,
                'free_variable_indices': self.free_vars
            }

    def format_solution_output(self) -> str:
        """Formats the determined solution for user-friendly display."""
        if self.solution_type == "No solution":
            return f"No solution: {self.solution}"
        
        elif self.solution_type == "Unique solution":
            vars_str = ", ".join([f"x{i+1} = {val:.4f}" for i, val in enumerate(self.solution)])
            return f"Unique solution: ({vars_str})"
        
        elif self.solution_type == "Infinite solutions":
            sol_data = self.solution
            n_vars = len(sol_data['particular_solution_xp'])
            
            output_str = "Infinite solutions:\nx = x_p"
            for i in range(len(sol_data['special_solutions_s'])):
                output_str += f" + c{i+1}*s{i+1}"
            output_str += "\n\n"
            
            xp_str = ", ".join([f"{val:.4f}" for val in sol_data['particular_solution_xp']])
            output_str += f"Particular solution (x_p): ({xp_str})\n"
            
            for i, s_sol in enumerate(sol_data['special_solutions_s']):
                s_str = ", ".join([f"{val:.4f}" for val in s_sol])
                output_str += f"Special solution (s{i+1}): ({s_str})\n"
            
            free_vars_str = ", ".join([f"x{idx+1} (param c{i+1})" for i, idx in enumerate(sol_data['free_variable_indices'])])
            output_str += f"\nFree variables: {free_vars_str}"
            return output_str
        return "Solution not yet sed."

    def _plot_plane(self, ax, coeffs: np.ndarray, color: str, label: Optional[str] = None, is_plotly: bool = False):
        """Helper function to plot a single plane ax+by+cz=d."""
        a, b, c, d_const = coeffs
        
        # Skip degenerate planes like 0x+0y+0z = 0
        if abs(a) < self.tolerance and abs(b) < self.tolerance and abs(c) < self.tolerance and abs(d_const) < self.tolerance:
            print(f"Skipping plot for degenerate equation: {coeffs}")
            return
        # Handle inconsistent planes like 0x+0y+0z = d (d!=0)
        if abs(a) < self.tolerance and abs(b) < self.tolerance and abs(c) < self.tolerance and abs(d_const) > self.tolerance:
            print(f"Cannot plot inconsistent equation: {coeffs} (0 = {d_const:.2f})")
            return

        plot_lim = 10 # Visual range for axes
        v_range = np.linspace(-plot_lim, plot_lim, 20) # Reduced points for clarity
        
        plane_label = label if label else f'{a:.1f}x + {b:.1f}y + {c:.1f}z = {d_const:.1f}'

        if abs(c) > self.tolerance: # Solve for z: z = (d - ax - by)/c
            X, Y = np.meshgrid(v_range, v_range)
            Z = (d_const - a * X - b * Y) / c
            if is_plotly:
                ax.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.5, colorscale=[[0,color],[1,color]], showscale=False, name=plane_label))
            else:
                ax.plot_surface(X, Y, Z, alpha=0.3, color=color, label=plane_label)
        elif abs(b) > self.tolerance: # Solve for y: y = (d - ax - cz)/c  -> Plane parallel to z-axis
            X, Z = np.meshgrid(v_range, v_range)
            Y = (d_const - a * X - c * Z) / b
            if is_plotly:
                ax.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.5, colorscale=[[0,color],[1,color]], showscale=False, name=plane_label))
            else:
                ax.plot_surface(X, Y, Z, alpha=0.3, color=color, label=plane_label)
        elif abs(a) > self.tolerance: # Solve for x: x = (d - by - cz)/a -> Plane parallel to yz-plane
            Y, Z = np.meshgrid(v_range, v_range)
            X = (d_const - b * Y - c * Z) / a
            if is_plotly:
                ax.add_trace(go.Surface(x=X, y=Y, z=Z, opacity=0.5, colorscale=[[0,color],[1,color]], showscale=False, name=plane_label))
            else:
                ax.plot_surface(X, Y, Z, alpha=0.3, color=color, label=plane_label)
        # else: handled by initial check for 0x+0y+0z=0 or 0x+0y+0z=d

    def plot_3d_system_matplotlib(self, matrix: np.ndarray, title: str, ax: plt.Axes):
        """Plots a 3-variable system using Matplotlib on the given Axes object."""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        for i in range(matrix.shape[0]):
            self._plot_plane(ax, matrix[i], colors[i % len(colors)], is_plotly=False)
        
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title(title)
        # Set consistent limits for comparison
        plot_lim = 10
        ax.set_xlim([-plot_lim, plot_lim])
        ax.set_ylim([-plot_lim, plot_lim])
        ax.set_zlim([-plot_lim, plot_lim])

        if self.solution_type == "Unique solution" and len(self.solution) == 3:
            sol_pt = self.solution
            ax.scatter(sol_pt[0], sol_pt[1], sol_pt[2], color='black', s=100, depthshade=True, label=f'Solution: ({sol_pt[0]:.2f}, {sol_pt[1]:.2f}, {sol_pt[2]:.2f})')
        elif self.solution_type == "Infinite solutions" and self.original_matrix.shape[1]-1 == 3 and len(self.free_vars) == 1:
            sol_data = self.solution
            xp = sol_data['particular_solution_xp']
            s1 = sol_data['special_solutions_s'][0]
            t_vals = np.array([-plot_lim*0.5, plot_lim*0.5]) # Plot a segment of the line
            line_points = xp.reshape(-1,1) + s1.reshape(-1,1) * t_vals.reshape(1,-1)
            line_points = line_points.T
            ax.plot(line_points[:,0], line_points[:,1], line_points[:,2], color='black', linewidth=3, label='Line of Solutions')
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))


    def plot_3d_system_plotly(self, matrix: np.ndarray, title: str, fig_ref: go.Figure, row: int, col: int):
        """Adds traces for a 3-variable system to a Plotly Figure reference."""
        colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'cyan', 'magenta']
        # Create a temporary figure to leverage _plot_plane, then transfer traces
        temp_fig = go.Figure()
        for i in range(matrix.shape[0]):
            self._plot_plane(temp_fig, matrix[i], colors[i % len(colors)], is_plotly=True, label=f'Eq {i+1}')
        
        for trace in temp_fig.data:
            trace.showlegend = True # Ensure individual plane legends are on
            fig_ref.add_trace(trace, row=row, col=col)

        # Add solution visualization to this specific subplot
        plot_lim = 10
        if self.solution_type == "Unique solution" and len(self.solution) == 3:
            sol_pt = self.solution
            fig_ref.add_trace(go.Scatter3d(
                x=[sol_pt[0]], y=[sol_pt[1]], z=[sol_pt[2]],
                mode='markers', marker=dict(size=8, color='black'),
                name=f'Solution: ({sol_pt[0]:.2f}, {sol_pt[1]:.2f}, {sol_pt[2]:.2f})'
            ), row=row, col=col)
        elif self.solution_type == "Infinite solutions" and self.original_matrix.shape[1]-1 == 3 and len(self.free_vars) == 1:
            sol_data = self.solution
            xp = sol_data['particular_solution_xp']
            s1 = sol_data['special_solutions_s'][0]
            t_vals = np.array([-plot_lim*0.5, plot_lim*0.5])
            line_points = xp.reshape(-1,1) + s1.reshape(-1,1) * t_vals.reshape(1,-1)
            line_points = line_points.T
            fig_ref.add_trace(go.Scatter3d(
                x=line_points[:,0], y=line_points[:,1], z=line_points[:,2],
                mode='lines', line=dict(width=5, color='black'), name='Line of Solutions'
            ), row=row, col=col)
        # Note: Plotting a plane of solutions for n=3, 2 free vars means the RREF defines that plane.
        # The _plot_plane will draw the RREF planes which constitute the solution.

    def _extend_to_3d(self, matrix: np.ndarray) -> np.ndarray:
        """
        Extends a system with n < 3 variables to a 3-variable system
        by adding zero coefficients for the missing variables, for visualization.
        Args:
            matrix (np.ndarray): The augmented matrix [A|b].
        Returns:
            np.ndarray: The 3-variable augmented matrix.
        """
        num_rows, num_cols_aug = matrix.shape
        num_vars_orig = num_cols_aug - 1
        
        if num_vars_orig >= 3:
            return matrix.copy() # Return a copy if already 3 or more vars
        
        # Create new matrix for 3 variables (4 columns: x, y, z, const)
        extended_matrix = np.zeros((num_rows, 4), dtype=float)
        
        # Copy existing variable coefficients
        extended_matrix[:, :num_vars_orig] = matrix[:, :num_vars_orig]
        # Copy constant terms to the last column of the new matrix
        extended_matrix[:, 3] = matrix[:, num_vars_orig] 
        
        return extended_matrix

    def visualize_system(self, use_plotly: bool = True):
        """
        Orchestrates the 3D visualization of both the original and RREF systems.
        Args:
            use_plotly (bool): If True, uses Plotly for interactive plots.
                               Otherwise, uses Matplotlib.
        """
        if self.original_matrix is None or self.rref_matrix is None:
            print("System not solved yet. Please run analysis first.")
            return

        num_vars = self.original_matrix.shape[1] - 1

        if num_vars > 3:
            print(f"Direct 3D visualization is not meaningful for {num_vars} variables.")
            return
        
        # Extend matrices to 3 variables if num_vars < 3
        original_vis_matrix = self._extend_to_3d(self.original_matrix)
        rref_vis_matrix = self._extend_to_3d(self.rref_matrix)

        # Determine plot titles based on solution type
        rref_plot_title = "RREF System"
        if self.solution_type == "No solution":
            rref_plot_title += " - Inconsistent (No Solution)"
        elif self.solution_type == "Unique solution":
            rref_plot_title += " - Unique Solution"
        elif self.solution_type == "Infinite solutions":
            if num_vars == 3 and len(self.free_vars) == 1:
                rref_plot_title += " - Line of Solutions"
            elif num_vars == 3 and len(self.free_vars) == 2:
                 # Rank A must be 1 for this case with 3 vars
                if self.rank_a == 1:
                     rref_plot_title += " - Plane of Solutions"
                else: # e.g. a 2-var system extended, solution is a line
                     rref_plot_title += " - Infinite Solutions"
            else: # Covers n<3 extended to 3D, or other general infinite cases
                rref_plot_title += " - Infinite Solutions"


        if use_plotly:
            fig = sp.make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'surface'}, {'type': 'surface'}]],
                subplot_titles=['Original System', rref_plot_title],
                horizontal_spacing=0.05 # Adjust spacing if needed
            )
            
            # Plot original system (without solution markers)
            self.plot_3d_system_plotly(original_vis_matrix, "Original System", fig, row=1, col=1)
            
            # Plot RREF system (with solution markers)
            self.plot_3d_system_plotly(rref_vis_matrix, rref_plot_title, fig, row=1, col=2)
            
            plot_lim = 10
            fig.update_layout(
                height=700, width=1400, title_text="Linear System Geometric Interpretation",
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                scene1=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                            xaxis=dict(range=[-plot_lim,plot_lim]), 
                            yaxis=dict(range=[-plot_lim,plot_lim]), 
                            zaxis=dict(range=[-plot_lim,plot_lim]),
                            aspectmode='cube'), # Maintain aspect ratio
                scene2=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
                            xaxis=dict(range=[-plot_lim,plot_lim]), 
                            yaxis=dict(range=[-plot_lim,plot_lim]), 
                            zaxis=dict(range=[-plot_lim,plot_lim]),
                            aspectmode='cube')  # Maintain aspect ratio
            )
            fig.show()
        else: # Use Matplotlib
            fig = plt.figure(figsize=(16, 7))
            
            ax1 = fig.add_subplot(121, projection='3d')
            self.plot_3d_system_matplotlib(original_vis_matrix, "Original System", ax1)
            
            ax2 = fig.add_subplot(122, projection='3d')
            self.plot_3d_system_matplotlib(rref_vis_matrix, rref_plot_title, ax2)
            
            fig.suptitle("Linear System Geometric Interpretation", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust layout to make space for suptitle
            plt.show()

    def print_matrix(self, matrix: np.ndarray, title: str):
        """Prints a matrix with a title and clear formatting."""
        print(f"\n{title}:")
        print("-" * (len(title) + 2)) # Adjust underline to title length
        if matrix is None or matrix.size == 0:
            print("Matrix is empty.")
            print("-" * (len(title) + 2))
            return

        num_rows, num_cols = matrix.shape
        num_vars = num_cols -1 # Assuming augmented matrix
        
        header = "  " + " ".join([f"x{j+1:<7}" for j in range(num_vars)]) + "| RHS"
        print(header)
        print("-" * len(header))

        for i in range(num_rows):
            row_str = f"R{i+1}: ["
            for j in range(num_cols):
                if j == num_vars: # Separator before RHS
                    row_str += " |"
                row_str += f"{matrix[i, j]:8.4f}" # Format for alignment
            row_str += " ]"
            print(row_str)
        print("-" * len(header))
    
    def run_analysis(self, matrix_input: Optional[np.ndarray] = None, visualize: bool = True, use_plotly: bool = True):
        """
        Main application flow: gets input, computes RREF, analyses solution,
        and visualizes the system.
        Args:
            matrix_input (Optional[np.ndarray]): Predefined augmented matrix.
                                                If None, prompts user for input.
            visualize (bool): Whether to generate 3D plots.
            use_plotly (bool): If True and visualize is True, uses Plotly.
                               Otherwise, uses Matplotlib.
        Returns:
            dict: A dictionary containing all analysis results.
        """
        print("=" * 70)
        print("    ADVANCED LINEAR SYSTEM analyser & 3D VISUALIZER")
        print("=" * 70)
        
        if matrix_input is None:
            current_matrix = self.get_user_input()
        else:
            print("Using provided matrix input.")
            self.original_matrix = matrix_input.copy()
            current_matrix = matrix_input.copy() # Work with a copy
        
        self.print_matrix(self.original_matrix, "Original Augmented Matrix [A|b]")
        
        print("\nComputing RREF with partial pivoting...")
        self.gauss_jordan_with_partial_pivoting(current_matrix) # Modifies self.rref_matrix
        self.print_matrix(self.rref_matrix, "Reduced Row Echelon Form (RREF) [R_0|d]")
        
        print("\nAnalyzing solution from RREF...")
        self.analyse_solution() # Uses self.rref_matrix, updates solution attributes
        
        print(f"\nRank of coefficient matrix A (r): {self.rank_a}")
        # Rank_ab from analyse_solution is the number of non-zero rows in RREF of [A|b]
        # This is more direct than comparing rank_a and rank_ab for inconsistency.
        # Inconsistency is directly checked in analyse_solution.
        print(f"Number of variables (n): {self.original_matrix.shape[1] - 1}")
        
        print(f"\nSOLUTION ANALYSIS:")
        print("=" * 50)
        print(self.format_solution_output())
        print("=" * 50)
        
        num_vars_orig = self.original_matrix.shape[1] - 1
        if visualize and num_vars_orig <= 3:
            print(f"\nCreating 3D visualization (using {'Plotly' if use_plotly else 'Matplotlib'})...")
            self.visualize_system(use_plotly=use_plotly)
        elif visualize and num_vars_orig > 3:
            print(f"\nNote: Direct 3D plane visualization is not meaningful for {num_vars_orig} variables.")
            
        return {
            'original_matrix': self.original_matrix,
            'rref_matrix': self.rref_matrix,
            'solution_type': self.solution_type,
            'solution': self.solution,
            'rank_a': self.rank_a,
            'rank_ab': self.rank_ab, # rank_ab as calculated in analyse_solution
            'pivot_columns': self.pivot_cols,
            'free_variables': self.free_vars
        }

def run_interactive_session():
    """Runs an interactive command-line session for the analyser."""
    analyser = LinearSystemanalyser()
    while True:
        try:
            analyser.run_analysis(use_plotly=True) # Default to Plotly for interactive
        except KeyboardInterrupt:
            print("\n\nSession terminated by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn unexpected error occurred: {e}")
            print("Restarting session or enter 'n' to exit.")
        
        print("\n" + "="*70)
        choice = input("analyse another system? (y/n): ").strip().lower()
        if choice != 'y':
            print("Exiting analyser. Goodbye!")
            break
        # Reset analyser state for new system (optional, or create new instance)
        analyser = LinearSystemanalyser() 


def run_examples():
    """Runs predefined example systems for testing and demonstration."""
    analyser_instance = LinearSystemanalyser()
    
    print("\n\nEXAMPLE 1: Unique solution (3 vars)")
    example1 = np.array([
        [1, 2, -1, 3],
        [2, -1, 1, 1],
        [1, -1, 2, -1]
    ], dtype=float)
    analyser_instance.run_analysis(matrix_input=example1.copy(), use_plotly=True)
    
    print("\n" + "="*80 + "\n")
    analyser_instance = LinearSystemanalyser() # Reset for next example
    print("EXAMPLE 2: Infinite solutions - Line (3 vars, 1 free)")
    example2 = np.array([
        [1, 2, 3, 6], # x + 2y + 3z = 6
        [2, 4, 7, 15],# 2x + 4y + 7z = 15 -> z = 3
        [1, 2, 1, 0]  # x + 2y + z = 0   -> x + 2y = -3
    ], dtype=float) # RREF should be [[1,2,0,-3],[0,0,1,3],[0,0,0,0]]
    analyser_instance.run_analysis(matrix_input=example2.copy(), use_plotly=True)

    print("\n" + "="*80 + "\n")
    analyser_instance = LinearSystemanalyser() 
    print("EXAMPLE 3: Infinite solutions - Plane (3 vars, 2 free)")
    example3 = np.array([
        [1, -2, 3, 4],  # One plane
        [2, -4, 6, 8],  # Dependent plane (2 * Eq1)
        [3, -6, 9, 12]  # Dependent plane (3 * Eq1)
    ], dtype=float) # RREF should be [[1,-2,3,4],[0,0,0,0],[0,0,0,0]]
    analyser_instance.run_analysis(matrix_input=example3.copy(), use_plotly=True)
    
    print("\n" + "="*80 + "\n")
    analyser_instance = LinearSystemanalyser() 
    print("EXAMPLE 4: No solution (3 vars)")
    example4 = np.array([
        [1, 1, 1, 2],
        [1, 1, 1, 3], # Inconsistent with Eq1
        [2, 3, 4, 5]
    ], dtype=float)
    analyser_instance.run_analysis(matrix_input=example4.copy(), use_plotly=True)

    print("\n" + "="*80 + "\n")
    analyser_instance = LinearSystemanalyser()
    print("EXAMPLE 5: 2-variable system (Unique Solution)")
    example5 = np.array([
        [1, 1, 3], # x + y = 3
        [1, -1, 1] # x - y = 1
    ], dtype=float) # Solution: x=2, y=1
    analyser_instance.run_analysis(matrix_input=example5.copy(), use_plotly=True)

    print("\n" + "="*80 + "\n")
    analyser_instance = LinearSystemanalyser()
    print("EXAMPLE 6: 2-variable system (Infinite Solutions)")
    example6 = np.array([
        [1, 2, 3],   # x + 2y = 3
        [2, 4, 6]    # 2x + 4y = 6 (Dependent)
    ], dtype=float)
    analyser_instance.run_analysis(matrix_input=example6.copy(), use_plotly=True)


if __name__ == "__main__":
    # Choose one mode to run:
    run_interactive_session()
    # run_examples()
    
    # Example of analyzing a specific system directly:
    # analyser = LinearSystemanalyser()
    # custom_matrix = np.array([
    #     [1, 0, 1, 4],  # x + z = 4
    #     [0, 1, -1, -1], # y - z = -1
    #     [1, 1, 0, 3]   # x + y = 3 (consistent with above, e.g. x=1,y=2,z=3)
    # ], dtype=float)
    # results = analyser.run_analysis(matrix_input=custom_matrix, use_plotly=True)
    # print("\nFinal results dictionary:", results)