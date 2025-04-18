from scipy.optimize import linprog

# Objective function coefficients (negated for maximization)
c = [-0.12, -0.055, -0.10, -0.11, -0.095]  # Maximizing return (negate since linprog minimizes)

# Left-hand side of inequality constraints (A_ub * x ≤ b_ub)
A_ub = [
    [1, 1, 1, 1, 1],          # Total investment constraint
    [1, 0, 0, 0, 0],          # Max Agriculture
    [0, 0, 0, 1, 0],          # Max Manufacturing
    [0, 0, 1, 0, 0],          # Max Banking
    [-0.45, -0.45, 0, 1, 0],  # Manufacturing percentage constraint
    [-2.00, -2.00, 0, 0, 1]   # Real Estate percentage constraint
]

# Right-hand side of inequality constraints (b_ub)
b_ub = [280000, 100000, 100000, 50000, 0, 0]  # Adjusted for inequality constraints

# Variable bounds (x_i ≥ 0)
x_bounds = [(0, None), (0, None), (0, None), (0, None), (0, None)]

# Solve the linear programming problem
result = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=x_bounds, method="highs")

# Print results
if result.success:
    optimal_values = result.x
    max_return = -result.fun  # Negate to get maximized return
    print("Optimal Investment Allocation:")
    print(f"Agriculture: ${optimal_values[0]:,.2f}")
    print(f"Healthcare: ${optimal_values[1]:,.2f}")
    print(f"Banking: ${optimal_values[2]:,.2f}")
    print(f"Manufacturing: ${optimal_values[3]:,.2f}")
    print(f"Real Estate: ${optimal_values[4]:,.2f}")
    print(f"Maximum Return: ${max_return:,.2f}")
else:
    print("Optimization failed:", result.message)


