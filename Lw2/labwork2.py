import matplotlib.pyplot as plt

# Load data from CSV file
file_path = "lr.csv"
x = []
y = []
losses = []

# Initialize weights
w0 = 1
w1 = 1

# Read data points from file
with open(file_path, "r") as file:
    for line in file:
        xi, yi = map(float, line.strip().split(","))
        x.append(xi)
        y.append(yi)

# Mean squared error for a single data point
def mse(w0, w1, xi, yi):
    return 0.5 * (w1 * xi + w0 - yi) ** 2

# Gradient of loss with respect to w0
def gradient_w0(w0, w1, xi, yi):
    return w1 * xi + w0 - yi

# Gradient of loss with respect to w1
def gradient_w1(w0, w1, xi, yi):
    return xi * (w1 * xi + w0 - yi)

# Total loss (mean squared error across all data points)
def compute_loss(w0, w1, x_vals, y_vals):
    N = len(x_vals)
    return sum(mse(w0, w1, xi, yi) for xi, yi in zip(x_vals, y_vals)) / N

# Gradient descent algorithm
def gradient_descent(w0, w1, x_vals, y_vals, learning_rate, iterations):
    for i in range(iterations):
        N = len(x_vals)
        dw0 = sum(gradient_w0(w0, w1, xi, yi) for xi, yi in zip(x_vals, y_vals)) / N
        dw1 = sum(gradient_w1(w0, w1, xi, yi) for xi, yi in zip(x_vals, y_vals)) / N
        w0 -= learning_rate * dw0
        w1 -= learning_rate * dw1

        current_loss = compute_loss(w0, w1, x_vals, y_vals)
        losses.append(current_loss)

        print(f"w0: {round(w0, 2)}, w1: {round(w1, 2)}, loss: {round(current_loss, 2)}")
    return w0, w1

# Train the model
w0, w1 = gradient_descent(w0, w1, x, y, learning_rate=0.001, iterations=200)

# Plot data and fitted line
plt.scatter(x, y, color='blue', label='Data Points')
plt.plot(x, [w1 * xi + w0 for xi in x], color='red', label='Regression Line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.show()

# Plot loss over iterations
plt.plot(range(len(losses)), losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Over Iterations')
plt.show()
