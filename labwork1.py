def function(x):
    return x ** 2

def derivative(x):
    return 2 * x

def perform_gradient_descent(initial_value, iterations, learning_rate=0.1):
    current_x = initial_value
    print("Function: f(x) = x^2")
    print("Step\t\t x\t\t f(x)")

    for step in range(1, iterations + 1):
        grad = derivative(current_x)
        current_x -= learning_rate * grad
        current_fx = function(current_x)
        print(f"Step {step}:\t {current_x:.6f}\t {current_fx:.6f}")

    return current_x, function(current_x)

start_value = 10
total_steps = 10
user_input = input("Enter learning rate (default is 0.1): ")
learning_rate = float(user_input) if user_input else 0.1

minimum_x, minimum_fx = perform_gradient_descent(start_value, total_steps, learning_rate)
print("\nApproximate minimum x:", minimum_x)
print("Function value at minimum f(x):", minimum_fx)
