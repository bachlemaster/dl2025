from math import log, exp
import matplotlib.pyplot as plt

file_path = "loan.csv"  

w0 = 0  
w1 = 1  
w2 = 2  

x1 = []
x2 = []
y = []
l = []  

with open(file_path, "r") as file:
    next(file)  
    for line in file:
        values = line.strip().split(",")
        x1.append(float(values[0]))
        x2.append(float(values[1]))
        y.append(int(values[2]))

def j(w0, w1, w2, x1, x2, y):
    N = len(x1)
    return -(1 / N) * sum(
        yi * (w1 * xi1 + w2 * xi2 + w0) - log(1 + exp(w1 * xi1 + w2 * xi2 + w0))
        for xi1, xi2, yi in zip(x1, x2, y)
    )

def sigmoid(x):
    return 1 / (1 + exp(-x))

def df0(w0, w1, w2, x1, x2, y):
    return 1 - y - sigmoid(-(w1 * x1 + w2 * x2 + w0))

def df1(w0, w1, w2, x1, x2, y):
    return -y * x1 + x1 * (1 - sigmoid(-(w1 * x1 + w2 * x2 + w0)))

def df2(w0, w1, w2, x1, x2, y):
    return -y * x2 + x2 * (1 - sigmoid(-(w1 * x1 + w2 * x2 + w0)))

def grad_desc(w0, w1, w2, x1, x2, y, lr, times):
    for i in range(times):
        N = len(x1)
        w0 = w0 - lr * sum(df0(w0, w1, w2, xi1, xi2, yi) for xi1, xi2, yi in zip(x1, x2, y)) / N
        w1 = w1 - lr * sum(df1(w0, w1, w2, xi1, xi2, yi) for xi1, xi2, yi in zip(x1, x2, y)) / N
        w2 = w2 - lr * sum(df2(w0, w1, w2, xi1, xi2, yi) for xi1, xi2, yi in zip(x1, x2, y)) / N

        loss = j(w0, w1, w2, x1, x2, y)
        l.append(loss)

        if i % 100 == 0 or i == times - 1:
            print(f"Iteration {i} -> w0: {round(w0, 2)}, w1: {round(w1, 2)}, w2: {round(w2, 2)}, loss: {round(loss, 4)}")
    return w0, w1, w2

w0, w1, w2 = grad_desc(w0, w1, w2, x1, x2, y, lr=0.9, times=10000)

plt.scatter(x1, x2, c=y, cmap='bwr', edgecolors='k')
x_vals = [min(x1), max(x1)]
y_vals = [-(w0 + w1 * x) / w2 for x in x_vals]
plt.plot(x_vals, y_vals, label='Decision Boundary')
plt.ylim(min(x2)-1, max(x2)+1)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.grid(True)
plt.show()

plt.plot(l)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Loss Function over Iterations')
plt.grid(True)
plt.show()
