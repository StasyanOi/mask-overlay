import math
import numpy as np


def stats(x, y):
    print("------------------------")
    print("x min")
    print(x.min())
    print("y min")
    print(y.min())

    print("x min")
    print(x.max())
    print("y min")
    print(y.max())


def line(k, x, b):
    return k * x + b


def l2(real, gen):
    return np.sum((real - gen) ** 2)


def get_projected_points(k_init, b_init, real_points):
    k_init = round(k_init, 3)
    b_init = round(b_init, 3)
    projected_points = []
    for x, y in real_points:
        x_proj = x + (math.fabs(y - line(k_init, x, b_init)) * math.cos(1.57 - math.atan(k_init))) * math.sin(math.atan(k_init))
        y_proj = line(k_init, x, b_init) \
                 + (math.fabs(y - line(k_init, x, b_init)) * math.cos(1.57 - math.atan(k_init))) * math.cos(math.atan(k_init))
        projected_points.append((x_proj, y_proj))

    return projected_points


if __name__ == '__main__':
    x = np.linspace(0, 100, 100)
    random = np.random.random(100)
    y = np.linspace(0, 10, 100) + random

    stats(x, y)

    mean_point = (np.sum(x) / 100, np.sum(y) / 100)

    x = x - mean_point[0]
    y = y - mean_point[1]

    stats(x, y)

    real_points = np.array(list(zip(x, y)))

    k_init = 0
    b_init = 0

    k = []
    b = []
    error = []

    counter = 0

    while True:

        i = 0
        k_old = k_init
        k_init = 0
        while i <= 5000:
            gen_points = line(k_init, x, b_init)
            l_2 = l2(real_points, np.array(list(zip(x, gen_points))))
            k.append(k_init)
            b.append(b_init)
            error.append(l_2)
            k_init = k_init + 0.01
            i = i + 1

        error_np = np.array(error)

        index = np.where(error_np == np.amin(error_np))
        k_init = k[index[0][0]]
        k.clear()
        b.clear()
        error.clear()

        j = 0
        b_old = b_init
        b_init = 0
        while j <= 5000:
            gen_points = line(k_init, x, b_init)
            l_2 = l2(real_points, np.array(list(zip(x, gen_points))))
            k.append(k_init)
            b.append(b_init)
            error.append(l_2)
            b_init = b_init + 0.1
            j = j + 1

        error_np = np.array(error)
        index = np.where(error_np == np.amin(error_np))
        b_init = b[index[0][0]]

        k.clear()
        b.clear()
        error.clear()

        print(str(k_init) + " " + str(b_init))

        if ((b_init - b_old) == 0) & ((k_init - k_old) == 0):
            break


    print(str(k_init) + " " + str(b_init))


    projected_points = get_projected_points(k_init, b_init, real_points)

    print(projected_points)
    print(real_points)

    new_axis = []
    for k in range(len(projected_points)):
        sqrt = math.sqrt(projected_points[k][0] ** 2 + projected_points[k][1] ** 2)
        new_axis.append(((math.fabs(projected_points[k][0]) / sqrt)) * sqrt)

    print(new_axis)


