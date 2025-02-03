import cv2
import numpy as np
import matplotlib.pyplot as plt
# np.random.seed(20)


n = 20
xk1 = 100 + np.random.randint(-25, 25, n)
yk1 = 100 + np.random.randint(-25, 25, n)

xk2 = 150 + np.random.randint(-25, 25, n)
yk2 = 150 + np.random.randint(-25, 25, n)

rk1 = np.repeat(1, n)
rk2 = np.repeat(2, n)

new_point = (124, 124)

knn = cv2.ml.KNearest_create()
train = np.stack([np.hstack([xk1, xk2]), np.hstack([yk1, yk2])]).T.astype("f4")
responses = np.hstack([rk1, rk2]).reshape(-1, 1).astype("f4")

print(train.shape, responses.shape)

knn.train(train, cv2.ml.ROW_SAMPLE, responses)

ret, results, neighbours, dist = knn.findNearest(np.array(new_point).astype("f4").reshape(1, 2), 3)

print(ret, results, neighbours, dist)

plt.scatter(xk1, yk1, 80, c='r')
plt.scatter(xk2, yk2, 80, c='b')
plt.scatter(*new_point, 120, c=['r', 'b'][int(ret)-1], marker='^')
plt.title(f"Green from {int(ret)}")
plt.show()
