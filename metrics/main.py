import numpy as np
import pathlib


path = pathlib.Path(__file__).parent
matrix = np.zeros((2, 2))
with open(path / "responses.txt") as file:
    for row in file:
        row = row.strip()
        match row:
            case "positive positive":
                matrix[0, 0] += 1
            case "negative positive":
                matrix[0, 1] += 1
            case "positive negative":
                matrix[1, 0] += 1
            case "negative negative":
                matrix[1, 1] += 1

print(matrix)

acc = (matrix[0, 0] + matrix[1, 1]) / matrix.sum()
recall = matrix[0, 0] / matrix[:, 0].sum()
precision = matrix[0, 0] / matrix[0].sum()
f1 = 2*recall*precision / (recall + precision)

print(f"Acc: {acc:.4f}\nRecall: {recall:.4f}\nPrecision: {precision:.4f}\nF1: {f1:.4f}")