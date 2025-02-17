import numpy as np
import pathlib
import matplotlib.pyplot as plt


path = pathlib.Path(__file__).parent
outpath = path / "images"
outpath.mkdir(exist_ok=True)

width, height = 100, 100
sq_width, sq_height = 5, 5

coords = []

for i in range(1, height-sq_height, sq_height+1):
    for j in range(1, width-sq_width, sq_width+1):
        image = np.zeros((width, height), dtype="uint8")
        image[i:i+sq_height, j:j+sq_width] = 1
        np.save(outpath / f"{i}_{j}.npy", image)
        
        coords += [f"{i} {j}"]
        
with open(path / "coords.txt", 'w') as file:
    file.write('\n'.join(coords))
    file.close()