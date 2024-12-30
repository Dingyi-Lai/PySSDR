import os
import numpy as np

save_path = os.path.join(os.environ["TMPDIR"], "output")
os.makedirs(save_path, exist_ok=True)

X = np.linspace(0, 1, 28)
np.save(os.path.join(save_path, "test.npy"), X)