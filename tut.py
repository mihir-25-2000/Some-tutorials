import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

ds = xr.tutorial.load_dataset("air_temperature")
test_plot1=ds['air'].isel(time=0).plot()
plt.savefig("/home/mihir.more/myenv/test_plot1.png")
print("Random image saved as test_plot1.png")

zarr_path = "/Datastorage/saptarishi.dhanuka_asp25/20230101_20240926_imerg_era5.zarr"
ds=xr.open_zarr(zarr_path, consolidated=True)
print(ds)
print("test")


# Some jax tutorial code below


import jax
import jax.numpy as jnp
from jax import random, grad, jit

# ----- 1) Generate toy 2D data (two Gaussian blobs) -----
key = random.PRNGKey(0)
n_per_class = 250

key1, key2, key3 = random.split(key, 3)
mean0 = jnp.array([-1.0, -1.0])
mean1 = jnp.array([+1.0, +1.0])

X0 = mean0 + 0.6 * random.normal(key1, (n_per_class, 2))
X1 = mean1 + 0.6 * random.normal(key2, (n_per_class, 2))
X = jnp.concatenate([X0, X1], axis=0)                             # (N, 2)
y = jnp.concatenate([jnp.zeros(n_per_class), jnp.ones(n_per_class)])  # labels in {0,1}

# Shuffle
perm = random.permutation(key3, X.shape[0])
X, y = X[perm], y[perm]

# ----- 2) Model & loss -----
def logits(params, X):
    w, b = params
    return X @ w + b  # (N,)

# Binary cross-entropy via softplus for stability:
# BCE = mean( softplus(logit) - y*logit )
def loss_fn(params, X, y, l2=0.0):
    z = logits(params, X)
    data_loss = jnp.mean(jnp.logaddexp(0.0, z) - y * z)
    reg = 0.5 * l2 * (jnp.sum(params[0] ** 2) + params[1] ** 2)
    return data_loss + reg

# ----- 3) Gradients (jax.grad) -----
loss_grad = grad(loss_fn)

# ----- 4) Gradient descent loop -----
def accuracy(params, X, y):
    z = logits(params, X)
    yhat = (z > 0).astype(y.dtype)
    return jnp.mean(yhat == y)

@jit
def step(params, X, y, lr, l2):
    g_w, g_b = loss_grad(params, X, y, l2)
    w, b = params
    w = w - lr * g_w
    b = b - lr * g_b
    return (w, b)

# Init
params = (jnp.zeros(2), 0.0)   # (w, b)
lr = 0.5
l2 = 1e-3
num_steps = 400

for t in range(1, num_steps + 1):
    params = step(params, X, y, lr, l2)
    if t % 50 == 0:
        current_loss = loss_fn(params, X, y, l2)
        acc = accuracy(params, X, y)
        print(f"step {t:3d} | loss {current_loss:.4f} | acc {acc*100:.1f}%")

w, b = params
print("\nFinal params:")
print("w:", w, " b:", float(b))
print("Final train accuracy:", float(accuracy(params, X, y)))
