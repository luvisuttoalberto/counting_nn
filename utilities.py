import numpy as np


def square(i, j, size):
    square = []
    for x in range(i, i + size):
        for y in range(j, j + size):
            square.append((x, y))
    return square


def neighbours(i, j, width, size):
    neighbours = []
    for x in range(i - 1, i + size + 1):
        for y in range(j - 1, j + size + 1):
            if (x, y) not in square(i, j, size) and 0 <= x < width and 0 <= y < width:
                neighbours.append((x, y))
    return neighbours


def generator(n_images=1000, width=28, n_max=10, size_min=3, size_max=5, seed=42, fixed_n=False, n=None, method='classic', scale=30):
    np.random.seed(seed)
    images = np.empty((n_images, width, width))
    counts = np.empty(n_images, 'i')
    for idx_image in range(n_images):
        image = np.zeros((width, width))
        if not fixed_n:
            n = np.random.randint(1, n_max + 1)
        else:
            if n is None:
                raise ValueError('If fixed_n is True then n must be set by the user')
        counts[idx_image] = n - 1
        for idx_square in range(n):
            size = np.random.randint(size_min, size_max + 1)
            i, j = np.random.randint(0, width - size + 1, 2)
            restart = True
            while restart:
                if np.any(image[tuple(zip(*square(i, j, size)))]) == False \
                        and np.any(image[tuple(zip(*neighbours(i, j, width, size)))]) == False:
                    image[tuple(zip(*square(i, j, size)))] = 255
                    restart = False
                else:
                    i, j = np.random.randint(0, width - size + 1, 2)
        images[idx_image] = image
    if method == 'gauss':
        for idx_image in range(n_images):
            gauss_noise = np.random.normal(0, scale, (width, width))
            images[idx_image] = np.clip(images[idx_image] + gauss_noise, 0, 255)
    if method == 'inverted':
        for idx_image in range(n_images):
            images[idx_image] = 255*np.ones((width, width)) - images[idx_image]
    return images, counts


