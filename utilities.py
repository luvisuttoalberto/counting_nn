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

def generator(n_images=1000, width=28, n_max=10, size_min=3, size_max=5, seed=42):
    np.random.seed(seed)
    images = np.empty((n_images, width, width))
    counts = np.empty((n_images), 'i')
    for idx_image in range(n_images):
        image = np.zeros((width, width))
        n = np.random.randint(1, n_max + 1)
        counts[idx_image] = n
        for idx_square in range(n):
            size = np.random.randint(size_min, size_max + 1)
            i, j = np.random.randint(0, width - size + 1, 2)
            restart = True
            while restart == True:
                if np.any(image[tuple(zip(*square(i, j, size)))]) == False \
                        and np.any(image[tuple(zip(*neighbours(i, j, width, size)))]) == False:
                    image[tuple(zip(*square(i, j, size)))] = 1
                    restart = False
                else:
                    i, j = np.random.randint(0, width - size + 1, 2)
        images[idx_image] = image
    return images, counts