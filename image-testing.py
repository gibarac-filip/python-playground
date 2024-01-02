import matplotlib.image as mpimg
import random
import matplotlib.pyplot as plt

def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip (v, w))

def sum_of_squares(v):
    return dot(v,v)

def scalar_multiply(c, v):
    return [c * v_i for v_i in v]

def vector_add(v, w):
    return [v_i + w_i for v_i, w_i in zip (v, w)]

def vector_subtract(v, w):
    return [v_i - w_i for v_i, w_i in zip (v, w)]

def squared_distance(v,w):
    return sum_of_squares(vector_subtract(v,w))

def vector_sum(vectors):
    result = vectors[0]
    for vector in vectors[1:]:
        result = vector_add(result, vector)
    return result

def vector_mean(vectors):
    n = len(vectors)
    return scalar_multiply(1/n, vector_sum(vectors))

class KMeans:
    def __init__(self, k):
        self.k = k
        self.means = None
    
    def classify(self, input):
        return min(range(self.k), key=lambda i: squared_distance(input, self.means[i]))
    
    def train(self, inputs):
        self.means = random.sample(inputs, self.k)
        assignments = None
        
        while True:
            new_assignments = list(map(self.classify, inputs))
            if assignments == new_assignments:
                return
            assignments = new_assignments
            for i in range(self.k):
                i_points = [p for p, a in zip(inputs, assignments) if a == i]
                if i_points:
                    self.means[i] = vector_mean(i_points)

path_to_file = 'image2.jpeg'
img = mpimg.imread(path_to_file)

pixels = [pixel for row in img for pixel in row]

# Reshape the image
reshaped_pixels = img.reshape(-1, 3)

# Normalize pixel values between 0 and 1
pixels = [list(pixel / 255) for pixel in reshaped_pixels]

clusterer = KMeans(5)
clusterer.train(pixels)

def recolor(pixel):
    cluster = clusterer.classify(pixel)
    return clusterer.means[cluster]

new_img = [[recolor(pixel) for pixel in row] for row in img]

plt.imshow(new_img)
plt.axis('off')
plt.show()