import os
import glob
import cv2
from matplotlib import pyplot as plt
import numpy as np
np.seterr(divide='ignore', invalid='ignore')

import sys
sys.setrecursionlimit(10**5)

# Rıdvan SIRMA
# 504181566

def read_all_image_files():
    current_working_directory = os.path.dirname(os.path.realpath(__file__))
    image_folder_name = "/bird images"
    images_directory = current_working_directory+image_folder_name

    all_image_files = glob.glob(images_directory + "/*")
    return all_image_files

def get_grayscale_image(image_file):
    image = cv2.imread(image_file)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def calculateHistogram(image):
    h = np.zeros((256,1), dtype=np.float128)  # histogram
    for g in range(0, 256):
        h[g, 0] = (image[:, :] == g).sum()
    return h #/ image.size

def Kmeans(image,clustersPoints):
    h = image.shape[0]
    w = image.shape[1]
    pointAndCluster = np.zeros([h, w, 2], dtype=np.uint8)
    oldClusterPoints = [0, 0]
    iteration = 0
    while (set(clustersPoints) != set(oldClusterPoints)):
        #print("iter: " + str(iteration) + "\n")
        iteration = iteration + 1
        for y in range(0, h):
            for x in range(0, w):
                point = image[y, x]
                closestIndexAndValue = min(enumerate(clustersPoints), key=lambda x: abs(x[1] - point))
                pointAndCluster[y, x] = (point, closestIndexAndValue[0])
        for i in range(0, len(clustersPoints)):
            list = []  # list of the points which are in the cluster i
            [list.append(pac[0]) for j in range(0, h) for pac in pointAndCluster[j] if pac[1] == i]
            mean = int(sum(list) / len(list))
            oldClusterPoints[i] = clustersPoints[i]
            clustersPoints[i] = mean
            #print("Cluster["+str(i)+"]: "+str(clustersPoints[i]))
    Ikmeans = image.copy()
    for y in range(0, h):
        for x in range(0, w):
            for i in range(0, len(clustersPoints)):
                if (pointAndCluster[y, x, 1] == i):
                    Ikmeans[y, x] = 1-i
    return Ikmeans
def GetValidNeighbours(x, y, binary_image):
    dx = [1, 0, -1,  0, -1, -1, 1,  1]
    dy = [0, 1,  0, -1,  1, -1, 1, -1]
    r, c = binary_image.shape
    result = []
    for k in range(8):
        nx, ny = x + dx[k], y + dy[k]
        if (nx >= 0 and nx < r and ny >= 0 and ny < c):
            result.append((nx,ny))
    return result


def ConnectedComponents(binary_image):
    x,y = binary_image.shape
    #print("Image shape: " , x, ", ", y)
    labeled_image = np.zeros((x,y), dtype=np.uint8)
    n=1;
    for i in range(0,x):
        for j in range(0,y):
            if(labeled_image[i,j] == 0 and binary_image[i,j] != 0):
                Label(i,j,n,binary_image,labeled_image)
                n = n + 1
    return labeled_image

def Label(x_start, y_start, n, binary_image, labeled_image):
    labeled_image[x_start,y_start] = n
    #print("Coordinate " , x_start, "and ", y_start, "labeled as", n)
    for point in GetValidNeighbours(x_start,y_start,binary_image):
        if(labeled_image[point] == 0 and binary_image[point] != 0):
            Label(point[0], point[1], n, binary_image, labeled_image)

for image_file in read_all_image_files():
    print("Objects are counting for image: " + image_file)

    grayscale_image = get_grayscale_image(image_file)
    plt.imshow(grayscale_image)
    plt.show()
    print(grayscale_image.shape)
    clustersPoints=[50,250]
    binary_image =Kmeans(grayscale_image,clustersPoints)
    print("K-Means applied to: " + image_file)
    plt.imshow(binary_image)
    plt.show()

    labeled_image = ConnectedComponents(binary_image)
    print("Image labeled: " + image_file)
    print("There are ", labeled_image.max(), "objects")
    plt.imshow(labeled_image)
    plt.show()


    #h = calculateHistogram(grayscale_image)
