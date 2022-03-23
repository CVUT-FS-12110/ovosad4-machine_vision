import numpy as np
import skimage.measure
import cv2
import matplotlib.pyplot as plt

# img = cv2.imread('data/troja1/picture_1647527041.4399743.tiff')
img = cv2.imread('data/troja1/picture_1647527064.8455985.tiff')


img = cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)

img = cv2.equalizeHist(img[:,:,1])
# img = cv2.blur(img, (10, 10))

# img = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3)

# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
# img = clahe.apply(img[:,:,1])


# display_img = img
# plt.imshow(display_img)
# plt.show()



img = img[3500:4000,:]

# trunk = display_img.mean(axis=0)

n = img.shape[1]
step = 5
trunk = np.zeros(n)
for k in range(step, n):
    A = img[:, k:k+step]
    # trunk[k] = skimage.measure.shannon_entropy(A)
    trunk[k] = A.mean()

display_img = img

plt.subplot(211)
plt.imshow(display_img)

plt.subplot(212)
plt.plot(trunk)
plt.xlim(0, len(trunk))
# plt.ylim(4, 6)

plt.show()
