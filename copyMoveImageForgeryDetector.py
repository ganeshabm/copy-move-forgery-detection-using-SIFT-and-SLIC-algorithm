import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from collections import Counter
from skimage import segmentation

def readImage(image_name):
    return cv2.imread(str(image_name))

def showImage(image):
    image = imutils.resize(image, width=600)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def featureExtraction(img):
    image_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    # SLIC result
    global slic,oimage_with_keypoints,image_with_keypoints
    slic = segmentation.slic(image_lab, n_segments=20, start_label=1, compactness=6)
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    okp, odes = sift.detectAndCompute(gray, None)
    oimage_with_keypoints = cv2.drawKeypoints(img, okp, None, (0, 255, 0),
                                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # Initialize a list to store SIFT keypoints and descriptors
    keypoints = []
    descriptors = []
    feature_vectors = []
    labels = []

    for segment_id in np.unique(slic):
        # Create a mask for the current superpixel
        # print(segment_id)
        mask = (slic == segment_id)
        mean_lab = np.mean(image_lab[mask], axis=0)
        # print(image_lab[mask])
        # print(mean_lab)
        feature_vectors.append(mean_lab)
        labels.append(segment_id)

        mask = (slic == segment_id).astype('uint8')
        # Apply the mask to the original image
        superpixel = cv2.bitwise_and(img, img, mask=mask)

        # Convert the superpixel to grayscale (SIFT works on grayscale images)
        superpixel_gray = cv2.cvtColor(superpixel, cv2.COLOR_BGR2GRAY)

        # Detect SIFT keypoints and compute descriptors for the superpixel
        kp, des = sift.detectAndCompute(superpixel_gray, None)
        keypoints.extend(kp)
        descriptors.extend(des)
    descriptors = np.array(descriptors, dtype=np.float32)
    image_with_keypoints = cv2.drawKeypoints(img, keypoints, None, (0, 255, 0),
                                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return keypoints, descriptors


def featureMatching(keypoints, descriptors):

    #cv2.NORM_L2 is used since we are using the SIFT algorithm
    norm = cv2.NORM_L2
    #number of closest match we want to find for each descriptor
    k = 5

    #uses a brute force matcher(compare each descriptor of desc1, with each descriptor of desc2...)
    bf_matcher = cv2.BFMatcher(norm)
    #finds 10 closest matches for each desc in desc1 with desc in desc2
    matches = bf_matcher.knnMatch(descriptors, descriptors, k)

    #apply ratio test to get good matches (2nn test)
    ratio = 0.24
    good_matches_1 = []
    good_matches_2 = []

    for match in matches:
        k = 1   #ignore the first element in the matches array (distance to itself is always 0)

        while match[k].distance < ratio * match[k + 1].distance:  #d_i/d_(i+1) < T (ratio)
            k += 1

        for i in range(1, k):
            #just to ensure points are spatially separated
            if pdist(np.array([keypoints[match[i].queryIdx].pt, keypoints[match[i].trainIdx].pt]), "euclidean") > 10:
                good_matches_1.append(keypoints[match[i].queryIdx])
                good_matches_2.append(keypoints[match[i].trainIdx])

    points_1 = [match.pt for match in good_matches_1]
    points_2 = [match.pt for match in good_matches_2]

    if len(points_1) > 0:
        points = np.hstack((points_1, points_2))    #column bind
        unique_points = np.unique(points, axis=0)   #remove any duplicated points
        return np.float32(unique_points[:, 0:2]), np.float32(unique_points[:, 2:4])
    else:
        return None, None


def hierarchicalClustering(points_1, points_2, metric, threshold):

    points = np.vstack((points_1, points_2))        #vertically stack both sets of points (row bind)
    dist_matrix = pdist(points, metric='euclidean') #obtain condensed distance matrix (needed in linkage function)
    Z = hierarchy.linkage(dist_matrix, metric)

    #perform agglomerative hierarchical clustering
    cluster = hierarchy.fcluster(Z, t=threshold, criterion='inconsistent', depth=4)
    #filter outliers
    cluster, points = filterOutliers(cluster, points)

    n = int(np.shape(points)[0]/2)
    return cluster, points[:n], points[n:]


def filterOutliers(cluster, points):

    cluster_count = Counter(cluster)
    to_remove = []  # find clusters that does not have more than 3 points (remove them)
    for key in cluster_count:
        if cluster_count[key] <= 3:
            to_remove.append(key)

    indices = np.array([])   # find indices of points that corresponds to the cluster that needs to be removed

    for i in range(len(to_remove)):
        indices = np.concatenate([indices, np.where(cluster == to_remove[i])], axis=None)

    indices = indices.astype(int)
    indices = sorted(indices, reverse=True)

    for i in range(len(indices)):   # remove points that belong to each unwanted cluster
        points = np.delete(points, indices[i], axis=0)

    for i in range(len(to_remove)):  # remove unwanted clusters
        cluster = cluster[cluster != to_remove[i]]

    return cluster, points


def plotImage(img, p1, p2, C):
    fig = plt.figure(figsize=(10, 5))
    gs = gridspec.GridSpec(3, 2, width_ratios=[1, 1])

    # Define subplots
    ax1 = plt.subplot(gs[0, 0])  # This subplot will be larger in width
    ax2 = plt.subplot(gs[0, 1])
    ax3 = plt.subplot(gs[1, 0])  # This subplot will span both columns
    ax4 = plt.subplot(gs[1, 1])  # This subplot will span both columns
    ax5 = plt.subplot(gs[2, :])  # This subplot will span both columns

    ax1.set_xticks([])
    ax1.set_yticks([])
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax5.set_xticks([])
    ax5.set_yticks([])

    ax1.imshow(img)
    ax1.set_title('Original image')

    ax3.imshow(segmentation.mark_boundaries(img, slic))
    ax3.set_title('SLIC')

    ax2.imshow(oimage_with_keypoints)
    ax2.set_title('Original Image with Keypoints')

    ax4.imshow(image_with_keypoints)
    ax4.set_title('Slic Image with Keypoints')

    fig = plt.gcf()
    manager = plt.get_current_fig_manager()
    manager.window.wm_geometry("+10+50")
    ax5.imshow(img)

    colors = C[:np.shape(p1)[0]]
    ax5.scatter(p1[:, 0], p1[:, 1], c=colors, s=1)

    fig_ax5 = plt.figure(figsize=(10, 5))
    ax5_ax = fig_ax5.add_subplot(111)
    ax5_ax.imshow(img)
    ax5_ax.set_xticks([])
    ax5_ax.set_yticks([])

    for coord1, coord2 in zip(p1, p2):
        x1 = coord1[0]
        y1 = coord1[1]
        x2 = coord2[0]
        y2 = coord2[1]

        ax5.plot([x1, x2], [y1, y2], colors, linestyle="dotted")
        ax5_ax.plot([x1, x2], [y1, y2], colors, linestyle="dotted")

    # Save ax5_ax as a separate image
    fig_ax5.savefig("result.png", bbox_inches='tight', pad_inches=0)

    #plt.clf()
    plt.gca().set_xticks([])  # Remove x-axis tick labels
    plt.gca().set_yticks([])  # Remove y-axis tick labels
    plt.tight_layout()
    plt.show()


def detectCopyMove(image):
    kp, desc = featureExtraction(image)
    p1, p2 = featureMatching(kp, desc)
    # showImage(image)x

    if p1 is None:
        print("No tampering was found")
        return False

    clusters, p1, p2 = hierarchicalClustering(p1, p2, 'ward', 2.2)

    if len(clusters) == 0 or len(p1) == 0 or len(p2) == 0:
        print("No tampering was found")
        return False
    else:
        print("Tampered image found")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plotImage(image, p1, p2, clusters)
        return True
image=readImage("images/forged/DSC_1535tamp1.jpg")
detectCopyMove(image)