# %%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from skimage.filters import sobel
from scipy.spatial.transform import Rotation as R
from scipy.spatial import ConvexHull
import cv2  # For chain code approx
import os
import plotly.graph_objects as go
import glob
import re
import warnings
import pandas as pd
warnings.filterwarnings('ignore')

# Robust GLCM import with fallback
try:
    from skimage.measure import moments_central, moments_hu
    from skimage.feature import greycomatrix, greycoprops
    GLCM_AVAILABLE = True
except ImportError:
    print("GLCM not available; using histogram entropy fallback.")
    GLCM_AVAILABLE = False
    def greycomatrix(image, distances, angles, **kwargs):
        # Dummy fallback - not used
        return np.zeros((len(distances), image.shape[0], image.shape[1], len(angles)))
    def greycoprops(P, prop):
        # Fallback to simple entropy
        if prop == 'contrast':
            hist, _ = np.histogram(image.flatten(), bins=256, range=(0, 1))
            hist = hist / np.sum(hist)
            entropy = -np.sum(hist * np.log2(hist + 1e-8))
            return np.array([[entropy]])
        return np.zeros((1, 1))

def NaturalSortKey(file_name):
    return [int(x) if x.isdigit() else x for x in re.split(r'(\d+)', file_name)]

def ReadImages(images_path):
    """
    readImages creates an array of images to be worked on.
    Args:
        images_path: path to where images are stored, not forgetting image format
        example: ../../*.bmp or ../../*.jpg, * indicating any filename with the following format
    """  
    img_array = []
    print('Reading images ...')
    file_names = glob.glob(images_path)
    file_names.sort(key=NaturalSortKey)
    for filename in file_names:
        img = cv2.imread(filename)
        img_array.append(img)
    print('Images read successfully.')
    return img_array

def ReadFolderImages(folder_path):
    """
    ReadFolderImages creates an array of images to be worked on.
    Args:
        folder_path: path to where images are stored in folders within a folder, not forgetting image format
        example: ../../*.bmp or ../../*.jpg, * indicating any filename with the following format
    """  
    directories = os.listdir(folder_path)
    img_array = []
    print('Reading images ...')
    for folder in directories:
        image_name = folder_path+"/"+folder+"/*.jpg"
        file_names = glob.glob(image_name)
        file_names.sort(key=NaturalSortKey)
        counter = 0
        for file_name in file_names:
            img = cv2.imread(file_name)
            img_array.append(img)
            counter+=1
        print(f"Files in {folder} read into images array with {counter} number of image files")
    
    print('Images read successfully.')
    return img_array

def ConvertToGrey(img):
    """
    convertToGrey converts image from color to grayscale and returns a grayscale image.
    Args:
        img: image to be converted
    """ 
    blue, green, red = img[:,:,0], img[:,:,1], img[:,:,2] # separate array BGR colors
    img_gray = 0.0722*red + 0.7152*green + 0.2126*blue
    img_gray = np.uint8(img_gray) #convert to uint8 to be displayed 
    return img_gray

def BlobsData(image, thresh, number, view_name, folder):
    """
    BlobsData extracts the contours of the bubble from the image and returns the contour and the grayscale image.
    Args:
        image: image to be processed
        thresh: threshold value for binary segmentation
        number: image number for file naming
        view_name: name of the view (e.g., "front", "side")
        folder: path to the folder where the image will be saved
    """
    img1_gray = ConvertToGrey(image)
    plt.figure(figsize=(12, 8))
    plt.rcParams.update({'font.size': 18})
    filename = folder + "/bubble_"+view_name+"_"+str(number)+".png"
    # Extract blobs from image one
    ret,thresh1 = cv2.threshold(img1_gray,thresh,255,cv2.THRESH_BINARY_INV) # distinguishing the pixel intensity
    contours1, hierarchy = cv2.findContours(thresh1,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    bubble = max(contours1, key=cv2.contourArea)
    bubble_frame = cv2.drawContours(image, bubble, -1, (0,0,255), 1)
    plt.imshow(bubble_frame, aspect="equal", cmap='gray', origin="lower");
    plt.tick_params(axis='both', which='major', labelsize=26)
    plt.xlabel('Bubble Width (X-Axis)', fontsize=28)
    plt.ylabel('Bubble Height (Z-Axis)', fontsize=28)
    plt.savefig(filename)
    return bubble, img1_gray

# Robust GLCM import with fallback (as previous)
try:
    from skimage import measure
    from skimage.measure import moments_central, moments_hu
    from skimage.feature import greycomatrix, greycoprops
    GLCM_AVAILABLE = True
except ImportError:
    print("GLCM not available; using histogram entropy fallback.")
    GLCM_AVAILABLE = False
    # Fallback functions (as previous)

def extract_single_view_features_and_points(front_contour, mm_per_pixel):
    """
    Extract 25 features from single front view contour in OpenCV format (N,1,2) or (N,2).
    Handles the sample shape directly.
    Args:
        front_contour: contour in OpenCV format (N,1,2) or (N,2)
        mm_per_pixel: conversion factor from pixels to millimeters
    Returns: features (25,), points_2d (M,3) in mm.
    """
    # Flatten contour to (N,2) x,y in pixels
    if front_contour.ndim == 3 and front_contour.shape[1] == 1:
        contour_2d = front_contour.squeeze(axis=1).astype(float)  # (N,2)
    else:
        contour_2d = np.array(front_contour).astype(float)  # Flexible
    N = len(contour_2d)
    if N == 0:
        return np.zeros(25), np.zeros((100, 3))  # Fallback
    
    # Close contour if open
    if not np.allclose(contour_2d[0], contour_2d[-1]):
        contour_2d = np.vstack([contour_2d, contour_2d[0]])
    
    # Generate binary mask from contour using cv2.fillPoly
    pts = contour_2d.astype(np.int32).reshape(-1, 1, 2)  # OpenCV format
    h, w = int(np.max(contour_2d[:,1]) + 1), int(np.max(contour_2d[:,0]) + 1)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 1)  # Fill interior
    mask = cv2.resize(mask, (128, 128))  # Standardize for moments/GLCM
    
    # Label for regionprops
    labeled = measure.label(mask > 0)
    
    # Original 15
    area_px = np.sum(mask)
    # Perimeter using cv2.arcLength
    perimeter_px = cv2.arcLength(pts, closed=True)
    centroid = np.mean(contour_2d, axis=0)
    
    # Shape from moments
    M = moments_central(mask)
    mu20, mu02 = M[2,0], M[0,2]
    mu11 = M[1,1]
    majr = np.sqrt(4 * mu20 / np.pi) if mu20 > 0 else 1
    minr = np.sqrt(4 * mu02 / np.pi) if mu02 > 0 else 1
    if majr < minr:
        minor = majr 
        major = minr 
    else:
        major = majr 
        minor = minr 

    if major > 0 and minor != 0:
        eccentricity = np.sqrt(1 - (minor / major)**2) 
    else: 
        eccentricity = 0
    
    aspect_ratio = major / minor if minor > 0 else 1
    hull = ConvexHull(contour_2d)
    hull_area = hull.volume if hasattr(hull, 'volume') else area_px
    solidity = area_px / hull_area if hull_area > 0 else 1
    circularity = 4 * np.pi * area_px / (perimeter_px**2 + 1e-8)
    
    # Hu moments (7)
    hu = moments_hu(mask)
    hu_moments = np.log1p(np.abs(hu))[:7]
    
    orig_feats = np.array([
        area_px * mm_per_pixel**2, perimeter_px * mm_per_pixel, centroid[0]*mm_per_pixel, centroid[1]*mm_per_pixel,
        eccentricity, aspect_ratio, solidity, circularity,
        *hu_moments
    ])  # 15
    
    # New 10
    props = measure.regionprops(labeled)
    # Convexity
    convexity = area_px / hull_area if hull_area > 0 else 1
    
    # Equivalent diameter
    equiv_diam = np.sqrt(4 * area_px / np.pi) * mm_per_pixel
    
    # Roundness
    roundness = 4 * area_px / (np.pi * major**2)
    
    # Zernike (3)
    zernike_add = props[0].zernike_moments[:3].real if props and hasattr(props[0], 'zernike_moments') else np.zeros(3)
    
    # Fourier (3)
    diffs = np.diff(np.vstack([contour_2d, contour_2d[0]]), axis=0)
    angles = np.arctan2(diffs[:, 1], diffs[:, 0])
    chain_code = np.round((angles + np.pi) / (2 * np.pi / 8)).astype(int) % 8
    fft_coeffs = np.fft.fft(chain_code)[:3].real
    
    # Inertia ratio
    inertia_ratio = (minor / major)**2 if major > 0 else 1
    
    # Curvature variance
    curv = np.diff(angles, prepend=angles[-1])
    curv_var = np.var(curv)
    
    # Feret min/max
    feret_max = major * 2 * mm_per_pixel
    feret_min = minor * 2 * mm_per_pixel
    
    # Texture entropy
    if GLCM_AVAILABLE:
        glcm = greycomatrix(mask, [1], [0], symmetric=True, normed=True)
        entropy_proxy = greycoprops(glcm, 'contrast')[0, 0]
    else:
        hist, _ = np.histogram(mask.flatten(), bins=256, range=(0, 1))
        hist = hist / np.sum(hist)
        entropy_proxy = -np.sum(hist * np.log2(hist + 1e-8))
    
    new_feats = np.array([convexity, equiv_diam, roundness, *zernike_add, *fft_coeffs, 
                          inertia_ratio, curv_var, feret_max, feret_min, entropy_proxy])
    
    features = np.concatenate([orig_feats, new_feats])  # 25
    
    # Pseudo-points: Contour as (N, 3) with z=0, pad to 100
    points_2d = np.column_stack([contour_2d, np.zeros(len(contour_2d))]) * mm_per_pixel
    if len(points_2d) < 100:
        extra = np.random.choice(len(points_2d), 100 - len(points_2d), replace=True)
        points_2d = np.vstack([points_2d, points_2d[extra]])
    
    return features, points_2d

def StandardizeBubble(array1, array2):
    """
    StandardizeBubble resizes and equalizes the bubble sizes in two views to be the same.
    Args:
    array1: array of contours in view 1
    array2: array of contours in view 2 
    Returns:
        bubbles_1: list of standardized contours in view 1
        bubbles_2: list of standardized contours in view 2
    """
    print("Bubble synchronization in both views ...")
    bubbles_1 = []
    bubbles_2 = []
    for i in range(len(array1)):
        bubble1 = array1[i]
        bubble2 = array2[i]
        
        # Extract y-coordinates
        y1_coords = bubble1[:, 0, 1]
        y2_coords = bubble2[:, 0, 1]

        # Find max and min of bubble height
        max_y1 = np.max(y1_coords)
        min_y1 = np.min(y1_coords)
        max_y2 = np.max(y2_coords)
        min_y2 = np.min(y2_coords)

        # Find height difference between two bubble views
        height_y1 = max_y1 - min_y1
        height_y2 = max_y2 - min_y2

        # Find ratio 
        ratio = height_y1 / height_y2
        # Resize and equalize bubble sizes
        bubble2 = np.round(bubble2 * ratio).astype(int)

        # Extract y-coordinates
        y1_coords = bubble1[:, 0, 1]
        y2_coords = bubble2[:, 0, 1]
        
        # Find max and min of bubble height
        min_y1 = np.min(y1_coords)
        min_y2 = np.min(y2_coords)
        new_min_height_diff = min_y1 - min_y2           # Find start height difference  
        bubble2 = bubble2 + new_min_height_diff   

        # Equalize start Height
        bubbles_1.append(bubble1)
        bubbles_2.append(bubble2)

    print(f"Image frames synchronization done with a ratio of {ratio}")
    return bubbles_1, bubbles_2

def GetContours(array, thresh, view_name, folder):
    """getting_contours extracts the contours of the bubble from the image and returns the contour and the grayscale image.
    Args:
        array: image array
        thresh: threshold value
        view_name: name of the view
        folder: folder path
    Returns:
        bubbles: list of extracted contours
        gray_images: list of grayscale images
    """
    bubbles = []
    gray_images = []
    print(f"Extracting bubble boundaries from {view_name} images ...")
    for number in range(len(array)):
        bubble, grey_image = BlobsData(array[number], thresh, number= number, view_name=view_name, folder=folder)
        bubbles.append(bubble)
        gray_images.append(grey_image)
    print(f"Bubble boundaries in {view_name} extracted.")
    return bubbles, gray_images

def BubbleParameters(bubble):
    """BubbleParameters extracts 17 parameters from a single bubble contour in OpenCV format (N,1,2) or (N,2).
    Args:
        bubble: contour in OpenCV format (N,1,2) or (N,2)
        mm_per_pixel: conversion factor from pixels to millimeters
    Returns:
        parameters: list of 17 extracted parameters
    """
    # Extract four corners of both straight and slanted rectangles
    bubbles_parameters = []
    x,y,w,h = cv2.boundingRect(bubble)              # straight rectangle coordinates
    rect = cv2.minAreaRect(bubble)     
    box = cv2.boxPoints(rect)   
    box = np.round(box).astype(int)                 # tilted rectangle coordinates
    x1 = box[0][0]
    y1 = box[0][1]
    x2 = box[1][0]
    y2 = box[1][1]
    
    # Compute tilted angle
    if np.abs(y1 - y2) != 0:
        tilt_angle = np.degrees(np.arctan(np.abs(x2 - x1)/np.abs(y1 - y2)))
    else:
        tilt_angle = 0
    
    
    
    # area_px = cv2.contourArea(bubble)               # Calculate the area in pixels
    # Sort by the first column (index 0)
    # sorted_arr = arr[np.argsort(arr[:, 0, 0])]
    # Convert to DataFrame
    bubble_area = 0
    bubble_df = pd.DataFrame(bubble[:, 0, :], columns=['X Values', 'Y Values'])
    bubble_heights = bubble_df["Y Values"].unique()
    bubble_widths = bubble_df["X Values"].unique()
    for row in bubble_heights:
        group = bubble_df[bubble_df["Y Values"] == row]
        mini = group["X Values"].min()
        maxi = group["X Values"].max()
        line_width = maxi - mini
        bubble_area += line_width

    # Compute aspect ratio
    bubble_height = max(bubble_heights) - min(bubble_heights)
    bubble_width = max(bubble_widths) - min(bubble_widths)
    major_axis = max([bubble_width, bubble_height])
    minor_axis = min([bubble_width, bubble_height])
    aspect_ratio = major_axis/minor_axis
    equiv_diam_px = np.sqrt(4 * bubble_area / np.pi)    # Calculate the equivalent diameter in pixels
    
    # Compute area of bubble circle and ellipse, and difference between the two
    x_radius = bubble_width * 0.5
    y_radius = bubble_height * 0.5
    area_circle = np.pi * max([x_radius, y_radius])**2
    area_ellipse = np.pi * x_radius * y_radius
    circle_to_ellipse_diff = area_circle - area_ellipse
    circle_to_area_diff = area_circle - bubble_area
    ellipse_to_area_diff = area_ellipse - bubble_area


    # Extract central moments
    moments = cv2.moments(np.array(bubble))
    mu120 = moments['mu20']
    mu102 = moments['mu02']
    mu111 = moments['mu11']

    # Compute centroid
    centroid_x = int(mu120 / mu111)
    centroid_y = int(mu102 / mu111)

    # Compute orientation angle
    if (mu120 - mu102) != 0:                                # Avoid division by zero
        theta = 0.5 * np.arctan2(2 * mu111, mu120 - mu102)  # arctan2 handles quadrants
        theta_deg = np.degrees(theta)
    else:
        theta_deg = 90

    
    # Extract the height and width of bubble
    x_coords = bubble[:, 0, 0]
    y_coords = bubble[:, 0, 1]
    x_radius = x_coords.max() - x_coords.min()
    y_radius = y_coords.max() - y_coords.min()

    # Store bubble parameters
    parameters = [x, y, w, h, major_axis, minor_axis, aspect_ratio, tilt_angle, theta_deg, bubble_area, equiv_diam_px, centroid_x, 
                    centroid_y, area_circle, area_ellipse, circle_to_ellipse_diff,  circle_to_area_diff, ellipse_to_area_diff]
    for parameter in parameters:
        bubbles_parameters.append(parameter)

    return bubbles_parameters

def ArrayToDataframe(bubble_array):
    """
    ArrayToDataframe restructures bubble array into a dataframe with columns "X1-Values" and "Y1-Values". 
    Args:
        bubble_array: array of bubble contours
    Returns:
        plot_data: dataframe with columns "X1-Values" and "Y1-Values"
    """
    # Restructure bubble array into a dataframe
    plot_data = pd.DataFrame(columns=["X1-Values", "Y1-Values"])
    counter = 0
    for array in bubble_array:
        for arr in array:
            x1, y1 = arr
            plot_data.at[counter, "X1-Values"] = x1
            plot_data.at[counter, "Y1-Values"] = y1
        counter += 1

    return plot_data

def Reconstruction(dataframe1, dataframe2, counter, folder, mm_per_pixel):
    """Reconstruction reconstructs the 3D bubble from two views and calculates the volume and surface area of the bubble.
    Args:
        dataframe1: dataframe of bubble contours in view 1
        dataframe2: dataframe of bubble contours in view 2
        counter: image number for file naming
        folder: path to the folder where the image will be saved
        mm_per_pixel: mm_per_pixel used to standardize bubble sizes in two views  
    Returns: 
        X_data: list of x-coordinates of bubble contours in view 1  
        Y_data: list of y-coordinates of bubble contours in view 1
        Z_data: list of z-coordinates of bubble contours in view 1
        X1_data: list of x-coordinates of bubble contours in view 2
        Y1_data: list of y-coordinates of bubble contours in view 2
        volume: volume of the reconstructed bubble
        surface_area: surface area of the reconstructed bubble
    """
    height_data = dataframe1["Y1-Values"].unique()
    X_data = []
    Y_data = []
    Z_data = []
    volume = 0
    surface_area = 0
    prev_a = 0
    prev_b = 0
    graphs = []
    readings = 0
    for row in height_data:
        group1 = dataframe1[dataframe1["Y1-Values"] == row]
        group2 = dataframe2[dataframe2["Y1-Values"] == row]
        mini1 = group1["X1-Values"].min()
        maxi1 = group1["X1-Values"].max()
        mini2 = group2["X1-Values"].min()
        maxi2 = group2["X1-Values"].max()
        if pd.isna(mini1) or pd.isna(maxi1):
            mini1 = prev_min1
            maxi1 = prev_max1
        if pd.isna(mini2) or pd.isna(maxi2):
            mini2 = prev_min2
            maxi2 = prev_max2
        diameter1 = maxi1 - mini1
        diameter2 = maxi2 - mini2
        radius1 = diameter1 / 2
        radius2 = diameter2 / 2
        a = np.max([radius1, radius2])
        b = np.min([radius1, radius2])
        offset1 = maxi1 - radius1
        offset2 = maxi2 - radius2
        theta = np.linspace(0, 2 * np.pi, 100)
        x = (radius1 * np.cos(theta)) + offset1
        y = (radius2 * np.sin(theta)) + offset2
        z = row * np.ones(100)
        r_i = np.mean(np.sqrt(radius1**2*np.cos(theta)**2 + radius2**2*np.sin(theta)**2))
        x1 = (r_i * np.cos(theta)) + offset1
        y1 = (r_i * np.sin(theta)) + offset2
        vol = np.pi * radius1 * radius2
        volume = vol + volume
        if row == height_data[0] or row == height_data[-1]:
            surface_area = surface_area + vol
        else:
            group1 = dataframe1[dataframe1["Y1-Values"] == height_data[readings+1]]
            group2 = dataframe2[dataframe2["Y1-Values"] == height_data[readings+1]]
            next_mini1 = group1["X1-Values"].min()
            next_maxi1 = group1["X1-Values"].max()
            next_mini2 = group2["X1-Values"].min()
            next_maxi2 = group2["X1-Values"].max()
            if pd.isna(next_mini1) or pd.isna(next_maxi1):
                next_mini1 = prev_min1
                next_maxi1 = prev_max1
            if pd.isna(next_mini2) or pd.isna(next_maxi2):
                next_mini2 = prev_min2
                next_maxi2 = prev_max2
            next_diameter1 = next_maxi1 - next_mini1
            next_diameter2 = next_maxi2 - next_mini2
            next_radius1 = next_diameter1 / 2
            next_radius2 = next_diameter2 / 2
            next_a = np.max([next_radius1, next_radius2])
            next_b = np.min([next_radius1, next_radius2])
            func = np.sqrt((a**2 + b**2)/2)
            func_prev = np.sqrt((prev_a**2 + prev_b**2)/2)
            func_next = np.sqrt((next_a**2 + next_b**2)/2)
            func_1 = (func_next - func_prev)/2
            surface = 2 * np.pi * func * (np.sqrt(1+(func_1)**2))
            surface_area = surface + surface_area
        if readings == 0:
            figure = go.Scatter3d(
            x = x,
            y = y,
            z = z,
            marker=dict(size=5, color='blue', opacity=0.4),
            name = "Enhanced"
            )
        else: 
            figure = go.Scatter3d(
            x = x,
            y = y,
            z = z,
            marker=dict(size=5, color='blue', opacity=0.5),
            name = "Enhanced",
            showlegend=False
            )

        graphs.append(figure)
        X_data.append(x)
        Y_data.append(y)
        Z_data.append(z)
        readings += 1
        prev_min1 = mini1
        prev_max1 = maxi1
        prev_min2 = mini2
        prev_max2 = maxi2
        prev_a = a
        prev_b = b 
    fig = go.Figure()
    fig.add_traces(graphs)
    fig.update_layout(height = 1200, width = 1600, 
                      scene= dict(xaxis=dict(title="x-axis", tickfont_size=18, title_font_size=40), 
                      yaxis=dict(title="y-axis", tickfont_size=18, title_font_size=40), 
                      zaxis=dict(title="z-axis", tickfont_size=18, title_font_size=40)),
                      margin=dict(l=120, r=120, t=120, b=120), legend=dict(font=dict(size=40)))
    file_name = folder + "/fig" + str(counter) + ".html"
    fig.write_html(file_name)
    fig.show()


    volume *= mm_per_pixel**3    
    surface_area *= mm_per_pixel**2
    return [X_data, Y_data, Z_data], volume, surface_area

def ReconstructionData(array1, array2, mm_per_pixel, folder):
    print("Reconstruction started ...")
    Point_Cloud =[]
    Volume = []
    Surface_Area = []
    for i in range(len(array1)):
        data1 = array1[i]
        data2 = array2[i]
        data1 = ArrayToDataframe(data1)
        data2 = ArrayToDataframe(data2)
        point_cloud, volume, surface = Reconstruction(data1, data2, i, folder, mm_per_pixel)
        Point_Cloud.append(point_cloud)
        Volume.append(volume)
        Surface_Area.append(surface)
    print("Reconstruction done!")
    return Point_Cloud, Volume, Surface_Area

def SaveData(dataframe, folder):
    """
    SaveData saves bubble cordinates to an excel sheet from a given set of images.
    Args:
        dataframe: dataframe to be saved
        folder: path to folder where excel sheet will be saved

    """
    print("Saving to excel started ...")
    results_path = folder + ".xlsx"
    dataframe.to_excel(results_path)
    print("Save to excel successful.")

    return 

def SplitViews(img_array): 
    """
    SplitViews splits the images into two views and returns two arrays of images for each view.    
    Args:
        img_array: array of images
    Returns:
        view_1: array of images for view 1
        view_2: array of images for view 2
    """   
    counter = 0
    view_1 = []
    view_2 = []
    for view in img_array:
        if counter % 2 == 0:
            view_1.append(view)
        else:
            view_2.append(view)
        counter += 1
    return view_1, view_2

def get_reconstructed_boundary(array1, array2, array3):
    """get_reconstructed_boundary extracts the boundary of the reconstructed bubble from two views and returns the boundary in 3D coordinates.  
    Args:
        array1: array of contours in view 1
        array2: array of contours in view 2
        array3: array of z-coordinates of bubble contours in view 1 
    Returns:    
        X: array of x-coordinates of bubble contours in view 1  
        Y: array of y-coordinates of bubble contours in view 1
    """
    X = []
    Y = []
    for i in range(len(array1)):
        x1 = np.min(array1[i])
        x2 = np.max(array1[i])
        y1 = np.min(array2[i])
        y2 = np.max(array2[i])
        z = np.min(array3[i])
        X.append([int(x1), int(z)])
        X.append([int(x2), int(z)])
        Y.append([int(y1), int(z)])
        Y.append([int(y2), int(z)])
    X = np.array(X).reshape((-1, 1, 2))
    Y = np.array(Y).reshape((-1, 1, 2))
    return X, Y

# def extract_bubble_features_from_contour(contour, gray_image=None, pixel_size_mm=1.0,
#                                          compute_exact_feret_min=False):
#     """
#     Extract 12 significant features from a bubble's contour points (closed polygon).
#     Handles missing 'feret_diameter_min' by approximation or exact computation.
    
#     Args:
#         contour (np.ndarray): Nx2 array of [row, col] points (y, x). Shape (N, 2).
#         gray_image (np.ndarray, optional): 2D grayscale image aligned with contour coords.
#         pixel_size_mm (float): Pixel size in mm (e.g., 0.074).
#         compute_exact_feret_min (bool): If True, compute exact min Feret (slower; for convex contours).
    
#     Returns:
#         dict: 12 features (core + added). Intensity features added if gray_image provided.
#     """
#     contour = np.asarray(contour, dtype=float)  # Use float for exact Feret
#     if contour.ndim != 2 or contour.shape[1] != 2:
#         raise ValueError(f"Contour must be Nx2 [row, col], got shape {contour.shape}")
    
#     # Close contour if not already
#     if not np.allclose(contour[0], contour[-1]):
#         contour = np.vstack([contour, contour[0]])
    
#     # Bounding box for mask (cast to int for indexing/reshape)
#     rows = contour[:, 0]
#     cols = contour[:, 1]
#     ymin = int(rows.min())
#     ymax = int(rows.max())
#     xmin = int(cols.min())
#     xmax = int(cols.max())
#     height = ymax - ymin + 1
#     width = xmax - xmin + 1
    
#     # Create filled binary mask using matplotlib Path
#     path = mpath.Path(np.column_stack([cols, rows]))  # (x=col, y=row)
#     yy, xx = np.mgrid[ymin:ymax+1, xmin:xmax+1]
#     points = np.column_stack([xx.ravel(), yy.ravel()])
#     mask_flat = path.contains_points(points)
#     binary_mask = mask_flat.reshape(height, width).astype(bool)
    
#     # Label and get props for largest region
#     labeled = measure.label(binary_mask)
#     regions = measure.regionprops(labeled)
#     if not regions:
#         raise ValueError("No regions found from contour.")
#     props = max(regions, key=lambda r: r.area)
    
#     # Core size features (scale to mm)
#     A = props.area * (pixel_size_mm ** 2)
#     d_eq = props.equivalent_diameter_area * pixel_size_mm
#     d_maj = props.major_axis_length * pixel_size_mm
#     d_min = props.minor_axis_length * pixel_size_mm
    
#     # Core shape features
#     AR = d_maj / d_min
#     P = props.perimeter * pixel_size_mm
#     C = (4 * np.pi * A) / (P ** 2) if P > 0 else 0
#     E = props.eccentricity
    
#     # Additional significant features
#     S = props.solidity  # 0–1
#     El = 1 - (props.minor_axis_length / props.major_axis_length)  # 0–1
    
#     # Feret Diameter Ratio (FDR): max / min
#     feret_max = props.feret_diameter_max * pixel_size_mm
#     if compute_exact_feret_min:
#         # Exact min Feret: Min distance between parallel supporting lines (slow, O(N^2))
#         # Rotate contour to find min width (projected onto perpendicular directions)
#         from scipy.spatial.distance import cdist
#         angles = np.linspace(0, np.pi, 180, endpoint=False)  # 1° steps
#         min_feret = np.inf
#         for theta in angles:
#             rot = np.array([[np.cos(theta), -np.sin(theta)],
#                             [np.sin(theta), np.cos(theta)]])
#             proj = np.dot(contour - contour.mean(axis=0), rot[:, 0])  # Project onto direction
#             min_feret = min(min_feret, proj.max() - proj.min())
#         feret_min = min_feret * pixel_size_mm
#     else:
#         # Approximation: Use minor_axis_length (fast, good for ellipsoidal bubbles)
#         feret_min = d_min
#     FDR = feret_max / feret_min if feret_min > 0 else 1.0
    
#     # Hu1: First Hu moment (invariant)
#     Hu1 = props.moments_hu[0]  # Correct attribute
    
#     # Inertia Ratio (IR): Minor / Major eigenvalue
#     eigvals = props.inertia_tensor_eigvals
#     IR = eigvals[1] / eigvals[0] if eigvals[0] != 0 else 1.0
    
#     features = {
#         'A': A, 'd_eq': d_eq, 'd_maj': d_maj, 'd_min': d_min,
#         'AR': AR, 'C': C, 'E': E,
#         'S': S, 'El': El, 'FDR': FDR, 'Hu1': Hu1, 'IR': IR
#     }
    
#     # Intensity and edge features (if gray_image provided; crop to bbox)
#     if gray_image is not None:
#         y_slice = slice(ymin, ymax + 1)
#         x_slice = slice(xmin, xmax + 1)
#         gray_crop = gray_image[y_slice, x_slice]
#         rel_coords = props.coords
#         intensities = gray_crop[rel_coords[:, 0], rel_coords[:, 1]]
#         features['mu_I'] = np.mean(intensities)
#         features['sigma_I2'] = np.var(intensities)
#         grad_mag = sobel(gray_crop)
#         edge_vals = grad_mag[rel_coords[:, 0], rel_coords[:, 1]]
#         features['edge_mean'] = np.mean(edge_vals)
    
#     return features

def extract_bubble_features_from_contour(contour, gray_image=None, pixel_size_mm=1.0,
                                         compute_exact_feret_min=False):
    """
    Extract 15 significant features from a bubble's contour points (closed polygon).
    Handles missing 'feret_diameter_min' by approximation or exact computation.
    
    Args:
        contour (np.ndarray): Nx2 array of [row, col] points (y, x). Shape (N, 2).
        gray_image (np.ndarray, optional): 2D grayscale image aligned with contour coords.
        pixel_size_mm (float): Pixel size in mm (e.g., 0.074).
        compute_exact_feret_min (bool): If True, compute exact min Feret (slower; for convex contours).
    
    Returns:
        dict: 15 features (core + added). Intensity features added if gray_image provided.
    """
    contour = np.asarray(contour, dtype=float)  # Use float for exact Feret
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError(f"Contour must be Nx2 [row, col], got shape {contour.shape}. "
                         f"Pass full array (e.g., view_1_bubbles[2]), not [:,0].")
    
    # Close contour if not already
    if not np.allclose(contour[0], contour[-1]):
        contour = np.vstack([contour, contour[0]])
    
    # Bounding box for mask (strict int casting)
    rows = contour[:, 0]
    cols = contour[:, 1]
    ymin = abs(int(np.floor(rows.min())))
    ymax = abs(int(np.ceil(rows.max())))
    xmin = abs(int(np.floor(cols.min())))
    xmax = abs(int(np.ceil(cols.max())))
    height = ymax - ymin + 1
    width = xmax - xmin + 1
    
    # Create filled binary mask using matplotlib Path
    path = mpath.Path(np.column_stack([cols, rows]))  # (x=col, y=row)
    yy, xx = np.mgrid[ymin:ymax+1, xmin:xmax+1]
    points = np.column_stack([xx.ravel(), yy.ravel()])
    mask_flat = path.contains_points(points)
    binary_mask = mask_flat.reshape(height, width).astype(bool)
    
    # Label and get props for largest region
    labeled = measure.label(binary_mask)
    regions = measure.regionprops(labeled)
    if not regions:
        raise ValueError("No regions found from contour.")
    props = max(regions, key=lambda r: r.area)
    
    # Core size features (scale to mm)
    A = props.area * (pixel_size_mm ** 2)
    d_eq = props.equivalent_diameter_area * pixel_size_mm
    d_maj = props.major_axis_length * pixel_size_mm
    d_min = props.minor_axis_length * pixel_size_mm
    
    # Core shape features
    AR = d_maj / d_min
    P = props.perimeter * pixel_size_mm
    C = (4 * np.pi * A) / (P ** 2) if P > 0 else 0
    E = props.eccentricity
    
    # Additional significant features
    S = props.solidity  # 0–1
    El = 1 - (props.minor_axis_length / props.major_axis_length)  # 0–1
    
    # Feret Diameter Ratio (FDR): max / min
    feret_max = props.feret_diameter_max * pixel_size_mm
    if compute_exact_feret_min:
        # Exact min Feret: Min distance between parallel supporting lines (slow, O(N^2))
        angles = np.linspace(0, np.pi, 180, endpoint=False)  # 1° steps
        min_feret = np.inf
        for theta in angles:
            rot = np.array([[np.cos(theta), -np.sin(theta)],
                            [np.sin(theta), np.cos(theta)]])
            proj = np.dot(contour - contour.mean(axis=0), rot[:, 0])  # Project onto direction
            min_feret = min(min_feret, proj.max() - proj.min())
        feret_min = min_feret * pixel_size_mm
    else:
        # Approximation: Use minor_axis_length (fast, good for ellipsoidal bubbles)
        feret_min = d_min
    FDR = feret_max / feret_min if feret_min > 0 else 1.0
    
    # Hu1: First Hu moment (invariant)
    Hu1 = props.moments_hu[0]  # Correct attribute
    
    # Inertia Ratio (IR): Minor / Major eigenvalue
    eigvals = props.inertia_tensor_eigvals
    IR = eigvals[1] / eigvals[0] if eigvals[0] != 0 else 1.0
    
    features = {
        'A': A, 'd_eq': d_eq, 'd_maj': d_maj, 'd_min': d_min,
        'AR': AR, 'C': C, 'E': E,
        'S': S, 'El': El, 'FDR': FDR, 'Hu1': Hu1, 'IR': IR
    }
    
    # Intensity and edge features (if gray_image provided; crop to bbox)
    if gray_image is not None:
        #print("True", gray_image)
        # Ensure gray_image is 2D array
        if gray_image.ndim != 2:
            raise ValueError("gray_image must be 2D (H x W).")
        y_slice = slice(ymin, ymax + 1)
        x_slice = slice(xmin, xmax + 1)
        gray_crop = gray_image[y_slice, x_slice]

        #print(f"gray_crop y{ymin, ymax} and x{xmin, xmax}", gray_crop)
        # Get relative coords and CLIP to prevent out-of-bounds (fixes IndexError)
        rel_coords = props.coords.astype(int)
        rel_coords[:, 0] = np.clip(rel_coords[:, 0], 0, gray_crop.shape[0] - 1)
        rel_coords[:, 1] = np.clip(rel_coords[:, 1], 0, gray_crop.shape[1] - 1)
        
        # Now safe indexing
        intensities = gray_crop[rel_coords[:, 0], rel_coords[:, 1]]
        features['mu_I'] = np.mean(intensities)
        features['sigma_I2'] = np.var(intensities)
        
        # Edge strength: Mean Sobel gradient in region
        grad_mag = sobel(gray_crop)
        edge_vals = grad_mag[rel_coords[:, 0], rel_coords[:, 1]]
        features['edge_mean'] = np.mean(edge_vals)
    
    return features

def FeaturesExtraction(data1, data2, pixel2mm):
    """
    FeaturesExtraction extracts features from the contours of the bubbles in two views and returns a list of features for each bubble.
    Args:
        data1: List of contours for the first view.
        data2: List of contours for the second view.
        pixel2mm: Conversion factor from pixels to millimeters.
    Returns:
        List of dictionaries containing extracted features for each bubble.
    """
    print("Features extraction started ...")
    Features = []
    for i in range(len(data1)):
        # print(i)
        features = extract_bubble_features_from_contour(data1[i][:,0], data2[i], pixel_size_mm=pixel2mm)
        Features.append(features)
    print("Features extraction ended.")
    return Features

def CreateFolder(folder_path):
    """
    CreateFolder creates a folder at the specified path if it does not already exist.
    """
    # Create the directory, skip if it exists
    try:
        os.makedirs(folder_path, exist_ok=True)
        print(f"Folder '{folder_path}' ensured to exist.")
    except OSError as e:
        # Handle other potential OS errors (e.g., permission denied, invalid path)
        print(f"Error creating directory: {e}")
