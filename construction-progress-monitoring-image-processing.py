# Author: Wong Tian Jie

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import cv2
import numpy as np
from matplotlib import pyplot as pt
from skimage.metrics import structural_similarity as ssim

# Function to resize an image using OpenCV's resize function with the option to specify a specific width or height
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    # Initialise the dimension (desired size) to resize the image into
    dim = None
    # Get the current dimension of the image
    (h, w) = image.shape[:2]

    # If the height and width parameter are both not specified, no resizing will be done, the original image is returned
    if width is None and height is None:
        return image
    # If the height parameter is specified and the width parameter is not, the new width of the resized image will be based on the height parameter specified (to retain its original aspect ratio)
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    # If the width parameter is specified and the height parameter is not, the new height of the resized image will be based on the width parameter specified (to retain its original aspect ratio)
    # If both width and height parameter are specified, the new height of the resized image will be based on the width parameter specified instead of the height parameter specified (to retain its original aspect ratio)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    # Resize the image into the new size and return the resized image
    return cv2.resize(image, dim, interpolation=inter)

# Function to perform image stitching on two images based on a homography matrix and identify the pixels occupied by each of the images in the stitched image
def stitchImages(img1, img2, H):

    # Get height and width of first and second image
    rows1, cols1 = img1.shape[:2]
    rows2, cols2 = img2.shape[:2]

    # List of locations of the four corners in the first image
    list_of_points_1 = np.float32([[0,0], [0, rows1],[cols1, rows1], [cols1, 0]]).reshape(-1, 1, 2) # Reshaped into an array of four 2x1 arrays
    # List of initial locations of the four corners in the second image
    temp_points = np.float32([[0,0], [0,rows2], [cols2,rows2], [cols2,0]]).reshape(-1,1,2) # Reshaped into an array of four 2x1 arrays

    # Transform the four corners of the second image using the homography matrix to get the new locations of the four corners after image transformation
    list_of_points_2 = cv2.perspectiveTransform(temp_points, H)

    # Append the new locations of second image's four corners to the list with the locations of the first image's four corners
    list_of_points = np.concatenate((list_of_points_1,list_of_points_2), axis=0)

    # Get the minimum x-coordinate (row) and minimum y-coordinate (column) among all the corners in the concatenated list of corners and minus 0.5 from them before truncating to int
    [x_min, y_min] = np.int32(list_of_points.min(axis=0).ravel() - 0.5)
    # Get the maximum x-coordinate (row) and maximum y-coordinate (column) among all the corners in the concatenated list of corners and add 0.5 from them before truncating to int
    [x_max, y_max] = np.int32(list_of_points.max(axis=0).ravel() + 0.5)

    # Offsets for x-axis and y-axis to be used in image translation
    translation_dist = [-x_min,-y_min]

    # Translation matrix with the x-axis and y-axis offsets
    H_translation = np.array([[1, 0, translation_dist[0]], [0, 1, translation_dist[1]], [0, 0, 1]])

    # Multiply the homography and translation matrices to combine them
    # First perform image transformation on the second image with the homography matrix, then perform image translation on it again with the translation matrix
    stitched_img = cv2.warpPerspective(img2, H_translation.dot(H), (x_max-x_min, y_max-y_min)) # Dimensions for stitched image is based on the offsets

    # Get the transformed and translated second image to extract pixels occupied by second image in stitched image
    img2only = stitched_img.copy()
    stitched_img[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1

    # Transform and translate a binary image filled with pixel values of 1 which have the same size as the second image
    # This gives us the binary map which indicates pixels occupied by the second image in the stitched image
    img2_binary = np.ones(img2.shape[:2])
    img2_binary = cv2.warpPerspective(img2_binary, H_translation.dot(H), (x_max-x_min, y_max-y_min))

    # Only Image Translation is applied on first image, performing image stitching by overlaying first image on top of second image
    img1only = np.zeros((y_max-y_min, x_max-x_min, 3), dtype=np.uint8)
    img1only[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = img1

    # Fill a binary map with values of 1 at locations occupied by pixels of first image after applying image translation on it
    # This gives us the binary map which indicates pixels occupied by the first image in the stitched image
    img1_binary = np.zeros((y_max-y_min, x_max-x_min))
    img1_binary[translation_dist[1]:rows1+translation_dist[1], translation_dist[0]:cols1+translation_dist[0]] = 1

    # Return the stitched image,two images with each image in their respective locations in the stitched image only, and the two binary maps
    return stitched_img, img1_binary,  img1only, img2_binary, img2only

# Function to align two images based on feature matching and obtain the overlapped parts from them
def imageAlignment(img1, img2, surfHessianThreshold, goodMatchPercent):

    # Convert the reference image and image to be aligned into grayscale images from RGB color space
    img1Gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    img2Gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)

    # Create SURF detector with the hessian threshold set earlier
    surf = cv2.xfeatures2d.SURF_create(surfHessianThreshold)

    # Detect key points(features) from the reference image and image to be aligned, and compute descriptors for the key points detected
    reference_keypoints, reference_descriptor = surf.detectAndCompute(img1Gray, None)
    toAlign_keypoints, toAlign_descriptor = surf.detectAndCompute(img2Gray, None)

    # Illustrate the key points detected on the reference image and image to be aligned
    referenceKP = cv2.drawKeypoints(img1.copy(), reference_keypoints, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    toAlignKP = cv2.drawKeypoints(img2.copy(), toAlign_keypoints, None, flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show all the key points detected in the reference image and image to be aligned
    # fig, axes = pt.subplots(1, 2)
    # axes[0].set_title("Reference Image keypoints")
    # axes[0].imshow(referenceKP)
    # axes[1].set_title("Image to be Aligned keypoints")
    # axes[1].imshow(toAlignKP)
    # fig.show()

    # Show the number of key points detected from the reference image and image to be aligned
    print("\nNumber of key points detected in reference image: ", len(reference_keypoints))
    print("Number of key points detected in image to be aligned: ", len(toAlign_keypoints))

    # Create Brute-Force matcher with crossCheck = True
    bf = cv2.BFMatcher(crossCheck = True)

    # Match the key points between reference image and image to be aligned using the Brute-Force matcher created
    matches = bf.match(toAlign_descriptor, reference_descriptor)

    # Sort the matches by their distance attribute, a lower distance indicates a better match (score of similarity between the two descriptors in a match)
    matches = sorted(matches, key = lambda x : x.distance)

    # Retain only a percentage of the top scored matches (percentage set earlier) (considered good matches)
    goodMatchPercent = goodMatchPercent
    numGoodMatches = int(len(matches) * goodMatchPercent)
    matches = matches[: numGoodMatches]

    # Show the number of good matches between the reference image and image to be aligned
    print("\nNumber of Good Matching Key Points Between The Training and Test Image: ", len(matches))

    # Illustrate the good matches on the two images
    result = cv2.drawMatches(img2.copy(), toAlign_keypoints, img1.copy(), reference_keypoints, matches, None, flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS) #flags = 2 is same as flags = cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS()

    # Show the good matches between image to be aligned and reference image
    # pt.figure()
    # pt.rcParams['figure.figsize'] = [14.0, 7.0]
    # pt.title('Best Matching Points (Image to be Aligned - Reference Image)')
    # pt.imshow(result)
    # pt.show()

    # Extract the location of each good matching point in the two images
    reference_points = np.zeros((len(matches), 2), dtype = np.float32)
    toAlign_points = np.zeros((len(matches), 2), dtype = np.float32)

    for (i, match) in enumerate(matches):
        reference_points[i] = reference_keypoints[match.trainIdx].pt
        toAlign_points[i] = toAlign_keypoints[match.queryIdx].pt

    # Compute homography matrix based on the good matching points' locations (RANSAC procedure is used to ensure accurate outcomes can be obtained even if there's a large number of bad matches)
    homography, _ = cv2.findHomography(toAlign_points, reference_points, cv2.RANSAC)

    # Transform Image to be Aligned to have the same perspective as Reference Image
    height, width, _ = img1.shape
    alignedImg = cv2.warpPerspective(img2, homography, (width, height))

    # Show the reference image and the transformed image to be aligned (aligned image)
    # fig, axes = pt.subplots(1, 2)
    # axes[0].set_title("Reference Image")
    # axes[0].imshow(img1)
    # axes[1].set_title("Aligned Image")
    # axes[1].imshow(alignedImg)
    # fig.show()

    # Perform image stitching and extract images in their respective locations in the stitched image, along with binary maps to indicate pixels occupied by each of them in the stitched image
    stitched_img, img1_binary, img1only, alignedImg_binary, alignedImgOnly = stitchImages(img1, img2, homography)

    # Show the stitched image of the reference image and aligned image
    # pt.figure()
    # pt.title("Stitched Image of Reference Image and Aligned Image")
    # pt.imshow(stitched_img)
    # pt.show()

    # Show the two images with pixels occupied by reference image and aligned image in the stitched image respectively
    # pt.figure()
    # pt.subplot(1, 2, 1)
    # pt.title("Pixels occupied by Reference Image in Stitched Image")
    # pt.imshow(img1only)
    # pt.subplot(1, 2, 2)
    # pt.title("Pixels occupied by Aligned Image in Stitched Image")
    # pt.imshow(alignedImgOnly)
    # pt.show()

    # Show the binary maps indicating pixels occupied by reference image and aligned image in the stitched image
    # pt.figure()
    # pt.subplot(1, 2, 1)
    # pt.title("Binary Map of Pixels occupied by Reference Image in Stitched Image")
    # pt.imshow(img1_binary, cmap = "gray")
    # pt.subplot(1, 2, 2)
    # pt.title("Binary Map of Pixels occupied by Aligned Image in Stitched Image")
    # pt.imshow(alignedImg_binary, cmap="gray")
    # pt.show()

    # Create the binary mask to indicate pixels occupied by the overlapped region between the two images based on the binary maps obtained
    [rows, cols] = img1_binary.shape
    overlapped = np.zeros((rows, cols), dtype = np.uint8)

    # If a value of 1 is output at a location in both binary maps, output a value of 255 in the binary mask at that location
    for i in range (rows):
        for j in range (cols):
            if img1_binary[i, j] == 1 and alignedImg_binary[i, j] == 1:
                overlapped[i, j] = 255

    # Show the binary mask indicating pixels occupied by the overlapped region between reference image and aligned image
    # pt.figure()
    # pt.imshow(overlapped, cmap="gray")
    # pt.title("Pixels occupied by Overlapped Region between Reference Image and Aligned Image")
    # pt.show()

    # Apply OpenCV's bitwise_and function with the binary mask to extract pixels in the overlapped region only from reference image and aligned image
    img1only_overlapped = cv2.bitwise_and(img1only, img1only, mask=overlapped)
    alignedImg_overlapped = cv2.bitwise_and(alignedImgOnly, alignedImgOnly, mask=overlapped)

    # Return the overlapped region extracted from the two images
    return img1only_overlapped, alignedImg_overlapped

# Function to pre-process two different images, then compare their similarity based on their overall structure by computing the Mean SSIM and SSIM image of them
def SSIMandDiff(img1only_overlapped, alignedImg_overlapped, winSize):

    # Convert reference image to HSV color space
    img1HSV = cv2.cvtColor(img1only_overlapped, cv2.COLOR_RGB2HSV)
    # Get the V channel of the HSV reference image (Grayscale representation)
    _, _, img1Gray = cv2.split(img1HSV)

    # Convert aligned image to HSV color space
    alignedImgHSV = cv2.cvtColor(alignedImg_overlapped, cv2.COLOR_RGB2HSV)
    # Get the V channel of the HSV aligned image (Grayscale representation)
    _, _, alignedImgGray = cv2.split(alignedImgHSV)

    # Apply convolution with a normalised box filter on the two V channel (grayscale) images with the appropriate kernel size calculated based on maximum image width of the two images
    img1Blur = cv2.blur(img1Gray, winSize)
    alignedImgBlur = cv2.blur(alignedImgGray, winSize)

    # Show the smoothed reference image and aligned image
    # fig, axes = pt.subplots(1, 2)
    # axes[0].set_title("Reference Image blurred with a normalised box filter")
    # axes[0].imshow(img1Blur, cmap="gray")
    # axes[1].set_title("Aligned Image blurred with a normalised box filter")
    # axes[1].imshow(alignedImgBlur, cmap="gray")
    # fig.show()

    # Compute Mean SSIM and the SSIM image using the blurred V channel images
    (ssim_score, SSIMimg) = ssim(img1Blur, alignedImgBlur, full=True)
    SSIMimg = (SSIMimg * 255).astype("uint8")

    # Return the Mean SSIM and SSIM image computed
    return ssim_score, SSIMimg

# Function to pre-process two images, then extract and outline the interested differences between them in blue on the second image
def processDifferences(img1only_overlapped, alignedImg_overlapped, imageWidth, SSIMimg, resizeFactor, winSize):

    # Apply Median Blur on the SSIM image with the appropriate kernel size calculated based on maximum image width of the two images
    SSIMimg = cv2.medianBlur(SSIMimg, winSize[0])

    # Apply pixel value thresholding with Otsu's method to extract significantly different areas between two images using an appropriate threshold
    _, diffBinary = cv2.threshold(SSIMimg, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find the external contours for the areas that are extremely different between the two images
    diffContours, _ = cv2.findContours(diffBinary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # Initialise a binary mask to indicate pixels occupied by contours of interested differences
    mask = np.zeros(alignedImg_overlapped.shape, dtype='uint8')

    # Initialise an output image to outline the interested differences identified in
    blueDifferencesOnImg = alignedImg_overlapped.copy()

    # For each determined contour
    for c in diffContours:
        # Calculate the contour area for the contour
        area = cv2.contourArea(c)

        # If the contour area is larger than a specific threshold (contour area thresholding)
        if area > (6.7568 * imageWidth - 11972.973) // resizeFactor: # If resizing procedure was applied, contour area threshold have to be readjusted as well
            cv2.drawContours(mask, [c], 0, (255, 255, 255), 2) # Outline the contours of areas that are considered interested differences in the binary mask
            cv2.drawContours(blueDifferencesOnImg, [c], 0, (0, 0, 255), 2) # Outline the contours of areas that are considered interested differences on the aligned image

    # Show the binary map and binary mask
    # fig, axes = pt.subplots(1, 2)
    # axes[0].set_title("Binary map indicating areas that are extremely dissimilar between the two images")
    # axes[0].imshow(diffBinary, cmap="gray")
    # axes[1].set_title("Binary mask with the contours of interested differences outlined")
    # axes[1].imshow(mask)
    # fig.show()

    # Return the aligned image with interested differences outlined in blue on it
    return blueDifferencesOnImg

def main():
    # Main menu
    print("\n" + "*"*76 + "\n Construction Site Progress Detection and Monitoring using Image Processing \n" + "*"*76)
    print("\n<" + "-"*32 + " MAIN MENU " + "-"*31 + ">\n")
    print("1) Image Alignment Only \n2) Image Alignment and Find Differences between Images\n\n0) Quit Program")

    # Main Menu Selection
    while True:
        try:
            # Prompt user to select a functionality
            userInput = int(input("\nPlease enter a selection: "))
            # Input Validation
            if userInput >= 0 and userInput < 3:
                # Exit Program
                if userInput == 0:
                    print("\nThank you for using the program. The system will exit now.")
                    sys.exit()
                # Initialise resizeInput
                else:
                    resizeInput = 0
                    break
            else:
                print("\nInvalid Selection! Please enter a valid number.")
        except ValueError:
            print("\nInvalid Selection! Please enter a valid number.")

    # Resize Option Choices
    if userInput == 2:
        print("\n<" + "-"*30 + " RESIZE OPTIONS " + "-"*29 + ">\n\nPlease select a resize option from the choices below:\n")
        print("1) No Resize (100% Image Size) \n2) Resize The Two Initial Images\n3) Resize the SSIM Image\n\n0) Quit Program")
        # Resize Option Selection
        while True:
            try:
                # Prompt user to select a resize option
                resizeInput = int(input("\nPlease enter a selection: "))
                if resizeInput >= 0 and resizeInput < 4:
                    # Quit Program
                    if resizeInput == 0:
                        print("\nThank you for using the program. The system will exit now.")
                        sys.exit()
                    else:
                        # Image Size is set to 100% initially
                        resizeFactor = 1
                        break
                else:
                    print("\nInvalid Selection! Please enter a valid number.")
            except ValueError:
                print("\nInvalid Selection! Please enter a valid number.")

        # Image Size Choices
        if resizeInput != 1:
            print("\n<" + "-"*31 + " IMAGE SIZES " + "-"*30 + ">\n\nPlease select an image size from the choices below:\n")
            print("1) 25% Image Size \n2) 12.5% Image Size\n3) 10% Image Size\n\n0) Quit Program")
            # Image Size Selection
            while True:
                try:
                    # Prompt user to select an image size
                    sizeInput = int(input("\nPlease enter a selection: "))
                    # Input Validation
                    if sizeInput >= 0 and sizeInput < 4:
                        # 25% Image Size
                        if sizeInput == 1:
                            resizeFactor = 4
                            break
                        # 12.5% Image Size
                        elif sizeInput == 2:
                            resizeFactor = 8
                            break
                        # 10% Image Size
                        elif sizeInput == 3:
                            resizeFactor = 10
                            break
                        # Quit Program
                        else:
                            print("\nThank you for using the program. The system will exit now.")
                            sys.exit()
                            break
                    else:
                        print("\nInvalid Selection! Please enter a valid number.")
                except ValueError:
                    print("\nInvalid Selection! Please enter a valid number.")

    # Image Set Choices
    print("\n<" + "-"*31 + " IMAGE SETS " + "-"*31 + ">\n\nPlease select a set of images from the choices below:\n")
    print("1)  TNB (24/07/2020 & 27/08/2020)\n2)  TNB (02/09/2020 & 07/10/2020)\n3)  TNB (25/07/2020 & 14/08/2020)\n4)  TNB (17/08/2020 & 07/09/2020)\n5)  TNB (09/09/2020 & 01/10/2020)")
    print("6)  PNLC - West (30/06/2020 & 10/07/2020)\n7)  PNLC - East (17/07/2020 & 24/07/2020)\n8)  PNLC - Overall (21/08/2020 & 28/08/2020)\n9)  PNLC - Overall (04/09/2020 & 11/09/2020)\n10) PNLC - Overall Front (25/09/2020 & 10/10/2020)\n\n0) Quit Program")

    # Image Set Selection
    while True:
        try:
            # Prompt user to select an image set
            imageInput = int(input("\nPlease enter a selection: "))
            # Input Validation
            if imageInput >= 0 and imageInput < 11:
                # Image Set 1 (TNB)
                if imageInput == 1:
                    filename1 = "TNB_1_1.jpg"
                    filename2 = "TNB_1_2.jpg"
                    break
                # Image Set 2 (TNB)
                elif imageInput == 2:
                    filename1 = "TNB_2_1.jpg"
                    filename2 = "TNB_2_2.jpg"
                    break
                # Image Set 3 (TNB)
                elif imageInput == 3:
                    filename1 = "TNB_3_1.jpg"
                    filename2 = "TNB_3_2.jpg"
                    break
                # Image Set 4 (TNB)
                elif imageInput == 4:
                    filename1 = "TNB_4_1.jpg"
                    filename2 = "TNB_4_2.jpg"
                    break
                # Image Set 5 (TNB)
                elif imageInput == 5:
                    filename1 = "TNB_5_1.jpg"
                    filename2 = "TNB_5_2.jpg"
                    break
                # Image Set 6 (PNLC West)
                elif imageInput == 6:
                    filename1 = "PNLC_1_1.JPG"
                    filename2 = "PNLC_1_2.JPG"
                    break
                # Image Set 7 (PNLC East)
                elif imageInput == 7:
                    filename1 = "PNLC_2_1.JPG"
                    filename2 = "PNLC_2_2.JPG"
                    break
                # Image Set 8 (PNLC Overall)
                elif imageInput == 8:
                    filename1 = "PNLC_3_1.JPG"
                    filename2 = "PNLC_3_2.jpg"
                    break
                # Image Set 9 (PNLC Overall)
                elif imageInput == 9:
                    filename1 = "PNLC_4_1.jpg"
                    filename2 = "PNLC_4_2.jpg"
                    break
                # Image Set 10 (PNLC Overall Front)
                elif imageInput == 10:
                    filename1 = "PNLC_5_1.JPG"
                    filename2 = "PNLC_5_2.JPG"
                    break
                # Quit Program
                else:
                    print("\nThank you for using the program. The system will exit now.")
                    sys.exit()
            else:
                print("\nInvalid Selection! Please enter a valid number.")
        except ValueError:
            print("\nInvalid Selection! Please enter a valid number.")

    # To ensure the correct relative paths to images are used on both Windows and MacOS/Linux systems
    img1 = cv2.imread(os.path.join("Resources", filename1), 1)
    img2 = cv2.imread(os.path.join("Resources", filename2), 1)

    # Image Alignment Functionality (Will be performed no matter which functionality is chosen)

    # Find maximum image width from the two images read
    imageWidth = max(img1.shape[1], img2.shape[1])

    # Convert to RGB Color Space to show the images using pyplot
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB) # Reference Image
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB) # Image to be Aligned

    # Showing the original reference image and original image to be aligned
    fig, axes = pt.subplots(1, 2)
    axes[0].set_title("Reference Image")
    axes[0].imshow(img1)
    axes[1].set_title("Image to be Aligned")
    axes[1].imshow(img2)
    fig.show()

    # Initialise hessian threshold for SURF detector
    surfHessianThreshold = 300

    # Set the percentage of top scored matches to retain and used in computing homography matrix for image transformation
    goodMatchPercent = 0.15

    # Execute the imageAlignment function to align the images based on feature matching and obtain the overlapped parts from the two images
    img1only_overlapped, alignedImg_overlapped = imageAlignment(img1, img2, surfHessianThreshold, goodMatchPercent)

    # Calculate the appropriate kernel size to be used for image blurring later
    winSize = int(0.0017 * imageWidth + 9.7568)

    # Ensure the kernel size is not an even number
    if winSize % 2 == 0:
        winSize += 1

    # Convert kernel size into a tuple
    winSize = (winSize, winSize)

    # Show the overlapped regions from reference image and aligned image which are obtained from performing image alignment on the images
    pt.figure()
    pt.subplot(1, 2, 1)
    pt.title("Reference Image - Overlapped Region Only")
    pt.imshow(img1only_overlapped)
    pt.subplot(1, 2, 2)
    pt.title("Aligned Image - Overlapped Region Only")
    pt.imshow(alignedImg_overlapped)
    pt.show()

    # Compute Mean SSIM and SSIM Image (Pixel values range between 0 and 1, 0 indicates extreme dissimilarity at that location, whereas 1 indicates the two images are nearly identical at that location (or same))
    ssim_score, SSIMimg = SSIMandDiff(img1only_overlapped, alignedImg_overlapped, winSize)
    print("\nMean SSIM: ", ssim_score)


    # Image Alignment and Find Differences between Images Functionality
    if userInput == 2:
        # Resize the initial reference image and aligned image before computing the SSIM image
        if resizeInput == 2:
            # Resizing the initial reference image and aligned image with the image size option selected by the user from main menu
            img1only_overlapped = ResizeWithAspectRatio(img1only_overlapped, img1only_overlapped.shape[1] // resizeFactor, img1only_overlapped.shape[0] // resizeFactor)
            alignedImg_overlapped = ResizeWithAspectRatio(alignedImg_overlapped, alignedImg_overlapped.shape[1] // resizeFactor, alignedImg_overlapped.shape[0] // resizeFactor)

            # Compute new SSIM image using the resized initial reference image and resized aligned image
            _, SSIMimg = SSIMandDiff(img1only_overlapped, alignedImg_overlapped, winSize)

        # Resize the SSIM image computed from the overlapped parts of non-resized reference image and aligned image
        if resizeInput == 3:
            # Process and Find Differences between images
            SSIMimg = ResizeWithAspectRatio(SSIMimg, SSIMimg.shape[1] // resizeFactor, SSIMimg.shape[0] // resizeFactor)
            alignedImg_overlapped = ResizeWithAspectRatio(alignedImg_overlapped, alignedImg_overlapped.shape[1] // resizeFactor, alignedImg_overlapped.shape[0] // resizeFactor)

        # Pre-process the images, then find and outline the interested differences between the reference image and aligned image in blue
        output = processDifferences(img1only_overlapped, alignedImg_overlapped, imageWidth, SSIMimg, resizeFactor, winSize)

        # Show the reference image and the interested differences outlined in blue on aligned image
        fig, axes = pt.subplots(1, 2)
        axes[0].set_title("Reference Image")
        axes[0].imshow(img1only_overlapped)
        axes[1].set_title("Interested Differences Outlined in Blue on Aligned Image")
        axes[1].imshow(output)
        fig.show()

if __name__ == "__main__":
    main()
