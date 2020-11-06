import cv2
import numpy as np
import time


def Var_Filter(Image, window_size=5, var_dof=0):
    loc = int(window_size/2)
    #Iz = np.pad(I, pad_width=loc, mode='constant', constant_values=0) #Zero-pad the original image.
    #height, width = Iz.shape
    height, width = Image.shape #Obtain the image's dimensions.
    I = Image.copy() #Copy the original input image's values into a new output matrix.
    for i in range(height-window_size):
        for j in range(width-window_size):
            neighborhood = [] #This list will have the values of the current pixels within the sliding window.
            miniI = Image[i:i+window_size, j:j+window_size] #Submatrix where window is currently "standing".
            for x in miniI: #Iterate over every element within the window.
                for y in x:
                    neighborhood.append(y) #Add the image's pixel value within the current window to the array.
            I[i+loc,j+loc] = np.var(neighborhood, ddof=var_dof) #Calculate the variance of the neighborhood and replace its result in the center pixel. ddof=1 >> unbiased estimator; ddof=0 >> maximum likelihood estimate
    return I

def SD_Filter(Image, window_size=5):
    loc = int(window_size/2)
    height, width = Image.shape #Obtain the image's dimensions.
    I = Image.copy() #Copy the original input image's values into a new output matrix.
    for i in range(height-window_size):
        for j in range(width-window_size):
            neighborhood = [] #This list will have the values of the current pixels within the sliding window.
            miniI = Image[i:i+window_size, j:j+window_size] #Submatrix where window is currently "standing".
            for x in miniI: #Iterate over every element within the window.
                for y in x:
                    neighborhood.append(y) #Add the image's pixel value within the current window to the array.
            I[i+loc,j+loc] = np.std(neighborhood) #Calculate the standard deviation of the neighborhood and replace its result in the center pixel.
    return I


def main():
    ti = time.time()
    #================================
    #Part 1: SD and Variance Filters
    #================================
    #I = cv2.imread("leaf.png",0) #Leaf image.
    I = cv2.imread("100_X_13_40_hrs_24_octubre_2.jpg",0) #Bacteria image.
    window_size = 5
    #I_var = Var_Filter(I, window_size, 0)
    #I_sd = SD_Filter(I,window_size)
    #Write the images.
    #cv2.imwrite("Bacteria_Variance_Unbiased_"+str(window_size)+"x"+str(window_size)+".jpg",I_var) #Save the image with a variance filter applied separately.
    #cv2.imwrite("Bacteria_SD_"+str(window_size)+"x"+str(window_size)+".jpg",I_sd)
    #=======================================
    #Part 2: Sobel and Canny Edge Detectors
    #=======================================
    I_sobel = cv2.Sobel(I, cv2.CV_64F, 1, 1, ksize=window_size) #Sobel edge detector incorporating x and y directions simultaneously.
    low_hist = 150
    upp_hist = 200
    I_canny = cv2.Canny(I,low_hist,upp_hist) #Canny edge detector.
    #Write the images.
    cv2.imwrite("Bacteria_Sobel_"+str(window_size)+"x"+str(window_size)+".jpg",I_sobel)
    cv2.imwrite("Bacteria_Canny_("+str(low_hist)+"-"+str(upp_hist)+").jpg",I_canny)
    tf = time.time()
    print("Elapsed Time:",tf-ti) #How many time ellapsed.

if __name__ == '__main__':
    main()