# Construction Site Progress Detection and Monitoring using Image Processing
The aim of this project is to study the feasibility of detecting and monitoring construction site progress using image processing. Construction progress monitoring is largely done through manual observation, exhausting time, and manpower. Succeeding in developing image processing algorithms, which would allow construction progress to be determined through the comparison of images taken at different stages of construction, would reduce the time and resources required for monitoring construction progress. 

In this project, I have managed to successfully built an image processing algorithm that aligns two construction site images taken from various angles, resolutions, and construction stages, then extract construction progress information by making relevant comparisons between them.

# Instructions to Run the Program
**As the SURF algorithm used in the program was patented quite some time ago, since then it is moved to opencv_contrib. Thus, we need to install the older version of opencv_contrib as it was removed from the later versions as well.**

## 1. Download and Install Anaconda Navigator
Download Link: https://www.anaconda.com/products/individual

## 2. Open Anaconda Prompt/Terminal
### Windows
Search for Anaconda Prompt and Open it.
### MacOS/Linux
Search for Terminal and Open it.

## 3. Type the following commands into Anaconda Prompt/Terminal and press enter (one at a time & following the order)
i) Create a new environment
`conda create --name py37 python=3.7`
Type y and press enter when *Proceed ([y]/n)?* is asked.

***Creating a new environment is highly recommended as having different versions of Python and OpenCV will cause errors when trying to install the relevant modules and running the program***

ii) Switch to the newly created environment
 `conda activate py37`

iii) Install the packages required to run the program in the new environment
`pip install -U opencv-contrib-python==3.4.2.16`
`pip install matplotlib`
`pip install scikit-image`

## 4. Install and Launch Spyder on new environment
i) Open Anaconda Navigator

ii) Click on Environments on the sidebar at the left

iii) Click on py37 to switch to the new environment (Make sure the green arrow is shown beside py37, indicating that it is the current selected environment)

iv)  Click on Home on the sidebar at the left

v) Install Spyder

vi) Launch Spyder

## 5. Open the python script and run the script
Open the script SC_18086793_Mar2021.py and run it using Spyder.

**Ensure that the Resources folder with the images from TNB and PNLC dataset are located in the same directory as the script**
