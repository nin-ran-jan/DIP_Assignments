from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

def interpolate(matrix, temp_point):
    x_floor = math.floor(temp_point[0])
    x_ceil = math.ceil(temp_point[0])
    y_floor = math.floor(temp_point[1])
    y_ceil = math.ceil(temp_point[1])

    try:
        if x_floor == x_ceil and y_floor == y_ceil:
            return matrix[x_floor, y_floor]
            #direct on point
        
        elif x_floor == x_ceil:
            m = (matrix[x_floor, y_ceil]-matrix[x_floor, y_floor])/(y_ceil-y_floor)
            return m*(temp_point[1]-y_floor) + matrix[x_floor, y_floor]
            #linear interpolation

        elif y_floor == y_ceil:
            m = (matrix[x_ceil, y_floor]-matrix[x_floor, y_floor])/(x_ceil-x_floor)
            return m*(temp_point[0]-x_floor) + matrix[x_floor, y_floor]
            #linear interpolation

        else:
            #bilinear interpolation
            X = np.array(
                [[x_floor, y_floor, x_floor*y_floor,1],
                [x_floor, y_ceil, x_floor*y_ceil,1],
                [x_ceil, y_floor, x_ceil*y_floor,1],
                [x_ceil, y_ceil, x_ceil*y_ceil,1]]
            )
            V = np.array(
                [[matrix[x_floor,y_floor]],
                [matrix[x_floor,y_ceil]],
                [matrix[x_ceil,y_floor]],
                [matrix[x_ceil,y_ceil]]]
            )
            A = np.linalg.inv(X+(1e-10*np.identity(4))) @ V

            return np.dot(A.T, 
            np.array([temp_point[0], 
            temp_point[1], 
            temp_point[0]*temp_point[1], 
            1]))

    except:
        # does not handle borders that well.
        # but a few pixels do not affect the overall quality of the image.
        IndexError
        return 0

def getImage(matrix, transform_matrix_inverse):
    
    output_matrix = np.zeros(matrix.shape)

    for i in range(output_matrix.shape[0]):
        for j in range(output_matrix.shape[1]):
            temp_point = np.array([i,j,1])
            temp_point = temp_point @ transform_matrix_inverse
            output_matrix[i,j] = interpolate(matrix, temp_point)

    return output_matrix.astype(np.uint8)


def getTrans(t_x, t_y, theta, s_x, s_y, order):

    translation_matrix = np.array([
        [1,0,0],
        [0,1,0],
        [t_x, t_y, 1]
    ])

    #According to the convention, CW -> +ve, ACW -> -ve
    rotate_matrix = np.array([
        [np.cos(theta*np.pi/180),-np.sin(theta*np.pi/180),0],
        [np.sin(theta*np.pi/180),np.cos(theta*np.pi/180),0],
        [0, 0, 1]
    ])

    scale_matrix = np.array([
        [s_x,0,0],
        [0,s_y,0],
        [0, 0, 1]
    ])

    if(order == "trs"):
        return translation_matrix @ rotate_matrix @ scale_matrix

def logTrans(matrix):
    output_matrix = np.zeros(matrix.shape)
    constant = 255/np.log(256)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            output_matrix[i,j] = math.floor(constant*np.log(1+matrix[i,j]))
    
    return output_matrix.astype(np.uint8)

def calcHist(matrix):
    hist = [0]*256
    for i in matrix:
        for j in i:
            hist[j] += 1
    
    return np.array(hist)/(matrix.size)

def calcCDF(hist):
    cdf = np.zeros(hist.shape)
    cdf[0] = hist[0]
    for i in range(1, len(hist)):
        cdf[i] = cdf[i-1]+hist[i]

    return cdf*(255)

def getMatching(cdf_input, cdf_log):
    matching = np.zeros(cdf_input.shape)
    for r in range(len(cdf_input)):
        argmins = 1e10
        s_matched = -1
        for s in range(len(cdf_log)):
            if(abs(cdf_input[r]-cdf_log[s]) < argmins):
                argmins = abs(cdf_input[r]-cdf_log[s])
                s_matched = s
        
        matching[r] = s_matched
    
    return matching

def replacePixels(matrix, matching):
    output_matrix = np.zeros(matrix.shape)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            output_matrix[i,j] = matching[matrix[i,j]]
    
    return output_matrix.astype(np.uint8)

def main():
    
    #WHAT IS THIS
    #Show the mapping of coordinates for which you need interpolation.

    # x5 = np.array(Image.open('x5.bmp'))
    # x5 = x5[:200, :200]

    # x5_cropped = np.zeros((300, 300))
    # x5_cropped[:x5.shape[0], :x5.shape[1]] = x5
    # x5_cropped = x5_cropped.astype(np.uint8)
    # #cropped and padded with zeros
    # output_image = Image.fromarray(x5_cropped, 'L')
    # output_image.save("x5_cropped.bmp")
    # output_image.show()

    # #Q1
    # #calculate transformed image, x5_trans
    # joint_transformation_matrix = getTrans(50, 50, 10, 1, 1, "trs")
    # print("Joint transformation matrix:\n",
    # joint_transformation_matrix, sep = "")

    # joint_transformation_matrix_inverse = np.linalg.inv(
    #     joint_transformation_matrix + 1e-10*np.identity(3)
    #     )

    # x5_trans = getImage(x5_cropped, joint_transformation_matrix_inverse)
    # output_image = Image.fromarray(x5_trans, 'L')
    # output_image.save("x5_trans.bmp")
    # output_image.show()

    # #Q2
    # #x5_final is the reference image, and x5_final_out is the unregistered image
    # #to register x5_final_out, we calculate x5_final_reg
    # x5_reg = getImage(x5_trans, joint_transformation_matrix)
    # output_image = Image.fromarray(x5_reg, 'L')
    # output_image.save("x5_reg.bmp")
    # output_image.show()

    #Q3
    x5 = np.array(Image.open('x5.bmp'))
    output_image = Image.fromarray(x5, 'L')
    output_image.show()

    x5_log = logTrans(x5)
    output_image = Image.fromarray(x5_log, 'L')
    output_image.save("x5_log.bmp")
    output_image.show()

    hist_input = calcHist(x5)
    hist_log = calcHist(x5_log)

    # print(hist_input, hist_log, sep = "\n")

    plt.bar(range(256), hist_input)
    plt.title("Normalized histogram input image")
    plt.show()
    plt.bar(range(256), hist_log)
    plt.title("Normalized histogram log transformed image")
    plt.show()

    cdf_input = calcCDF(hist_input)
    cdf_log = calcCDF(hist_log)

    plt.bar(range(256), cdf_input)
    plt.title("CDF input image")
    plt.show()
    plt.bar(range(256), cdf_log)
    plt.title("CDF log transformed image")
    plt.show()

    matching = getMatching(cdf_input, cdf_log)

    # print(matching)

    x5_matched = replacePixels(x5, matching)
    output_image = Image.fromarray(x5_matched, 'L')
    output_image.save("x5_matched.bmp")
    output_image.show()


if __name__ == "__main__":
    main()