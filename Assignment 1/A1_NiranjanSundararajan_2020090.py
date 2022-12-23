x`import math
import numpy as np
from PIL import Image

def linearInt(input_point, input_matrix_v, dim):
    """
    This function does linear interpolation.
    This case is used in case the image lies on the border. 
    'dim' parameter handles whether pixel lies on x-axis border or y-axis border.
    It returns the intensity pixel value at the particular point.
    """
    x = int(input_point[dim])
    y1 = math.floor(input_point[1-dim])
    y2 = math.ceil(input_point[1-dim])
    if(y1 >= input_matrix_v.shape[1-dim]):
        v1 = 0
        v2 = 0
    elif(y2 >= input_matrix_v.shape[1-dim]):
        v2 = 0
        if dim==0:
            v1 = input_matrix_v[x][y1]
        elif dim==1:
            v1 = input_matrix_v[y1][x]
    else:
        if dim==0:
            v1 = input_matrix_v[x][y1]
            v2 = input_matrix_v[x][y2]
        elif dim==1:
            v1 = input_matrix_v[y1][x]
            v2 = input_matrix_v[y2][x]
    m = (v2-v1)/(y2-y1)
    v = m*(input_point[1-dim]-y1) + v1
    return v

def bilinearInt(input_point, input_matrix_v):
    """
    This function handles bilinear interpolation.
    It find neighbour points and adds zero padding in case the points lie outside the i/p image.
    It returns the intensity pixel value at the particular point.
    """

    x1,y1 = math.floor(input_point[0]), math.floor(input_point[1])
    x2,y2 = math.floor(input_point[0]), math.ceil(input_point[1])
    x3,y3 = math.ceil(input_point[0]), math.floor(input_point[1])
    x4,y4 = math.ceil(input_point[0]), math.ceil(input_point[1])
    if(input_point[0]<=input_matrix_v.shape[0]-1 
    and input_point[1]<=input_matrix_v.shape[1]-1):
        #normal
        v1 = input_matrix_v[x1][y1]
        v2 = input_matrix_v[x2][y2]
        v3 = input_matrix_v[x3][y3]
        v4 = input_matrix_v[x4][y4]
    elif(input_point[0]<input_matrix_v.shape[0] 
    and input_point[0]>input_matrix_v.shape[0]-1 
    and input_point[1]<input_matrix_v.shape[1] 
    and input_point[1]>input_matrix_v.shape[1]-1):
        v1 = input_matrix_v[x1][y1]
        v2 = 0
        v3 = 0
        v4 = 0
    elif(input_point[0]<input_matrix_v.shape[0] 
    and input_point[0]>input_matrix_v.shape[0]-1
    and input_point[1]<input_matrix_v.shape[1]-1):
        #y inside, x outside
        v1 = input_matrix_v[x1][y1]
        v2 = input_matrix_v[x2][y2]
        v3 = 0
        v4 = 0
    elif(input_point[0]<input_matrix_v.shape[0]-1 
    and input_point[1]<input_matrix_v.shape[1] 
    and input_point[1]>input_matrix_v.shape[1]-1):
        #x inside, y outside
        v1 = input_matrix_v[x1][y1]
        v2 = 0
        v3 = input_matrix_v[x3][y3]
        v4 = 0
    else:
        return 0
    
    X = np.array([[x1,y1,x1*y1,1],
    [x2,y2,x2*y2,1],
    [x3,y3,x3*y3,1],
    [x4,y4,x4*y4,1]])
    V = np.array([v1,v2,v3,v4]).reshape((4,1))

    # print(X)
    # print(V)

    A = np.matmul(np.linalg.inv((X+(1e-10*np.identity(4)))),V)

    # print(A)

    temp_point = np.array([input_point[0], 
    input_point[1], 
    input_point[0]*input_point[1], 
    1])
    # print(temp_point)
    # print(A)
    return np.dot(A.T, temp_point)
    
def main(matrix, interpolation_factor):
    """
    This is the main function. 
    It computes linear/bilinear interpolation for all points in the i/p matrix.
    It also checks whether zero-padding is necessary or not in some conditions.
    It returns the output matrix.
    """

    input_matrix_v = matrix
    output_matrix_v = np.zeros(input_matrix_v.shape)
    print("Input matrix is: \n", input_matrix_v)
    # print(input_matrix_v.shape)

    for i in range(input_matrix_v.shape[0]):
        for j in range(input_matrix_v.shape[1]): #output points
            # if(i == 1 and j == 1):
                input_point = (i/interpolation_factor, j/interpolation_factor)
                # print(input_point)
                if (input_point[0] == int(input_point[0])
                and input_point[1] == int(input_point[1])):
                    if(input_point[0]<=input_matrix_v.shape[0]-1 
                    and input_point[1]<=input_matrix_v.shape[1]-1):
                        output_matrix_v[i][j] = input_matrix_v[int(input_point[0])][int(input_point[1])]
                    else:
                        output_matrix_v[i][j] = 0
                    continue
                elif (input_point[0] == int(input_point[0])):
                    #linearInt
                    if(input_point[0]<=input_matrix_v.shape[0]-1):
                        output_matrix_v[i][j] = linearInt(input_point, input_matrix_v, 0)
                    else:
                        output_matrix_v[i][j] = 0
                    continue
                elif(input_point[1] == int(input_point[1])):
                    #linearInt
                    if(input_point[1]<=input_matrix_v.shape[1]-1):
                        output_matrix_v[i][j] = linearInt(input_point, input_matrix_v, 1)
                    else:
                        output_matrix_v[i][j] = 0
                    continue
                else:
                    #bilinearInt
                    output_matrix_v[i][j] = bilinearInt(input_point, input_matrix_v)

    print("Output matrix is: \n", output_matrix_v)
    output_matrix_v = np.floor(output_matrix_v).astype(np.uint8)

    # f = open("out.txt", 'w')
    # for l in range(output_matrix_v.shape[0]):
    #     for m in range(output_matrix_v.shape[1]):
    #         f.write(str(output_matrix_v[l][m]) + " ")
    #     f.write("\n")
    # f.close()

    # f = open("inp.txt", 'w')
    # for l in range(input_matrix_v.shape[0]):
    #     for m in range(input_matrix_v.shape[1]):
    #         f.write(str(input_matrix_v[l][m]) + " ")
    #     f.write("\n")
    # f.close()

    return output_matrix_v



if __name__ == "__main__":
    #only for lesser scale

    option = int(input("Solve Q2b.(1) or Q2c.(2)? "))
    if(option == 1):
        main(np.array([[2,0,0,0],[0,1,3,1],[3,0,2,0]]), 0.75)

    elif(option == 2):
        x5 = np.array(Image.open('x5.bmp'))

        # for i in range(x5.shape[0]):
        #     for j in range(x5.shape[1]):
        #         if(x5[i][j] != x5out[i][j]):
        #             print("ERROR")

        output_image = Image.fromarray(main(x5, 0.2), 'L')
        output_image.save('x5interpolated.bmp')
        output_image.show()
    
    else:
        print("Incorrect input! Please choose 1 or 2.")