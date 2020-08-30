import numpy as np

def main():
    a = np.array([[1,1], [2,2], [3,3]])
    b= np.insert(a, 0, 1, axis=1) #for part (g)
    n,m = a.shape
    for i in range(n):
        for j in range(1, m+1):
            b[i][j] = pow(b[i][j], j)
    print(b)

if __name__ == "__main__":
    main()
