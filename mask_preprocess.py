import cv2


def create_rgba(img, m):
    specs_ori = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    (h, w, _) = specs_ori.shape
    for i in range(h):
        for j in range(w):
            if (specs_ori[i][j][0] > 254) & (specs_ori[i][j][1] > 254) & (specs_ori[i][j][2] > 254):
                specs_ori[i][j][3] = 0
    cv2.imwrite("masks/" + str(m) + ".png", specs_ori)


if __name__ == '__main__':
    for i in range(1, 6):
        first = cv2.imread("masks/" + str(i) + "_raw.png")
        create_rgba(first, i)
