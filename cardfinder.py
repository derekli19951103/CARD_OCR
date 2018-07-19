import numpy as np
import sys
import cv2


def show(windowname, img):
    cv2.imshow(windowname, img)
    cv2.waitKey(0)


def fix_landsacepe(img):
    (h, w) = img.shape[:2]
    if h > w:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        return img


def shuffle_list(l):
    comb = []
    comb.append([l[0], l[1], l[2], l[3]])
    comb.append([l[0], l[1], l[3], l[2]])
    comb.append([l[0], l[2], l[1], l[3]])
    comb.append([l[0], l[2], l[3], l[1]])
    comb.append([l[0], l[3], l[2], l[1]])
    comb.append([l[0], l[3], l[1], l[2]])

    comb.append([l[1], l[0], l[2], l[3]])
    comb.append([l[1], l[0], l[3], l[2]])
    comb.append([l[1], l[2], l[0], l[3]])
    comb.append([l[1], l[2], l[3], l[0]])
    comb.append([l[1], l[3], l[2], l[0]])
    comb.append([l[1], l[3], l[0], l[2]])

    comb.append([l[2], l[1], l[0], l[3]])
    comb.append([l[2], l[1], l[3], l[0]])
    comb.append([l[2], l[0], l[1], l[3]])
    comb.append([l[2], l[0], l[3], l[1]])
    comb.append([l[2], l[3], l[0], l[1]])
    comb.append([l[2], l[3], l[1], l[0]])

    comb.append([l[3], l[1], l[2], l[0]])
    comb.append([l[3], l[1], l[0], l[2]])
    comb.append([l[3], l[2], l[1], l[0]])
    comb.append([l[3], l[2], l[0], l[1]])
    comb.append([l[3], l[0], l[2], l[1]])
    comb.append([l[3], l[0], l[1], l[2]])

    return comb


def fit_quad(quad_pts, square_pts):
    combs = shuffle_list(quad_pts)
    mindiff = float('Inf')
    best_quad = None
    for comb in combs:
        diff = comb - square_pts
        diff = np.sum(np.absolute(diff))
        if diff < mindiff:
            mindiff = diff
            best_quad = comb
    return best_quad


def removeOutliers(x, IQR):
    a = np.array(x)
    upper_quartile = np.percentile(a, 75)
    lower_quartile = np.percentile(a, 25)
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)
    resultIdx = []
    for idx in range(len(x)):
        if x[idx] >= quartileSet[0] and x[idx] <= quartileSet[1]:
            resultIdx.append(idx)
    return resultIdx


def preprocessing(denoise):
    grayscale = cv2.cvtColor(denoise, cv2.COLOR_BGR2GRAY)
    grad = cv2.morphologyEx(grayscale, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
    _, binary = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return binary


def BoxText(original, ratio_threshold=0.45, angle_threshold=10.0, margin=10):
    # ratio + means less boxes, angle + means more boxes
    preprocess = preprocessing(original)
    _, contours, hierarchy = cv2.findContours(preprocess, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    Boxed = preprocess.copy() * 0
    mask = preprocess.copy() * 0
    (h, w) = original.shape[:2]
    min_area = (h // 100) * (w // 100)
    max_height = h // 3
    max_width = w // 3
    min_height = h // 100
    min_width = w // 100
    angles = []
    interested_contours = []
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        rect = cv2.minAreaRect(contours[idx])
        box = cv2.boxPoints(rect)
        area = cv2.contourArea(np.array([box]))
        cv2.drawContours(mask, contours, idx, 255, -1)

        if area >= min_area:
            r = float(cv2.countNonZero(mask[y:y + h - 1, x:x + w - 1])) / area
            if r > ratio_threshold and min_width <= w <= max_width and min_height <= h <= max_height:
                center, dim, angle = cv2.minAreaRect(contours[idx])
                if angle < -45.:
                    angle += 90.
                angles.append(angle)
                interested_contours.append(contours[idx])

    NiceIdx = removeOutliers(angles, angle_threshold)
    for idx in range(len(interested_contours)):
        if idx in NiceIdx:
            rect = cv2.minAreaRect(interested_contours[idx])
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(Boxed, [box], 0, 255, margin)
    return Boxed


def deskew(denoise):
    boxed = BoxText(denoise)
    # cv2.imwrite('ori_box.png', boxed)
    coords = np.column_stack(np.where(boxed == 255))
    center, dim, angle = cv2.minAreaRect(coords)
    (h, w) = original.shape[:2]
    if angle < -45:
        angle += 90
    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    deskewed = cv2.warpAffine(denoise, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return deskewed, boxed, M


def perspective_correct(deskewed):
    hsv = cv2.cvtColor(deskewed, cv2.COLOR_BGR2HSV)
    edges = cv2.Canny(hsv, 100, 200)
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    connected = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, morph_kernel)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    _, contours, hierarchy = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    largest_area = 0
    largest_contour = None
    idx = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > largest_area:
            largest_contour = c
            largest_area = area
            l_idx = idx
        idx += 1
    best_poly = None
    for i in range(len(largest_contour) // 4):
        contour_poly = cv2.approxPolyDP(largest_contour, i, True)
        if len(contour_poly) == 4:
            best_poly = contour_poly
    temp = deskewed.copy()
    cv2.drawContours(temp, contours, l_idx, (0, 255, 0), 10)
    # cv2.imwrite('big_c.png', temp)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # getting pts
    quad_pts = np.float32([(best_poly[0][0][0], best_poly[0][0][1]),
                           (best_poly[1][0][0], best_poly[1][0][1]),
                           (best_poly[2][0][0], best_poly[2][0][1]),
                           (best_poly[3][0][0], best_poly[3][0][1])])
    square_pts = np.float32([(x, y),
                             (x, y + h),
                             (x + w, y),
                             (x + w, y + h)])
    best_quad = fit_quad(quad_pts, square_pts)
    best_quad = np.float32([best_quad[0], best_quad[1], best_quad[2], best_quad[3]])
    (h, w) = deskewed.shape[:2]
    transmtx = cv2.getPerspectiveTransform(best_quad, square_pts)
    corrected = cv2.warpPerspective(deskewed, transmtx, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return corrected, transmtx


def crop_card(original):
    denoise = cv2.fastNlMeansDenoisingColored(original)
    deskewed, ori_box, affine1 = deskew(denoise)
    # cv2.imwrite('deskewed.png', deskewed)
    deperspective, shear = perspective_correct(deskewed)
    redeskewed, _, affine2 = deskew(deperspective)
    # cv2.imwrite('depers.png', deperspective)
    # cv2.imwrite('redeskewed.png',redeskewed)

    boxed = BoxText(deperspective)
    coords = np.column_stack(np.where(boxed == 255))
    x, y, w, h = cv2.boundingRect(coords)
    card1 = redeskewed[x:x + w, y:y + h]

    (h, w) = ori_box.shape[:2]
    ori_box = cv2.warpAffine(ori_box, affine1, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    ori_box = cv2.warpPerspective(ori_box, shear, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    ori_box = cv2.warpAffine(ori_box, affine2, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    coords = np.column_stack(np.where(ori_box == 255))
    x, y, w, h = cv2.boundingRect(coords)
    card2 = redeskewed[x:x + w, y:y + h]

    area1 = card1.shape[0] * card1.shape[1]
    area2 = card2.shape[0] * card1.shape[1]
    if area1 >= area2:
        card = card1
        # cv2.imwrite('final_box.png', boxed)
    elif area2 > area1:
        card = card2
        # cv2.imwrite('final_box.png',ori_box)

    return card


def manual_crop(original, quad_pt1, quad_pt2, quad_pt3, quad_pt4):
    # points in format (x,y)
    quad_pts = np.float32([quad_pt1, quad_pt2, quad_pt3, quad_pt4])
    center, dim, angle = cv2.minAreaRect(np.array([quad_pts]))
    if angle < -45:
        angle += 90
    (hi, wd) = original.shape[:2]
    affine = cv2.getRotationMatrix2D(center, -angle, 1.0)
    deskewed = cv2.warpAffine(original, affine, (wd, hi), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    transformed_quad_pts = cv2.transform(np.array([quad_pts]), affine)
    x, y, w, h = cv2.boundingRect(transformed_quad_pts)
    square_pts = np.float32([(x, y),
                             (x, y + h),
                             (x + w, y),
                             (x + w, y + h)])
    best_quad = fit_quad(quad_pts, square_pts)
    best_quad = np.float32([best_quad[0], best_quad[1], best_quad[2], best_quad[3]])
    transmtx = cv2.getPerspectiveTransform(best_quad, square_pts)
    corrected = cv2.warpPerspective(deskewed, transmtx, (wd, hi), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
    return corrected[x:x + w, y:y + h]


if __name__ == "__main__":
    file = sys.argv[1]
    flag = sys.argv[2]
    filename = str(file).split('/')[1].split('.')[0]
    original = fix_landsacepe(cv2.imread(file))
    if flag == 'auto':
        try:
            card = crop_card(original)
            print(filename, ' finished processing')
            cv2.imwrite('cardfinal/' + filename + '.png', card)
        except TypeError:
            print('not a good card')
    elif flag == 'manual':
        quad_pt1 = input('point1:').split(',')
        quad_pt2 = input('point2:').split(',')
        quad_pt3 = input('point3:').split(',')
        quad_pt4 = input('point4:').split(',')
        quad_pt1 = [int(num) for num in quad_pt1]
        quad_pt2 = [int(num) for num in quad_pt2]
        quad_pt3 = [int(num) for num in quad_pt3]
        quad_pt4 = [int(num) for num in quad_pt4]
        card = manual_crop(original, quad_pt1, quad_pt2, quad_pt3, quad_pt4)
        print(filename, ' finished processing')
        cv2.imwrite('cardfinal/' + filename + '.png', card)
