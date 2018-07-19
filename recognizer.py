import cv2
import numpy as np
import pytesseract
import string
import sys
from fuzzywuzzy import process
from zhon.hanzi import punctuation
import re

def show(windowname, img):
    cv2.imshow(windowname, img)
    cv2.waitKey(0)


def fix_landsacepe(img):
    (h, w) = img.shape[:2]
    if h > w:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    else:
        return img


def remove_line(img):
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
    connected = cv2.morphologyEx(img, cv2.MORPH_CLOSE, morph_kernel)

    _, contours, hierarchy = cv2.findContours(connected, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = img.copy() * 0
    biggest = 0
    biggest_idx = 0
    for i in range(len(contours)):
        area = cv2.contourArea(contours[i])
        if area > biggest:
            biggest_idx = i
            biggest = area
    cv2.drawContours(mask, contours, biggest_idx, (255, 255, 255))

    coords = np.column_stack(np.where(mask == 255))
    x, y, w, h = cv2.boundingRect(coords)

    ROI = img[x:x + w, y:y + h]
    real_digits = get_digits(ROI)
    img[x:x + w, y:y + h] = real_digits
    return img


def get_digits(ROI):
    edges = cv2.Canny(ROI, 20, 60)
    rho = 1  # distance resolution in pixels of the Hough grid
    theta = np.pi / 180  # angular resolution in radians of the Hough grid
    threshold = 150  # minimum number of votes (intersections in Hough grid cell)
    lines = cv2.HoughLines(edges, rho, theta, threshold, max_theta=3)
    for line in lines:
        rho = line[0][0]
        theta = line[0][1]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        point1 = (int(round(x0 + 1000 * (-b))), int(round(y0 + 1000 * a)))
        point2 = (int(round(x0 - 1000 * (-b))), int(round(y0 - 1000 * a)))
        cv2.line(ROI, point1, point2, 0, 1)

    (h, w) = ROI.shape[:2]
    div = w // 8
    real_digits = ROI.copy() * 0
    j = 0
    for i in range(8):
        digit = ROI[:, i * div:(i + 1) * div]
        mask = digit.copy() * 0
        _, contours, hierarchy = cv2.findContours(digit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        biggest = 0
        biggest_idx = 0
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if area > biggest:
                biggest_idx = i
                biggest = area
        cv2.drawContours(mask, contours, biggest_idx, (255, 255, 255))
        coords = np.column_stack(np.where(mask == 255))
        x, y, w, h = cv2.boundingRect(coords)
        real_digit = digit[x:x + w, y:y + h]
        mask = digit.copy() * 0
        mask[x:x + w, y:y + h] = real_digit
        real_digits[:, j * div:(j + 1) * div] = mask  # for some reasons,if there's no j, the i
        j += 1  # will alter through out one iteration
    return real_digits


def better(img):
    rgb_planes = cv2.split(img)

    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 31)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = diff_img.copy()
        cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)

    deshade = cv2.merge(result_norm_planes)
    grayscale = 255 - cv2.cvtColor(deshade, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(grayscale, 80, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    hilight = cv2.inRange(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 240, 255)
    coords = np.column_stack(np.where(hilight == 255))
    if coords is not None and len(coords) != 0:
        x, y, w, h = cv2.boundingRect(coords)
        binary[x:x + w, y:y + h] = cv2.threshold(cv2.cvtColor(img[x:x + w, y:y + h], cv2.COLOR_BGR2GRAY), 80, 255,
                                                 cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    try:
        lineless = remove_line(closing)
    except TypeError:
        lineless = closing
        print('no lines detected')
    return lineless


def raw_process(img):
    ori_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, ori_binary = cv2.threshold(ori_grayscale, 80, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    opening = cv2.morphologyEx(ori_binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1)))
    return closing


def fix(s, correction):
    intab = ""
    outtab = ""
    for ori, cor in correction.items():
        intab += ori
        outtab += cor
    trantab = str.maketrans(intab, outtab)
    return s.translate(trantab)


def process_line(char, correction):
    raw_characters = char.replace(" ", "").strip()
    exclude = set(string.punctuation)
    exclude.add('・')
    exclude.add('〟')
    characters = ''.join(ch for ch in raw_characters if ch not in exclude)
    exclude=set(punctuation)
    characters = ''.join(ch for ch in characters if ch not in exclude)
    corrected = fix(characters, correction)
    return corrected


if __name__ == "__main__":
    file = sys.argv[1]
    filename = str(file).split('/')[1].split('.')[0]
    original = fix_landsacepe(cv2.imread(file))
    card = better(original)
    print(filename, ' finished processing')
    cv2.imwrite('processed/' + filename + '.png', card)
    answer = pytesseract.image_to_string(card, lang='jpn',config='--psm 1')
    lines = answer.split('\n')
    correction = {'ー': '1'}
    cleaned_lines = [process_line(line, correction) for line in lines if line != '']
    print(cleaned_lines)
    # TODO: identify cleaned lines
    who = process.extractOne("本人被保険", cleaned_lines)
    who = re.search(r'\D{2}被保険', who[0]).group(0)
    print("谁被保险？",who)

    birth=process.extractOne("生年月日", cleaned_lines)
    birth=process.extractOne("平成00年0月00日",birth[0].split("生年月日"))[0]
    print("出生年月日？",birth)

    mark=process.extractOne("記号00000000",cleaned_lines)
    mark=re.search(r'\d{8}',mark[0]).group(0)
    print("记号？",mark)

    number=process.extractOne("番号01",cleaned_lines)
    number=re.search(r'\d{1,2}$',number[0]).group(0)
    print("番号？",number)

    name=process.extractOne("氏名",cleaned_lines)
    name=re.search(r'\D{4}$',name[0]).group(0)
    print("名字？",name)

    company = process.extractOne("事業所名称", cleaned_lines)
    company = process.extractOne("株式会社", company[0].split("事業所名称"))[0]
    print("单位?",company)



