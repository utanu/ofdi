import numpy as np
import cv2
import copy
import glob
import io
from scipy.spatial import distance
from PIL import Image
import random
import math
import PySimpleGUI as sg
import win32clipboard
import ctypes
from ctypes import windll

NUM_OF_ANGLES = 360 * 4
THICKNESS_THRESHOLD = 1.0


def is_pressed(key):
    """キーが押されているかどうか判定する

        :param key: キーコード

        """
    return bool(ctypes.windll.user32.GetAsyncKeyState(key) & 0x8000)


def dump_remover(img, circle_only=False):
    mask = np.full((img.shape[0], img.shape[1], img.shape[2]), 29, dtype=np.uint8)
    image = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8)

    if not circle_only:
        if img[int((2345 / 2444) * img.shape[0]), int((1245 / 2444) * img.shape[1])][2] >= 230:
            mask2, _ = cal_region(img, show=False)
            mask2 = pil_2_cv2(cv2_2_pil(mask2).convert('RGB'))

            img[int((2310 / 2444) * img.shape[0]):, :, :] = np.where(
                mask2[int((2310 / 2444) * img.shape[0])] == [255, 255, 255], img[int((2345 / 2444)) * img.shape[0]],
                mask2[int((2310 / 2444) * img.shape[0])])

    cv2.circle(mask, center=(int(img.shape[0] / 2), int(img.shape[1] / 2)), radius=int(img.shape[1] / 2),
               color=(255, 255, 255), thickness=-1)

    image[:] = np.where(mask == [255, 255, 255], img, mask)

    # cv2.imshow(l, cv2.resize(image, (512, 512)))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image


def cal_region(img, show=True):
    mistake = False
    blank_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    # 赤色取り出し(1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 128, 150])
    upper = np.array([30, 255, 255])
    img_mask = cv2.inRange(hsv, lower, upper)

    # 赤色取り出し(2)
    lower = np.array([150, 128, 150])
    upper = np.array([179, 255, 255])
    img_mask = img_mask + cv2.inRange(hsv, lower, upper)

    # 黄色取り出し
    lower = np.array([20, 128, 180])
    upper = np.array([50, 255, 255])
    img_mask = img_mask + cv2.inRange(hsv, lower, upper)

    bf = False
    if np.all(img_mask == 0):
        # 青色取り出し
        lower = np.array([85, 120, 90])
        upper = np.array([130, 255, 255])
        img_mask = cv2.inRange(hsv, lower, upper)
        print(np.count_nonzero(img_mask))
        # if np.count_nonzero(img_mask) <= 50:
        #     img_mask = img_mask_bck
        bf = True

    # cv2.imshow(l, img_mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    img_color = cv2.bitwise_and(img, img, mask=img_mask)
    # img_bgr = cv2.cvtColor(img_color, cv2.COLOR_HSV2BGR)

    luminance = (0.298912 * img_color[:, :, 2] + 0.586611 * img_color[:, :, 1] + 0.114478 * img_color[:, :, 0])

    bin_th = 50
    if bf:
        bin_th = 20
    # 二値化処理
    img_color[luminance < bin_th] = 0
    # img_color[img_color >= 127] = 255
    img_color[luminance >= bin_th] = 255

    gt = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    ogt = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 方法2 （OpenCVで実装）
    ret, th = cv2.threshold(gt, 0, 255, cv2.THRESH_OTSU)

    # # カーネルの定義
    kernel = np.ones((6, 6), np.uint8)

    #
    # # 膨張・収縮処理(方法2)
    result = cv2.dilate(th, kernel)

    #  result = cv2.erode(result, kernel)

    result = th

    contours, hierarchy = cv2.findContours(result, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # print(hierarchy)
    for i, c in enumerate(contours):
        # print(cv2.contourArea(c))
        # cv2.fillPoly(result, pts=c.reshape(1, -1, 2), color=(i*60, i*60, i*60))
        # カルシウムが血管全体を囲うような形の時、外側の輪郭線の外側の親は存在せず-1、外側の輪郭線の内側の親は0
        # 内側の輪郭線の外側の親は1、内側の輪郭線の内側の親は2となる
        # これを辿って入れ子になっている領域か判定し、ラベル付けされていない領域とわかれば黒で塗り直す
        cn = i
        chain = 0
        while True:
            if hierarchy[0][cn][3] != -1:
                chain += 1
                cn = hierarchy[0][cn][3]
            else:
                break
        if chain >= 2:
            cv2.fillPoly(result, pts=c.reshape(1, -1, 2), color=(0, 0, 0))
        else:
            cv2.fillPoly(result, pts=c.reshape(1, -1, 2), color=(255, 255, 255))

    result = cv2.erode(result, np.ones((2, 2), np.uint8))
    result = cv2.dilate(result, np.ones((2, 2), np.uint8))

    mimg = np.hstack((np.stack((result,) * 3, -1), img))
    # mimg = np.vstack((mimg, np.hstack((cv2.cvtColor(simg, cv2.COLOR_BGR2GRAY), cv2.cvtColor(simg, cv2.COLOR_BGR2GRAY)))))

    # オリジナル画像の高さ・幅を取得
    # height = mimg.shape[0]
    # width = mimg.shape[1]

    # (imshowの)画像の大きさを設定
    new_size = (512 * 2, 512)

    # 結果の表示
    if show:
        cv2.imshow("img", cv2.resize(mimg, (int(new_size[0]), int(new_size[1]))))
        cv2.waitKey(0)
        if mlib.is_pressed(ord('B')):
            result = blank_mask
        if mlib.is_pressed(ord('D')):
            mistake = True
            print("miss")
        cv2.destroyAllWindows()

    return result, mistake


# 画像結合(縦)
def lvstack(l):
    out = l[0]
    for piece in l[1:]:
        print(piece)
        out = np.vstack((out, piece))

    return out


# 画像結合(横)
def lhstack(l):
    out = l[0]
    for i, piece in enumerate(l[1:]):
        print(i)
        out = np.hstack((out, piece))

    return out


def pil_2_cv2(pil_image):
    """PIL形式->OpenCV形式へ画像変換を行う

                    :param pil_image: PIL形式の画像

                    """
    open_cv_image = np.array(pil_image)
    # Convert RGB to BGR (method-1):
    open_cv_image = cv2.cvtColor(open_cv_image, cv2.COLOR_RGB2BGR)
    # Convert RGB to BGR (method-2):
    # b, g, r = cv2.split(open_cv_image)
    # open_cv_image = cv2.merge([r, g, b])
    return open_cv_image


def cv2_2_pil(cv2_image):
    """OpenCV形式->PIL形式へ画像変換を行う

                        :param cv2_image: PIL形式の画像

                        """
    cv2_im = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_im = Image.fromarray(cv2_im)
    return pil_im


def ccode_2_bgr(ccode):
    R = int(ccode[1:3], 16)
    G = int(ccode[3:5], 16)
    B = int(ccode[5:7], 16)

    return (B, G, R)


# 外枠追加
def addborder(image, size, color=None):
    bk1 = np.zeros((size, image.shape[1], 3), np.uint8)
    if color is not None:
        bk1[:] = list(color)
    else:
        bk1[:] = [45, 38, 39]
    image = np.insert(image, 0, bk1, axis=0)
    image = np.insert(image, image.shape[0], bk1, axis=0)
    bk2 = np.zeros((image.shape[0], size, 3), np.uint8)
    if color is not None:
        bk2[:] = list(color)
    else:
        bk2[:] = [45, 38, 39]

    image = np.insert(image, [0], bk2, axis=1)
    image = np.insert(image, [image.shape[1]], bk2, axis=1)

    return image


def copy_to_clipboard(image):
    # メモリストリームにBMP形式で保存してから読み出す
    output = io.BytesIO()
    image.convert('RGB').save(output, 'BMP')
    data = output.getvalue()[14:]
    output.close()
    send_to_clipboard(win32clipboard.CF_DIB, data)

def send_to_clipboard(clip_type, data):
    # クリップボードをクリアして、データをセットする
    win32clipboard.OpenClipboard()
    win32clipboard.EmptyClipboard()
    win32clipboard.SetClipboardData(clip_type, data)
    win32clipboard.CloseClipboard()


# 円検出
def detect_circles(img):
    image = np.array(img)

    # delete pink
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([130, 30, 80])  # 紫に近いピンク
    upper = np.array([175, 255, 255])  # 赤に近いピンク
    img_mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((2, 2), np.uint8)
    img_mask = cv2.dilate(img_mask, kernel)

    image = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(img_mask))

    # delete green
    lower = np.array([50, 100, 100])
    upper = np.array([90, 255, 255])
    img_mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((2, 2), np.uint8)
    img_mask = cv2.dilate(img_mask, kernel)
    image = cv2.bitwise_and(image, img, mask=cv2.bitwise_not(img_mask))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.medianBlur(cv2.resize(image[223:289, 223:289], (512, 512)), 5)
    cimg = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    cv2.imshow('cimg', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    p2 = 100
    circles = None
    while circles is None:
        circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT, 1, 10, param1=50, param2=p2, minRadius=int(23*(512/66)), maxRadius=0)
        p2 -= 1


    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        # draw the outer circle
        cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
        # draw the center of the circle
        cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    cv2.imshow('detected circles', cimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def cross_section(rotate = 0):
    out = []
    for i in range(len(oimg_list)):
        img = oimg_list[i]
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img_hsv[:, :, 1] = img_hsv[:, :, 1] * 0  # 彩度の計算
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        img = cv2.medianBlur(img, 1)
        cimg = calim_list[i]
        cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2GRAY)
        cimg[cimg > 10] = 255
        cimg[cimg <= 10] = 0
        # cimg_hsv = cv2.cvtColor(cimg, cv2.COLOR_BGR2HSV)
        # cimg_hsv[:, :, 0] = cimg_hsv[:, :, 0] - 185  # 色相の計算
        # cimg = cv2.cvtColor(cimg_hsv, cv2.COLOR_HSV2BGR)  # 色空間をHSVからBGRに変換
        orange = np.zeros((512, 512, 3))
        orange += [213, 114, 49][::-1]  # RGBで指定
        orange = orange.astype(np.uint8)
        orange = cv2.bitwise_or(orange, orange, mask=cimg)
        # img = np.where(orange != 0, orange, img) #  なぜか上書きが成功した例
        img = np.where(orange != 0, orange, img)
        img = cv2.GaussianBlur(img, (3, 3), 0)

        # cv2.imshow('detected circles', img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if rotate != 0:
            h, w = img.shape[:2]
            xc, yc = int(w/2), int(h/2)  # 回転中心

            # 回転中心と四隅の距離の最大値を求める
            pts = np.array([(0, 0), (w, 0), (w, h), (0, h)])
            ctr = np.array([(xc, yc)])
            r = np.sqrt(max(np.sum((pts - ctr) ** 2, axis=1)))
            winH, winW = int(2 * r), int(2 * r)

            M = cv2.getRotationMatrix2D((xc, yc), rotate, 1)
            M[0][2] += r - xc
            M[1][2] += r - yc

            img = cv2.warpAffine(img, M, (winW, winH))

        out.append(img[:, img.shape[1] // 2])

        # print(np.array(out).shape)

    out = np.transpose(out, (1, 0, 2))
    out = cv2.resize(out, (int(out.shape[1] * 9 / 4), int(out.shape[0] / 4)), interpolation=cv2.INTER_NEAREST)
    out = cv2.resize(out, (int(out.shape[1] * (510 / max(out.shape[0], out.shape[1]))),
                           int(out.shape[0] * (510 / max(out.shape[0], out.shape[1])))),
                     interpolation=cv2.INTER_NEAREST)
    # print(out.shape)

    # cv2.imshow("out", out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return out


def cs_scale(length):
    scale_bar = np.zeros((12, 516, 3), np.uint8)
    scale_bar[:] = ccode_2_bgr('#27262d')

    for i in range(6):
        cv2.putText(scale_bar, str(i*(length*0.2)), (int(49*i*2), 12), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

    # cv2.imshow("out", scale_bar)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return scale_bar

def mk_color_bar():
    background_color = ccode_2_bgr('#27262d')
    color_bar = np.zeros((256, 256, 3), np.uint8)
    for t in range(256):
        color_bar[t] = list(ring_color(t/4.5))

    color_bar = cv2.resize(np.flipud(color_bar), (16, 512))
    print(color_bar)

    color_bar = addborder(addborder(color_bar, 1, (0, 0, 0)), 50, color=background_color)

    color_bar = color_bar[:, 49:110]

    for i in range(11):
        cv2.putText(color_bar, str(i*0.1), (34, 512 - int(50*(i-1))), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))

    color_bar = color_bar[45:color_bar.shape[0]-45, :]

    # cv2.imshow("out", color_bar)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return color_bar


# RINGの色を計算
def ring_color(t):
    h = 240
    # h = 180

    # if (t * (9 / WIDTH)) > THICKNESS_THRESHOLD:
    #     h = 0
    # if t == 0:
    #     return hsv_to_rgb(0, 0, 255)  # V = 0～255で濃淡調整
    # else:
    h -= h * ((t * (9 / WIDTH)) / THICKNESS_THRESHOLD)
    if h <= 0:
        h = 0
    # print("h is ", h)

    return hsv_to_rgb(h, 255, 255)


# HSV形式からRGB形式に変換
# R=0 ~ 255, G=0 ~ 255, B=0 ~ 255,
def hsv_to_rgb(h, s, v):
    #    h = 0 ~ 360
    #    s = 0 ~ 255
    #    v = 0 ~ 255

    i = int(h / 60.0)
    mx = v
    mn = v - ((s / 255.0) * v)

    if h is None:
        return (0, 0, 0)

    if i == 0:
        (r1, g1, b1) = (mx, (h / 60.0) * (mx - mn) + mn, mn)
    elif i == 1:
        (r1, g1, b1) = (((120.0 - h) / 60.0) * (mx - mn) + mn, mx, mn)
    elif i == 2:
        (r1, g1, b1) = (mn, mx, ((h - 120.0) / 60.0) * (mx - mn) + mn)
    elif i == 3:
        (r1, g1, b1) = (mn, ((240.0 - h) / 60.0) * (mx - mn) + mn, mx)
    elif i == 4:
        (r1, g1, b1) = (((h - 240.0) / 60.0) * (mx - mn) + mn, mn, mx)
    elif 5 <= i:
        (r1, g1, b1) = (mx, mn, ((360.0 - h) / 60.0) * (mx - mn) + mn)

    return (int(b1), int(g1), int(r1))


# 石灰化領域抽出(黄色抽出)
def extract_calcium(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 黄色取り出し
    lower = np.array([20, 80, 10])
    upper = np.array([50, 255, 255])
    img_mask = cv2.inRange(hsv, lower, upper)
    img_color = cv2.bitwise_and(img, img, mask=img_mask)
    img_bgr = cv2.cvtColor(img_color, cv2.COLOR_HSV2BGR)

    return img_color


def wire_region(img):
    image = np.array(img)

    # # delete pink
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower = np.array([130, 30, 80])  # 紫に近いピンク
    # upper = np.array([175, 255, 255])  # 赤に近いピンク
    # img_mask = cv2.inRange(hsv, lower, upper)
    # kernel = np.ones((2, 2), np.uint8)
    # img_mask = cv2.dilate(img_mask, kernel)
    #
    # image = cv2.bitwise_and(img, img, mask=cv2.bitwise_not(img_mask))
    #
    # # delete green
    # lower = np.array([50, 100, 100])
    # upper = np.array([90, 255, 255])
    # img_mask = cv2.inRange(hsv, lower, upper)
    # kernel = np.ones((2, 2), np.uint8)
    # img_mask = cv2.dilate(img_mask, kernel)
    # image = cv2.bitwise_and(image, img, mask=cv2.bitwise_not(img_mask))

    ret, bin = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU)

    # copy_to_clipboard(cv2_2_pil(bin))
    # cv2.imshow("img", bin)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #delete catheter
    img_mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)
    img_mask = cv2.circle(img_mask, (256, 256), 28, (255, 255, 255), thickness=-1)
    image = cv2.bitwise_and(image, img, mask=cv2.bitwise_not(img_mask))
    # cv2.imshow(l, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # find center
    green = np.where(img_mask != 0)
    top, bottom, left, right = 0, 0, 0, 0
    for i, (x, y) in enumerate(zip(green[0], green[1])):
        if i == 0:
            top, bottom = y, y
            right, left = x, x
        if y < top:
            top = y
        if y > bottom:
            bottom = y
        if x < left:
            left = x
        if x > right:
            right = x
    center = [int((left + right) / 2), int((top + bottom) / 2)]
    center = [int(img.shape[1] / 2), int(img.shape[0] / 2)]
    print(top, bottom, left, right, center)
    # cv2.drawMarker(img, (center[0], center[1]), (255, 0, 0))
    # cv2.imshow(l, img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # luminance = (0.298912 * image[:, :, 2] + 0.586611 * image[:, :, 1] + 0.114478 * image[:, :, 0])
    #
    # # 二値化処理
    # luminance[luminance < 80] = 0
    # luminance[luminance >= 80] = 255

    ret, luminance = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU)

    lumicolor = cv2.cvtColor(luminance.astype(np.float32), cv2.COLOR_GRAY2BGR)

    # ret, th = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 0, 255, cv2.THRESH_OTSU)
    # luminance = th



    # copy_to_clipboard(cv2_2_pil(luminance.astype(np.uint8)))
    # cv2.imshow("img", luminance)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    luminance = cv2.medianBlur(luminance, 15)
    # copy_to_clipboard(cv2_2_pil(luminance.astype(np.uint8)))
    # cv2.imshow("img", luminance)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # luminance = np.rot90(np.fliplr(luminance))

    # luminance = np.stack((luminance,) * 3, -1)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # circles(gray)
    gray[gray < 80] = 0
    gray[gray >= 80] = 255


    img_canny = cv2.Canny(gray, 100, 200)
    img_canny[img_canny < 80] = 0
    img_canny[img_canny >= 80] = 255
    # cv2.imshow(l, img_canny)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # cv2.imshow(l, gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    image = np.array(img)

    # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    # image = cv2.flip(image, 1)
    oimg = np.array(image)
    # cv2.imshow(l, oimg)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    endpoint = []
    notendpoint = []
    endangle = []
    r_endpoint = []
    r_endangle = []
    trace = []
    border = np.zeros((luminance.shape[0], luminance.shape[1], 3), np.uint8)

    maxlumi = 0
    mx, my = -1, -1
    mlangle = -1

    gray = np.array(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    # cv2.imshow(l, gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    for i in range(3):
        for j in range(360):
            tj = j * (360 / 360)
            # orgx = center[0] + (img.shape[1] / 2) * math.sin(math.radians(ti))
            # orgy = center[1] + (img.shape[0] / 2) * (-math.cos(math.radians(ti)))
            orgx = center[0] + (42 - i) * math.sin(math.radians(tj))
            orgy = center[1] + (42 - i) * (-math.cos(math.radians(tj)))
            # print(orgx, orgy)
            x, y = orgx, orgy

            # print(x, y)
            if gray[int(y), int(x)] >= maxlumi:
                # print(gray[int(x), int(y)])
                maxlumi = gray[int(y), int(x)]
                mx, my = int(x), int(y)
                mlangle = j

            # cv2.drawMarker(image, (int(x), int(y)), (255, 0, 0), markerSize=1)

        if maxlumi >= 240:
            break

    # print("maxlumi=", maxlumi)
    # print("mlangle=", mlangle)
    # cv2.drawMarker(image, (mx, my), (255, 0, 0))

    # copy_to_clipboard(cv2_2_pil(image))
    # cv2.imshow("img", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # luminance = cv2.rotate(luminance, cv2.ROTATE_90_CLOCKWISE)

    mhp = 2
    hp = copy.copy(mhp)
    x, y = center[0], center[1]
    end = [x, y]
    while (hp > 0):
        x += math.sin(math.radians(-135))
        y += (-math.cos(math.radians(-135)))
        # print(x, y)
        if x >= img.shape[1]:
            x = img.shape[1] - 1
            break
        if y >= img.shape[0]:
            y = img.shape[0] - 1
            break
        if x <= 0:
            x = 0
            break
        if y <= 0:
            y = 0
            break

        # print(distance.euclidean(center, (x, y)), (center[0] - left))
        #
        # if luminance[int(x), int(y)] != 0:
        #     # print(x, y)
        #     if distance.euclidean(center, (x, y)) >= (center[0] - left):
        #     # if (y < top or y > bottom) or (x < left or x > right):
        #         if abs(i - mlangle) >= 5:
        #             hp -= 1
        if luminance[int(y), int(x)] != 0:
            if abs(i - mlangle) <= 30:
                if distance.euclidean(center, (x, y)) >= 48:
                    hp -= 1
            else:
                if distance.euclidean(center, (x, y)) >= (center[0] - left):
                    hp -= 1
    # if hp == 0:
    #     x -= mhp*math.sin(math.radians(50))
    #     y -= mhp*(-math.cos(math.radians(50)))

    end = [int(x), int(y)]
    trace.append(end)
    trace.append(end)
    if hp > 0:
        endpoint.append(end)
        endangle.append(i)
    # print(center, end)
    # cv2.line(image, (center[0], center[1]), (end[0], end[1]), (0, 0, 255), thickness=1, lineType=cv2.LINE_4)
    # cv2.drawMarker(image, (center[0], center[1]), (255, 0, 0))

    # cv2.imshow(l, image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    def calc_angle(trace, center):
        # print(trace[-1])
        # image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        # print(np.array(trace[-1]), np.array(center))
        vec = np.array(trace[-1]) - np.array(center)
        # print(vec[0], vec[1])
        # print(np.arctan2(vec[0], -vec[1]))

        if np.arctan2(vec[0], -vec[1])*180/math.pi < 0:
            angle = 360 + np.arctan2(vec[0], -vec[1])*180/math.pi
        else:
            angle = np.round(np.arctan2(vec[0], -vec[1])*180/math.pi)

        # print(angle)
        return angle

    # 線を出す
    for i in range(360):
        hp = 2
        x, y = center[0], center[1]
        end = [x, y]
        while (hp > 0):
            x += math.sin(math.radians(i))
            y += (-math.cos(math.radians(i)))
            # print(x, y)
            if x >= img.shape[1]:
                x = img.shape[1] - 1
                break
            if y >= img.shape[0]:
                y = img.shape[0] - 1
                break
            if x <= 0:
                x = 0
                break
            if y <= 0:
                y = 0
                break

            # print(distance.euclidean(center, (x, y)), (center[0] - left))
            #
            # if luminance[int(x), int(y)] != 0:
            #     # print(x, y)
            #     if distance.euclidean(center, (x, y)) >= (center[0] - left):
            #     # if (y < top or y > bottom) or (x < left or x > right):
            #         if abs(i - mlangle) >= 5:
            #             hp -= 1
            if luminance[int(y), int(x)] != 0:
                if abs(i - mlangle) <= 30:
                    # if distance.euclidean(center, (x, y)) >= 48:
                    hp -= 1
                else:
                    # if distance.euclidean(center, (x, y)) >= (center[0] - left):
                    hp -= 1
        end = [int(x), int(y)]
        if hp > 0:
            endpoint.append(end)
            endangle.append(i)
        else:
            notendpoint.append(end)
        # print(center, end)
        cv2.line(image, (center[0], center[1]), (end[0], end[1]), (0, 0, 255), thickness=1, lineType=cv2.LINE_4)
    cv2.drawMarker(image, (center[0], center[1]), (255, 0, 0))

    # 逆から線を出す
    for i in range(360):
        hp = 1
        x = center[0] + math.sin(math.radians(i))*256
        y = center[1] + (-math.cos(math.radians(i)))*256
        if x >= img.shape[1]:
            x = img.shape[1] - 1
        if y >= img.shape[0]:
            y = img.shape[0] - 1
        if x <= 0:
            x = 0
        if y <= 0:
            y = 0
        start = [int(copy.copy(x)), int(copy.copy(y))]
        # print("start:", start, i)
        end = [x, y]
        while (hp > 0):
            x -= math.sin(math.radians(i))
            y -= (-math.cos(math.radians(i)))
            # print(x, y)
            if 255 < x < 257 and 255 < y < 257:
                x = 256
                y = 256
                break

            # print(distance.euclidean(center, (x, y)), (center[0] - left))
            #
            # if luminance[int(x), int(y)] != 0:
            #     # print(x, y)
            #     if distance.euclidean(center, (x, y)) >= (center[0] - left):
            #     # if (y < top or y > bottom) or (x < left or x > right):
            #         if abs(i - mlangle) >= 5:
            #             hp -= 1
            if luminance[int(y), int(x)] != 0:
                if abs(i - mlangle) <= 30:
                    # if distance.euclidean(center, (x, y)) >= 48:
                    hp -= 1
                else:
                    # if distance.euclidean(center, (x, y)) >= (center[0] - left):
                    hp -= 1
        end = [int(x), int(y)]
        # if hp > 0:
        #     r_endpoint.append(end)
        #     r_endangle.append(i)
        # print(center, end)
        if distance.euclidean(center, (x, y)) <= 35:
            r_endpoint.append(end)
            r_endangle.append(i)
            cv2.line(image, (end[0], end[1]), (start[0], start[1]), (0, 255, 0), thickness=1, lineType=cv2.LINE_4)
    cv2.drawMarker(image, (center[0], center[1]), (255, 0, 0))
    # for j in r_endpoint:
    #     vec = j - np.array(center)
    #     print(np.degrees(np.arctan2(vec[0], vec[1])))
    # print(r_endangle)
    # print(r_endpoint)
    # if abs(r_endangle[0] - r_endangle[-1]) >= 90:
    #     c = 0
    #     for i in range(len(r_endangle)):
    #         if abs(mlangle - r_endangle[i - c]) >= 30:
    #             del r_endpoint[i - c]
    #             del r_endangle[i - c]
    #             c += 1
    if len(r_endpoint) != 0:
        # print(tuple(r_endpoint[0]), tuple(r_endpoint[-1]))
        ea0 = [int(center[0] + math.sin(math.radians(r_endangle[0]))*300), int(center[1] + (-math.cos(math.radians(r_endangle[0])))*300)]
        ea1 = [int(center[0] + math.sin(math.radians(r_endangle[-1])) * 300), int(center[1] + (-math.cos(math.radians(r_endangle[-1])))*300)]
        pts = np.array((tuple(ea0), tuple(ea1), (center[0], center[1])))
        pts1 = None
        pts2 = None
        # (pts)
        image = dump_remover(image, circle_only=True)
        # copy_to_clipboard(cv2_2_pil(image))
        # cv2.imshow("img", image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # GWの影を描画
        r_ea0, r_ea1 = None, None
        r_ea00, r_ea01, r_ea10, r_ea11 = None, None, None, None
        # print("re1", len(r_endangle))
        for j in range(len(r_endangle)-2):
            if abs(r_endangle[j+1] - r_endangle[j]) >= 30:
                r_ea0, r_ea1 = np.split(r_endangle, [j+1])
                r_ea00 = [int(center[0] + math.sin(math.radians(r_ea0[0])) * 300),
                       int(center[1] + (-math.cos(math.radians(r_ea0[0]))) * 300)]
                r_ea01 = [int(center[0] + math.sin(math.radians(r_ea0[-1])) * 300),
                       int(center[1] + (-math.cos(math.radians(r_ea0[-1]))) * 300)]
                pts1 = np.array((tuple(r_ea00), tuple(r_ea01), (center[0], center[1])))
                r_ea10 = [int(center[0] + math.sin(math.radians(r_ea1[0])) * 300),
                          int(center[1] + (-math.cos(math.radians(r_ea1[0]))) * 300)]
                r_ea11 = [int(center[0] + math.sin(math.radians(r_ea1[-1])) * 300),
                          int(center[1] + (-math.cos(math.radians(r_ea1[-1]))) * 300)]
                pts2 = np.array((tuple(r_ea10), tuple(r_ea11), (center[0], center[1])))

                break

        if r_ea1 is not None:
            cv2.fillPoly(image, [pts1], (0, 255, 0))
            cv2.fillPoly(image, [pts2], (0, 255, 0))
        else:
            cv2.fillPoly(image, [pts], (0, 255, 0))
    # 血管内腔描画
    nep = []
    nep.append(notendpoint[0])
    s = 0
    for j in range(len(notendpoint)-2):
        if abs(distance.euclidean(notendpoint[j+1], center) - distance.euclidean(notendpoint[j-s], center)) <= 30:
            nep.append(notendpoint[j+1])
            s = 0
        else:
            s += 1

    # print(nep)
    nep = np.array(nep).reshape((-1, 1, 2)).astype(np.int32)
    # (nep)
    cv2.fillPoly(image, [nep], (0, 0, 255))
    lumen_mask1 = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8)
    cv2.fillPoly(lumen_mask1, [nep], (255, 255, 255))

    nep = []
    notendpoint = notendpoint[::-1]
    nep.append(notendpoint[0])
    s = 0
    for j in range(len(notendpoint) - 2):
        if abs(distance.euclidean(notendpoint[j + 1], center) - distance.euclidean(notendpoint[j - s], center)) <= 30:
            nep.append(notendpoint[j + 1])
            s = 0
        else:
            s += 1

    # print(nep)
    nep = np.array(nep).reshape((-1, 1, 2)).astype(np.int32)
    # print(nep)
    cv2.fillPoly(image, [nep], (0, 0, 255))
    lumen_mask2 = np.zeros((img.shape[0], img.shape[1], img.shape[2]), np.uint8)
    cv2.fillPoly(lumen_mask2, [nep], (255, 255, 255))

    lumen_mask = np.bitwise_or(lumen_mask1, lumen_mask2)

    # 重心描画
    lumen_mask = cv2.cvtColor(lumen_mask, cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(lumen_mask, 0, 255, cv2.THRESH_OTSU)
    mono_src = cv2.threshold(th, 48, 255, cv2.THRESH_BINARY_INV)[1]
    mono_src = cv2.bitwise_not(mono_src)
    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(mono_src)
    print(centroids[1])
    # cv2.drawMarker(image, (int(centroids[1, 0]), int(centroids[1, 1])), (255, 0, 0))
    cv2.circle(image, (int(centroids[1, 0]), int(centroids[1, 1])), 8, (255, 0, 0), thickness=-1)

    image = dump_remover(image, circle_only=True)
    # copy_to_clipboard(cv2_2_pil(image))
    # cv2.imshow("img", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # luminance = cv2.rotate(luminance, cv2.ROTATE_90_CLOCKWISE)
    # luminance = cv2.flip(luminance, 1)
    mimg = lhstack((lumicolor, image))
    # mimg = np.vstack((mimg, np.hstack((cv2.cvtColor(simg, cv2.COLOR_BGR2GRAY), cv2.cvtColor(simg, cv2.COLOR_BGR2GRAY)))))

    # 画像の拡大率を設定
    multiple = 1
    # オリジナル画像の高さ・幅を取得
    height = mimg.shape[0]
    width = mimg.shape[1]

    # 結果の表示
    # cv2.imshow(l, cv2.resize(mimg, (int(width * multiple), int(height * multiple))))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return image, centroids


# RING描画
def drawring(thickness):
    img_size = [512, 512]
    frame = 256
    rthick = 2

    out = np.zeros((img_size[0] + frame * 2, img_size[1] + frame * 2, 3), np.uint8)
    # out[:] = [45, 38, 39]
    print(out.shape)

    center = [int((img_size[0] + frame * 2) / 2), int((img_size[0] + frame * 2) / 2)]
    print(center)

    total_angle = 0
    for i in range(NUM_OF_ANGLES):
        ti = i * (360 / NUM_OF_ANGLES)
        orgx = center[0] + (img_size[0] / 2) * math.sin(math.radians(ti))
        orgy = center[1] + (img_size[0] / 2) * (-math.cos(math.radians(ti)))
        # print(orgx, orgy)
        x, y = orgx, orgy

        # x += thickness[i]*math.sin(math.radians(ti))
        # y -= thickness[i]*math.cos(math.radians(ti))
        # 円版
        x += 4 * math.sin(math.radians(ti))
        y -= 4 * math.cos(math.radians(ti))
        # print(i, orgx, x, orgy, y)

        # if thickness[i] != 0:
        cv2.line(out, (int(orgx), int(orgy)), (int(x), int(y)), ring_color(thickness[i]), thickness=rthick,
                 lineType=cv2.LINE_4)
        # else:
        #     cv2.line(out, (int(orgx), int(orgy)), (int(x), int(y)), (128, 128, 128), thickness=rthick, lineType=cv2.LINE_4)
    multiple = 1
    # リサイズ
    out = cv2.resize(out.astype(np.float32), (int(out.shape[1] * multiple), int(out.shape[0] * multiple)),
                     interpolation=cv2.INTER_NEAREST)

    # cv2.imshow("out", out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return out


def lumen(fname):
    img = cv2.imread(fname)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([50, 100, 100])  # 紫に近いピンク
    upper = np.array([90, 255, 255])  # 赤に近いピンク
    img_mask = cv2.inRange(hsv, lower, upper)
    img_color = cv2.bitwise_and(img, img, mask=img_mask)
    img_bgr = cv2.cvtColor(img_color, cv2.COLOR_HSV2BGR)

    # # 画像をグレースケールで読み込み
    # gray_src = cv2.imread(fname, 0)
    # cv2.imshow("color_src", gray_src)
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    # 二値化処理
    gray[gray < 127] = 0
    gray[gray >= 127] = 255

    # カーネルの定義
    kernel = np.ones((6, 6), np.uint8)

    # 膨張・収縮処理(方法2)
    result = cv2.dilate(gray, kernel)
    result = cv2.erode(result, kernel)
    result = cv2.dilate(result, kernel)
    result = cv2.erode(result, kernel)
    result = cv2.dilate(result, kernel)
    result = cv2.dilate(result, kernel)
    result = cv2.erode(result, kernel)

    # 前処理（平準化フィルターを適用した場合）
    # 前処理が不要な場合は下記行をコメントアウト
    # blur_src = cv2.GaussianBlur(gray_src, (5, 5), 2)

    # 二値変換
    # 前処理を使用しなかった場合は、blur_srcではなくgray_srcに書き換えるする
    mono_src = cv2.threshold(result, 48, 255, cv2.THRESH_BINARY_INV)[1]

    # ラベリング処理
    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(mono_src)

    # ラベリング結果書き出し準備
    color_src = cv2.cvtColor(mono_src, cv2.COLOR_GRAY2BGR)
    height, width = mono_src.shape[:2]
    outer_mask = color_src
    colors = []

    for j in range(1, ret + 1):
        colors.append(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))

    np.set_printoptions(threshold=1000000)
    # print(markers)

    # 各オブジェクトをランダム色でペイント
    for y in range(0, height):

        for x in range(0, width):
            if markers[y, x] != 0 and markers[y, x] != 1:
                color_src[y, x] = [0, 0, 0]
            else:
                color_src[y, x] = [255, 255, 255]
    color_src = cv2.erode(color_src, kernel)
    color_src = cv2.erode(color_src, kernel)
    color_src = cv2.bitwise_not(color_src)

    count = 0
    while len(stats) + count >= 3:
        gray = cv2.cvtColor(color_src, cv2.COLOR_BGR2GRAY)
        # 二値化処理
        ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
        mono_src = cv2.threshold(th, 48, 255, cv2.THRESH_BINARY_INV)[1]

        # mono_src = cv2.bitwise_not(mono_src)

        # cv2.imshow(fname, mono_src)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # ラベリング処理
        ret, markers, stats, centroids = cv2.connectedComponentsWithStats(mono_src)

        # print(stats)

        # ラベリング結果書き出し準備
        color_src = cv2.cvtColor(mono_src, cv2.COLOR_GRAY2BGR)
        height, width = mono_src.shape[:2]
        outer_mask = color_src
        colors = []

        for j in range(1, ret + 1):
            colors.append(np.array([random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]))

        np.set_printoptions(threshold=1000000)
        # print(markers)

        # 各オブジェクトをランダム色でペイント
        for y in range(0, height):

            for x in range(0, width):
                if markers[y, x] > 0 and stats[markers[y, x], 4] >= 2000:
                    color_src[y, x] = colors[markers[y, x]]
                else:
                    color_src[y, x] = [0, 0, 0]

        count = int(not count)

        # print(len(stats), count)

    # オブジェクトの総数を黄文字で表示
    # cv2.putText(color_src, str(ret - 1), (100, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255))

    # cv2.imwrite('octlumen/'+str(i)+'.jpg', color_src)

    gray = cv2.cvtColor(color_src, cv2.COLOR_BGR2GRAY)
    # 二値化処理
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    mono_src = cv2.threshold(th, 48, 255, cv2.THRESH_BINARY_INV)[1]
    mono_src = cv2.bitwise_not(mono_src)
    # mono_src = cv2.GaussianBlur(mono_src, (5, 5), 2)
    color_src = cv2.cvtColor(mono_src, cv2.COLOR_GRAY2BGR)

    contours, hierarchy = cv2.findContours(mono_src, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    cnt = []
    for contour in contours:
        epsilon = 0.008 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cnt.append(approx)

    cv2.drawContours(color_src, cnt, -1, (0, 0, 255), 1)

    # print(centroids[1])
    return centroids[1]

img_list = []
total_angle_list = []
max_angle_list = []
max_thickness_list = []

oimg_list = np.load('data/case_191/original.npy')
osimg_list = np.load('data/case_191/segm.npy')
calim_list = np.load('data/case_191/cal.npy')

count = 0

for i, (p1name, p1oname, p1cname) in enumerate(zip(osimg_list, oimg_list, calim_list)):
    # center = lumen(p1[62])
    center = [256, 256]
    oimg = oimg_list[i]
    osimg = osimg_list[i]
    img = np.array(osimg)
    HEIGHT, WIDTH = oimg.shape[1], oimg.shape[0]

    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # calim = extract_calcium(img)
    calim = calim_list[i]
    gt = cv2.cvtColor(calim, cv2.COLOR_BGR2GRAY)

    # luminance = (0.298912 * calim[:, :, 2] + 0.586611 * calim[:, :, 1] + 0.114478 * calim[:, :, 0])
    #
    # # 二値化処理
    # luminance[luminance < 80] = 0
    # luminance[luminance >= 80] = 255

    luminance = copy.copy(gt)
    luminance[luminance <= 10] = 0
    luminance[luminance > 10] = 255


    # luminance = np.rot90(np.fliplr(luminance))

    # cv2.imshow("img", luminance)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    print(center)

    _, centro = wire_region(oimg)

    thickness = []
    total_angle = 0
    for j in range(NUM_OF_ANGLES):
        tj = j * (360 / NUM_OF_ANGLES)
        # x, y = center[0], center[1]
        x, y = int(centro[1, 0]), int(centro[1, 1])
        calnum = 0
        end = [x, y]
        tmp_thickness = [0]

        while (1):
            x += math.sin(math.radians(tj))
            y -= math.cos(math.radians(tj))
            # print(x, y)
            if x >= img.shape[1]:
                x = img.shape[1] - 1
                break
            if y >= img.shape[0]:
                y = img.shape[0] - 1
                break
            if x <= 0:
                x = 0
                break
            if y <= 0:
                y = 0
                break

            if luminance[int(y), int(x)] != 0:
                tmp_thickness[calnum] += 1
            elif tmp_thickness[calnum] != 0:
                calnum += 1
                tmp_thickness.append(0)

        # print(j, x, y)
        thickness.append(max(tmp_thickness))

    print(thickness)
    print("Total Angle =", np.count_nonzero(thickness) * 360 / NUM_OF_ANGLES)
    print("Max Thickness =", max(thickness) * (9 / WIDTH), "mm")

    total_angle_list.append(int(np.count_nonzero(thickness) * 360 / NUM_OF_ANGLES))
    print(total_angle_list)
    max_thickness_list.append(max(thickness) * (9 / WIDTH))

    tcount = 0
    max_angle = 0
    for t in thickness:
        if t != 0:
            tcount += 1
            if tcount > max_angle:
                max_angle = tcount
        else:
            tcount = 0

    max_angle_list.append(int(max_angle * 360 / NUM_OF_ANGLES))
    print(max_angle_list)


    ring = drawring(thickness)

    # cs_img = cv2.resize(cs_img, (512, cs_img.shape[0]))

    img = addborder(osimg, 256)
    oimg = addborder(oimg, 256)
    calim = addborder(calim, 256)

    print(len(img), len(oimg), len(calim), len(ring))

    img[:] = np.where(np.any(ring != 0, axis=2, keepdims=True), ring, oimg)
    # img[:] = np.where(calim[:, :] != 0, calim, img)

    img = img[251:773, 251:773]
    cv2.drawMarker(img, (int(centro[1, 0]), int(centro[1, 1])), (0, 0, 255), markerSize=20, thickness=1)

    img_list.append(img)

    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    count += 1

    if i == 4:
        break

# cv2.imwrite('C:/Users/noeri/PycharmProjects/macro_editor/' + 'oct_ringtest2.jpg', img)

cs_list = []

# for i in range(361):
#     cs_img = cross_section(i)
#     cs_img = cv2.resize(cs_img, (512, 80))
#     cs_list.append(cs_img)
cs_img = cross_section()
cs_img = cv2.resize(cs_img, (570, 80))
css_img = cs_scale(len(oimg_list))
css_img = cv2.resize(css_img, (570, css_img.shape[0]))
cbar_img = mk_color_bar()

window_layout = [[sg.Image(filename='', key='image', pad=((0, 0), (0, 0)), background_color=('#27262d')), sg.Image(filename='', key='cbar', pad=((0, 0), (0, 0)), background_color=('#27262d'))],
                 [sg.Text('Max Angle', font=('Arial', 15), background_color=('#27262d')),
                  sg.Text('               ', font=('Arial', 40), background_color=('#27262d')),
                  sg.Text('Max Thickness', font=('Arial', 15), background_color=('#27262d'))],
                 [sg.Text('{:>3f}'.format(int(np.count_nonzero(thickness) * 360 / NUM_OF_ANGLES)), font=('Arial', 45),
                          text_color='#d57231', background_color=('#27262d'), key='angle'),
                  sg.Text('°', font=('Arial', 45), background_color=('#27262d'), text_color='#d57231', key='degree'),
                  sg.Text('              ', font=('Arial', 40), background_color=('#27262d')),
                  sg.Text('{:.2f}'.format(max(thickness) * (9 / WIDTH)), font=('Arial', 45),
                          background_color=('#27262d'), key='thickness'),
                  sg.Text('mm', font=('Arial', 15), background_color=('#27262d'))],
                 [sg.Slider(range=(0, count - 1), default_value=count // 2, size=(512, 15), orientation='h',
                            resolution=1, background_color=('#27262d'), key='sld1')],
                 [sg.Image(filename='', key='cs', pad=((4, 0), (0, 0)), background_color=('#27262d'))
                  # sg.Slider(range=(0, 360), default_value=180, size=(15, 80), orientation='v',
                  #           resolution=1, background_color=('#27262d'), key='sld2')
                  ],
                 [sg.Image(filename='', key='css', pad=((0, 0), (0, 0)), background_color=('#27262d'))]]
window = sg.Window('OCT', window_layout, size=(600, 820), background_color=('#27262d'))

first = True
while True:
    event, value = window.read(timeout=20)
    if first is True:
        imgbytes = cv2.imencode('.png', img_list[count // 2])[1].tobytes()
        window['image'].update(data=imgbytes)
        imgbytes_cs = cv2.imencode('.png', cs_img)[1].tobytes()
        window['cs'].update(data=imgbytes_cs)
        imgbytes_css = cv2.imencode('.png', css_img)[1].tobytes()
        window['css'].update(data=imgbytes_css)
        imgbytes_cbar = cv2.imencode('.png', cbar_img)[1].tobytes()
        window['cbar'].update(data=imgbytes_cbar)
        first = False

    # if event == 'sld1':
    window['image'].update(data=cv2.imencode('.png', img_list[int(value['sld1'])])[1].tobytes())
    window['angle'].update(max_angle_list[int(value['sld1'])])
    if max_angle_list[int(value['sld1'])] >= 180:
        window['angle'].update(text_color='#d57231')
        window['degree'].update(text_color='#d57231')
    else:
        window['angle'].update(text_color='#ffffff')
        window['degree'].update(text_color='#ffffff')
    window['thickness'].update(max_thickness_list[int(value['sld1'])])
    # window['cs'].update(data=cv2.imencode('.png', cs_list[int(value['sld2'])])[1].tobytes())

    cv2.destroyAllWindows()

window.close()

# print(center, end)
# cv2.line(img, (center[0], center[1]), (end[0], end[1]), (0, 0, 255), thickness=1, lineType=cv2.LINE_4)
# cv2.drawMarker(img, (center[0], center[1]), (255, 0, 0))