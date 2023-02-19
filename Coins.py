import argparse
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def main(args):
    DIRECTORY = args.input_image
    coins_images = []
    for entry in os.scandir(DIRECTORY):
        if entry.path.endswith('.jpg') and entry.is_file():
            try:
                img_data = cv2.imread(entry.path)
                coins_images.append(img_data)
            except Exception:
                print('Error reading image: {}'.format(entry.path))
    j = 1
    for i in range(len(coins_images)):
        smaller_coins, bigger_coins = find_coins(coins_images[i], j)
        tray_array, h, w, f = find_tray(coins_images[i], j)
        print(f"zdjęcie {j}")
        print(f'Wymiary zdjęcia: {h}px x {w}px, pole: {f}px^2')
        calculate(bigger_coins, smaller_coins, tray_array)
        j += 1


def find_coins(image, iterator):
    # zmiana na skalę szarości
    conv_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # pozbycie się zewnętrznych pikseli przez rozmycie
    blur = cv2.GaussianBlur(conv_img, (5, 5), 2)
    # obliczenie zakresu do stworzenia krawędzi na małych obszarach obrazu
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    # operacje morfologiczne
    kernel = np.ones((4, 3), np.uint8)
    k_open = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    # wykrycie krawędzi
    outline = cv2.Canny(k_open, 100, 100, L2gradient=True)
    # wykrycie okręgów przy użyciu transformaty Hougha
    coins = cv2.HoughCircles(outline, cv2.HOUGH_GRADIENT, 1.55, param1=100, param2=92, minDist=20, minRadius=5,
                             maxRadius=39)
    smaller_coins = []
    if coins is not None:
        coins = np.round(coins[0, :]).astype("int")
        for (x, y, r) in coins:
            cv2.circle(image, (x, y), r, (0, 0, 0), 6)
            smaller_coins.append((x, y))
    else:
        print("No circles found")
    # posortowanie okręgów po promieniu
    coins = coins[np.argsort(coins[:, 2])]
    biger_coins = [[coins[-1][0], coins[-1][1]], [coins[-2][0], coins[-2][1]]]
    smaller_coins.remove((biger_coins[0][0], biger_coins[0][1]))
    smaller_coins.remove((biger_coins[1][0], biger_coins[1][1]))
    cv2.circle(image, (coins[-1][0], coins[-1][1]), coins[-1][2], (0, 0, 255), 6)
    cv2.circle(image, (coins[-2][0], coins[-2][1]), coins[-2][2], (0, 0, 255), 6)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Zdjęcie {iterator}")
    plt.show()
    return smaller_coins, biger_coins


def find_tray(image, iterator):
    # zmiana na skalę szarości
    conv_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # pozbycie się zewnętrznych pikseli przez rozmycie
    blur = cv2.GaussianBlur(conv_img, (5, 5), 2)
    # obliczenie zakresu do stworzenia krawędzi na małych obszarach obrazu
    threshold = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    # operacje morfologiczne
    kernel = np.ones((4, 3), np.uint8)
    k_open = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)
    # wykrycie krawędzi
    outline = cv2.Canny(k_open, 100, 100, L2gradient=True)
    # wykrycie lini przy użyciu transformaty Hougha
    lines = cv2.HoughLinesP(outline, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=20)
    arr = np.zeros(4)
    i = 0
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x_max = max(x1, x2)
        y_max = max(y1, y2)
        x_min = min(x1, x2)
        y_min = min(y1, y2)
        if i == 0:
            arr[0] = x_max
            arr[1] = y_max
            arr[2] = x_min
            arr[3] = y_min
        i += 1
        if x_max > arr[0]:
            arr[0] = x_max
        if y_max > arr[1]:
            arr[1] = y_max
        if x_min < arr[2]:
            arr[2] = x_min
        if y_min < arr[3]:
            arr[3] = y_min
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 6)
    cv2.rectangle(image, (int(arr[2]), int(arr[1])), (int(arr[0]), int(arr[3])), (255, 255, 255), 3)
    # print(arr)
    width = (arr[0] - arr[2])
    height = arr[1] - arr[3]
    field = width * height
    #print(f'width= {width}, hight = {height}, field = {field}')
    #print(conv_img.shape)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Zdjęcie {iterator}")
    plt.show()
    return arr, width, height, field


def calculate(bigger_coins, smaller_coins, tray_arr):
    bigger_coins_in_tray = []
    bigger_coins_outer_tray = []
    smaller_coins_in_tray = []
    smaller_coins_outer_tray = []
    for c in bigger_coins:
        if np.logical_and(c[0] > tray_arr[2], c[0] < tray_arr[0]) & np.logical_and(c[1] > tray_arr[3],
                                                                                   c[1] < tray_arr[1]):
            bigger_coins_in_tray.append(c)
        else:
            bigger_coins_outer_tray.append(c)
    for c in smaller_coins:
        if np.logical_and(c[0] > tray_arr[2], c[0] < tray_arr[0]) & np.logical_and(c[1] > tray_arr[3],
                                                                                   c[1] < tray_arr[1]):
            smaller_coins_in_tray.append(c)
        else:
            smaller_coins_outer_tray.append(c)
    print(f"Na tacy jest {len(bigger_coins_in_tray)} monet o nominale 5zł")
    print(f"Poza tacą jest {len(bigger_coins_outer_tray)} monet o nominale 5zł")
    print(f"Na tacy jest {len(smaller_coins_in_tray)} monet o nominale 5gr")
    print(f"Poza tacą jest {len(smaller_coins_outer_tray)} monet o nominale 5gr\n")
    print(f"Suma monet na tacy wynosi: {len(bigger_coins_in_tray) * 5 + len(smaller_coins_in_tray) * 0.05} zł")
    print(
        f"Suma monet na tacy wynosi: {len(bigger_coins_outer_tray) * 5 + len(smaller_coins_outer_tray) * 0.05} zł\n\n")

#parametr przyjmowany -i Coins
def parse_arguments():
    parser = argparse.ArgumentParser(description='This script identifies the coins in the tray with OpenCV.'
                                                 ' To run script you need to provide the path to the image as -i argument.')
    parser.add_argument('-i',
                        '--input_image',
                        type=str,
                        required=True,
                        help='Input image')
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_arguments())
