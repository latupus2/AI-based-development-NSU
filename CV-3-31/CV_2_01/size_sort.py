import os
import cv2
from CV_2_01.lab_1.coins_contour_detection import counting_contours

def sorted_conturs(image_bgr_input, min_size, max_size):
    """
    Выполняет поиск и фильтрацию контуров на изображении, 
    отрисовывает только подходящие контуры и сохраняет результат.

    Parameters
    ----------
    image_path : str
        Путь к исходному изображению.
    min_size : int
        Минимальная площадь контура для фильтрации.
    max_size : int
        Максимальная площадь контура для фильтрации.

    Returns
    -------
    int
        Количество найденных объектов, площадь которых находится в пределах [min_size, max_size].
        Возвращает 0 в случае ошибки.
    """
    try:
        image_bgr = image_bgr_input.copy()
        img_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        contours = counting_contours(img_gray, 50, 150, 127, 255, min_size, max_size)
        cv2.drawContours(image_bgr, contours, -1, (0, 255, 0), 3)

        return len(contours), image_bgr

    except FileNotFoundError as e:
        print("File error:", e)
        return 0
    except cv2.error as e:
        print("OpenCV error:", e)
        return 0
    except Exception as e:
        print("Unexpected error:", e)
        return 0


if __name__ == "__main__":
    count = sorted_conturs("input_data/res.png", "output_data/image_1.png", 100, 5000)
    print("Objects count:", count)
