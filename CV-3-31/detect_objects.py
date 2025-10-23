import cv2
import numpy as np
import argparse
import os
import sys

from CV_1_17.brightness_and_contrast import adjust_brightness_contrast, load_image
from CV_2_01.size_sort import sorted_conturs


def load_image_bgr(image_path):
    img_rgb = load_image(image_path)
    return cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

def detect_objects_dark_image(image_bgr_input, min_area=100, max_area=5000, contrast_alpha=2.0, brightness_beta=30):
    """
    Обнаружение объектов на затемненном изображении.
    
    Parameters:
        image_bgr_input: Изображение в формате BGR
        min_area (int): Минимальная площадь контура
        max_area (int): Максимальная площадь контура
        contrast_alpha (float): Коэффициент контраста
        brightness_beta (float): Коэффициент яркости
        
    Returns:
        tuple: (count_of_objects, enhanced_image)
    """
    try:
        image_bgr = image_bgr_input.copy()
        
        enhanced_img_bgr = np.zeros_like(image_bgr)
        for i in range(3):
            enhanced_img_bgr[:, :, i] = adjust_brightness_contrast(
                image_bgr[:, :, i], 
                alpha=contrast_alpha, 
                beta=brightness_beta
            )
         
        count, image_bgr_with_contoures = sorted_conturs(enhanced_img_bgr, min_area, max_area)
        print(f"Количество объектов на изображении: {count}")
        
        return count, image_bgr_with_contoures
        
    except Exception as e:
        print(f"Ошибка при обработке изображения: {e}")
        return -1, None

def display_comparison(original_img_bgr, enhanced_img_bgr):
    """
    Визуальное сравнение исходного и улучшенного изображений с помощью OpenCV.
    Объединяет два изображения в одно и отображает.
    
    Parameters:
        original_img_bgr: Исходное изображение в формате BGR
        enhanced_img_bgr: Улучшенное изображение в формате BGR  
    """
    original_img = original_img_bgr.copy()
    enhanced_img = enhanced_img_bgr.copy()

    height = min(original_img.shape[0], enhanced_img.shape[0])
    
    if original_img.shape[0] != height:
        scale = height / original_img.shape[0]
        new_width = int(original_img.shape[1] * scale)
        original_img = cv2.resize(original_img, (new_width, height))
    
    if enhanced_img.shape[0] != height:
        scale = height / enhanced_img.shape[0]
        new_width = int(enhanced_img.shape[1] * scale)
        enhanced_img = cv2.resize(enhanced_img, (new_width, height))

    combined_img = np.hstack([original_img, enhanced_img])
    
    line_pos = original_img.shape[1]
    cv2.line(combined_img, (line_pos, 0), (line_pos, height), (255, 255, 255), 2)
    
    cv2.imshow("Object Detection Comparison", combined_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calculate_detection_percentage(detected_count, expected_count):
    """
    Рассчитывает процент найденных объектов относительно ожидаемого количества.

    Parameters:
        detected_count (int): Количество обнаруженных объектов
        expected_count (int): Ожидаемое количество объектов
        
    Returns:
        float: Процент найденных объектов
    """
    if expected_count <= 0:
        return 0.0

    percentage = (detected_count / expected_count) * 100
    return round(percentage, 2)

def main():
    """
    Основная функция для обработки изображений из командной строки.
    """
    parser = argparse.ArgumentParser(description='Обнаружение объектов на затемненных изображениях')
    parser.add_argument('input_image', help='Путь к исходному изображению')
    parser.add_argument('--output', help='Путь для сохранения результата (опционально)')
    parser.add_argument('--min_area', type=int, default=100, help='Минимальная площадь объекта (по умолчанию: 100)')
    parser.add_argument('--max_area', type=int, default=5000, help='Максимальная площадь объекта (по умолчанию: 5000)')
    parser.add_argument('--contrast', type=float, default=2.0, help='Коэффициент контраста (по умолчанию: 2.0)')
    parser.add_argument('--brightness', type=float, default=30, help='Коэффициент яркости (по умолчанию: 30)')
    parser.add_argument('--show', action='store_true', help='Показать сравнение изображений')
    parser.add_argument('--test_count', type=int, help='Количество объектов на изображении (Для проверки работы программы)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_image):
        print(f"Ошибка: Файл '{args.input_image}' не существует.")
        sys.exit(1)

    image = load_image_bgr(args.input_image)
    count, enhanced_with_contours = detect_objects_dark_image(
        image,
        args.min_area,
        args.max_area,
        args.contrast,
        args.brightness
    )
    
    if enhanced_with_contours is not None:
        if args.test_count is not None:
            percentage = calculate_detection_percentage(count, args.test_count)
            print(f"Найдено: {percentage}% ({count}/{args.test_count})")

        if args.output:
            cv2.imwrite(args.output, enhanced_with_contours)
            print(f"\nРезультат сохранен в: {args.output}")

        if args.show:
            display_comparison(image, enhanced_with_contours)
    else:
        print("\nОшибка при обработке изображения")

if __name__ == "__main__":
    main()