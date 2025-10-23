import cv2

# Функция-заглушка для трекбара
def nothing(x):
    pass

# Запрашиваем путь к изображению у пользователя
image_path = input("Введите путь к изображению: ").strip()

# Загружаем изображение в градациях серого
img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("Ошибка: Не удалось загрузить изображение.")
    exit(1)

# Создаем окно
cv2.namedWindow('Binarization')

# Создаем трекбары для порога и максимального значения
cv2.createTrackbar('Threshold', 'Binarization', 127, 255, nothing)
cv2.createTrackbar('Maxval', 'Binarization', 255, 255, nothing)

while True:
    # Получаем текущие значения трекбаров
    thresh_val = cv2.getTrackbarPos('Threshold', 'Binarization')
    maxval = cv2.getTrackbarPos('Maxval', 'Binarization')
    
    # Применяем бинаризацию
    _, binary_img = cv2.threshold(img, thresh_val, maxval, cv2.THRESH_BINARY)
    
    # Показываем результат
    cv2.imshow('Binarization', binary_img)
    
    # Выход по нажатию ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()