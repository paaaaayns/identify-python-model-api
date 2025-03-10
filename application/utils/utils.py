import cv2
import numpy as np

def circular_shift(arr, shift):
    return np.roll(arr, shift, axis=1)

def bbox_to_circle(x1, y1, x2, y2):
    print("Converting Bounding Box to Circle...")
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    radius = (x2 - x1) / 2.0
    return center_x, center_y, radius

def apply_clahe(image):
    """Apply CLAHE to enhance contrast."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def sharpen_image(image):
    """Sharpen image to enhance iris details."""
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(image, -1, kernel)

def denoise_image(image):
    """Apply bilateral filter to reduce noise while preserving details."""
    return cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

def preprocess_image(image):
    """Full preprocessing pipeline."""
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # image = apply_clahe(image)
    # image = denoise_image(image)
    # image = sharpen_image(image)
    return image

def normalize_iris(image, mask, iris_pupil_info, output_size=(64, 360)):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    iris_info, pupil_info = iris_pupil_info
    iris_center_x, iris_center_y, iris_radius = iris_info
    pupil_center_x, pupil_center_y, pupil_radius = pupil_info

    output_height, output_width = output_size

    theta = np.linspace(0, 2 * np.pi, output_width)
    r = np.linspace(0, 1, output_height)

    theta, r = np.meshgrid(theta, r)

    x_pupil = pupil_center_x + pupil_radius * np.cos(theta)
    y_pupil = pupil_center_y + pupil_radius * np.sin(theta)

    x_iris = iris_center_x + iris_radius * np.cos(theta)
    y_iris = iris_center_y + iris_radius * np.sin(theta)

    x = ((1 - r) * x_pupil + r * x_iris).astype(np.float32)
    y = ((1 - r) * y_pupil + r * y_iris).astype(np.float32)

    normalized_iris = cv2.remap(image, x, y, interpolation=cv2.INTER_LINEAR)
    normalized_mask = cv2.remap(mask, x, y, interpolation=cv2.INTER_NEAREST)

    normalized_iris_masked = cv2.bitwise_and(normalized_iris, normalized_iris, mask=normalized_mask)
    
    return normalized_iris, normalized_mask, normalized_iris_masked

def localize_iris(image, iris_pupil_info):
    iris_info, pupil_info = iris_pupil_info
    iris_center_x, iris_center_y, iris_radius = iris_info
    pupil_center_x, pupil_center_y, pupil_radius = pupil_info

    cv2.circle(image, (round(iris_center_x), round(iris_center_y)), round(iris_radius), (0, 255, 0), 2)

    cv2.circle(image, (round(pupil_center_x), round(pupil_center_y)), round(pupil_radius), (0, 0, 255), 2)

    return image

def extract_iris_code(normalized_iris, normalized_mask):
    ksize = (9, 9)
    sigma = 2.0
    theta = 0
    lambd = 8.0
    gamma = 0.5
    psi = 0

    gabor_real = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F)
    gabor_imag = cv2.getGaborKernel(ksize, sigma, theta, lambd, gamma, psi + np.pi/2, ktype=cv2.CV_32F)

    iris_real = cv2.filter2D(normalized_iris, cv2.CV_32F, gabor_real)
    iris_imag = cv2.filter2D(normalized_iris, cv2.CV_32F, gabor_imag)

    iris_code_real = (iris_real > 0).astype(np.uint8)
    iris_code_imag = (iris_imag > 0).astype(np.uint8)

    iris_code = np.stack((iris_code_real, iris_code_imag), axis=-1)

    kernel = np.ones((3, 3), np.uint8)

    refined_mask = cv2.morphologyEx(normalized_mask, cv2.MORPH_CLOSE, kernel)

    iris_code_real = cv2.morphologyEx(iris_code_real, cv2.MORPH_OPEN, kernel)
    iris_code_imag = cv2.morphologyEx(iris_code_imag, cv2.MORPH_OPEN, kernel)

    iris_code = np.stack((iris_code_real, iris_code_imag), axis=-1)

    mask_code = (refined_mask > 0).astype(np.uint8)

    return iris_code, mask_code

def iris_code_to_image(iris_code):
    rows, cols, _ = iris_code.shape

    iris_image = np.zeros((rows * 2, cols), dtype=np.uint8)

    iris_image[:rows, :] = iris_code[:, :, 0] * 255
    iris_image[rows:, :] = iris_code[:, :, 1] * 255

    return iris_image

def compute_shifted_hamming_distance(code1, mask1, code2, mask2, max_shift=20, threshold=0.38):
    best_hamming_distance = 1.0
    best_shift = 0
    
    mask1 = np.expand_dims(mask1, axis=-1)
    mask2 = np.expand_dims(mask2, axis=-1)
    
    for shift in range(-max_shift, max_shift + 1):
        shifted_code2 = circular_shift(code2, shift)
        shifted_mask2 = circular_shift(mask2, shift)
        
        valid_mask = np.bitwise_and(mask1, shifted_mask2)
        xor_result = np.bitwise_xor(code1, shifted_code2)
        xor_masked = np.bitwise_and(xor_result, valid_mask)
        
        total_bits = np.sum(valid_mask)
        differing_bits = np.sum(xor_masked)
        
        if total_bits > 0:
            hamming_distance = differing_bits / total_bits
        else:
            hamming_distance = 1.0
        
        if hamming_distance < best_hamming_distance:
            best_hamming_distance = hamming_distance
            best_shift = shift
    
    match_status = "Match ✅" if best_hamming_distance <= threshold else "No Match ❌"
    return best_hamming_distance, best_shift, match_status