import cv2
import numpy as np

def convert(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    detect_edge = cv2.adaptiveThreshold(blur_image, 255,
                                        cv2.ADAPTIVE_THRESH_MEAN_C,
                                        cv2.THRESH_BINARY, 11, 2)
    output = cv2.bitwise_and(image, image, mask=detect_edge)
    return output

def create_pop_art(image, max_dots=250, multiplier=100, background_colour=[50, 205, 50], dots_colour=[255, 255, 0]):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    original_image_height, original_image_width = gray_image.shape
    aspect_ratio = original_image_width / original_image_height

    if original_image_height > original_image_width:
        downsized_height = max_dots
        downsized_width = int(max_dots * aspect_ratio)
    else:
        downsized_width = max_dots
        downsized_height = int(max_dots / aspect_ratio)
        
    downsized_image = cv2.resize(gray_image, (downsized_width, downsized_height))
    downsized_image_height, downsized_image_width = downsized_image.shape

    blank_img_height = downsized_image_height * multiplier
    blank_img_width = downsized_image_width * multiplier
    padding = int(multiplier / 2)
    blank_image = np.full((blank_img_height, blank_img_width, 3), background_colour, dtype=np.uint8)

    for y in range(downsized_image_height):
        for x in range(downsized_image_width):
            cv2.circle(blank_image, (x * multiplier + padding, y * multiplier + padding),
                       int(0.7 * multiplier * (255 - downsized_image[y, x]) / 255), dots_colour, -1)
    
    return blank_image

def main():
    
    image_path = r"/content/tm.jpg"  #Provide an imahe path here

    original_image = cv2.imread(image_path)
    if original_image is None:
        print("Error: Image not found at the specified path.")
        return

    cartoon_image = convert(original_image)
    pop_art_image = create_pop_art(original_image)
    pop_art_resized = cv2.resize(pop_art_image, (cartoon_image.shape[1], cartoon_image.shape[0]))
    blended_image = cv2.addWeighted(cartoon_image, 0.5, pop_art_resized, 0.5, 0)

    
    cv2.imshow("Blended Image", blended_image)

    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("Blended_Output.png", blended_image)
    print("Blended image saved successfully")

if __name__ == "__main__":
    main()
