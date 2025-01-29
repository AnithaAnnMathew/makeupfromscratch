import cv2
import dlib
import numpy as np
import tkinter as tk
from tkinter import colorchooser
# from scipy.interpolate import interp1d

# # Define facial regions
# regions = {
#     "face_shape_right": list(range(0, 8)),
#     "face_shape_left": list(range(8, 17)),
#     "right_eyebrow": list(range(17, 22)),
#     "left_eyebrow ": list(range(22, 27)),
#     "nose": list(range(27, 36)),
#     "right_eye upper": list(range(36, 40)),
#     "right_eye lower": list(range(39, 42)),
#     "left_eye upper": list(range(42, 46)),
#     "left_eye lower": list(range(45, 48)),
#     "outer_upper_lips": list(range(48, 55)),
#     "outer_lower_lips": list(range(54, 60)),
#     "inner_upper_lips": list(range(60, 64)),  
#     "inner_lower_lips": list(range(64, 68)),
# }





# # Apply lipstick

def apply_lipstick(outer_lips, inner_lips, image, color, intensity):

      # Combine upper and lower lip landmarks
    lips_landmarks = outer_lips + inner_lips
    points = np.array(lips_landmarks, dtype=np.int32)

    image_copy=image.copy()

    # Fill the lips with black

    # cv2.fillPoly(image_copy, [points], (0,0,0))

    # Create a mask for the lips
    lips_mask = np.zeros_like(image, dtype=np.uint8)
    cv2.fillPoly(lips_mask, [points], color)
    blur_mask=cv2.GaussianBlur(lips_mask,(15,15),0)

    # Combine the mask and original iblended_imagemage
    blended_image = cv2.addWeighted(image_copy, 1, blur_mask, intensity, 0)

    result = cv2.addWeighted(image, 0.7, blended_image, 0.3, 0)
   
    return result

    #  Apply eyeshadow
def apply_eyeshadow(upper_eyelid_points, lower_eyelid_points, lower_eyebrow_points, image, color, intensity):
    # Calculate midpoints between upper eyelid and lower eyebrow
    midpoints = [
        (0.5 * (u[0] + b[0]), 0.5 * (u[-1] + b[-1]))
        for u, b in zip(upper_eyelid_points, lower_eyebrow_points)
    ]

    # Combine upper eyelid points and midpoints to form the eyeshadow region
    eyeshadow_region = np.array(upper_eyelid_points + midpoints[::-1], dtype=np.int32)

    # Define the eye region using upper and lower eyelid points
    eye_region = np.array(upper_eyelid_points + lower_eyelid_points[::-1], dtype=np.int32)

    # Create masks for the eyeshadow and the eye region
    mask = np.zeros_like(image)
    eye_mask = np.zeros_like(image)

    # Draw the border around the eyeshadow region
    # cv2.polylines(mask, [cv2.convexHull(eyeshadow_region)], isClosed=True, color=(255,0,0), thickness=2)


    # Fill the eyeshadow region
    cv2.fillConvexPoly(mask, cv2.convexHull(eyeshadow_region), color)


    # Fill the eye region
    cv2.fillConvexPoly(eye_mask, cv2.convexHull(eye_region), (255, 255, 255))  # White mask for the eye

    # Subtract the eye region from the eyeshadow mask
    mask = cv2.subtract(mask, eye_mask)

    # Blur the mask for smooth blending
    blurred_mask = cv2.blur(mask, (15, 15), 0)

    # Blend the mask with the original image
    result = cv2.addWeighted(image, 1, blurred_mask, intensity, 0)

    return result



def apply_blush(cheekbone,mouth_corner, image, color, intensity):  


    # Compute midpoint for natural blush placement
    cheek_center = (
        (cheekbone[0] + mouth_corner[0]) // 2,
        (cheekbone[1] + mouth_corner[1]) // 2
    )
    

    # Define ellipse size for blush
    blush_width = abs(cheekbone[0] - mouth_corner[0]) // 2
    blush_height = blush_width // 1.5  # Slightly flatter ellipse
    print(blush_width, blush_height)

    # Create a mask
    mask = np.zeros_like(image)


    cv2.circle(mask, cheek_center, 15, color, -1)
    # blurred_blush = cv2.GaussianBlur(mask, (35, 35), 0)
    blurred_blush = cv2.blur(mask,(35,35),0)

    return cv2.addWeighted(image, 1, blurred_blush, intensity, 0)


    # # Draw blush as an ellipse on the cheeks
    # cv2.ellipse(mask, cheek_center, (int(blush_width), int(blush_height)), 0, 0, 360, color, -1)


    # # Apply Gaussian blur for smooth blending
    # blurred_mask = cv2.GaussianBlur(mask, (35, 35), 0)

    # # Blend the blush with the original image
    # result = cv2.addWeighted(image, 1, blurred_mask, intensity, 0)

    # return result


def pick_color(msg):
    """Pick a color using a color chooser and return it in BGR format."""
    color_code = colorchooser.askcolor(title=msg)[1]
    if color_code:
        print(f"Selected Color: {color_code}")
        hex_color = color_code.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        # print(f"RGB: ({r}, {g}, {b}) | BGR: ({b}, {g}, {r})")
        return (b, g, r)  # OpenCV uses BGR


# Main function
def main():
    # Read the image
    img=cv2.imread("000506.png")
    img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Show the image
    cv2.imshow('Original image',img)
    # cv2.imshow('Gray image',img_gray)

    #Find the face
    detector=dlib.get_frontal_face_detector()
    predictor=dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    faces=detector(img_gray)
    if len(faces) == 0:
        print("No face detected in the image.")
    # print(faces)

    # for face in faces:
    #     x1,y1=face.left(),face.top()
    #     x2,y2=face.right(), face.bottom()
    #     img=cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
    # cv2.imshow('found face',img)

        # # Get facial landmarks
        # landmarks = predictor(img_gray, face)

        # # Annotate landmarks on the image
        # for i in range(68):  # 68 landmark points
        #     x = landmarks.part(i).x
        #     y = landmarks.part(i).y
        #     cv2.circle(img, (x, y), 2, (0, 255, 0), cv2.FILLED)  # Draw a circle for each landmark
        #     cv2.putText(img, str(i), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 0), 1)  # Label the points
        # cv2.imshow('Marked face',img)
        # cv2.imwrite("out.png",img)

    # Process the first detected face
    face = faces[0]
    landmarks = predictor(img_gray, face)
    # print(type(landmarks))
    # Extract lip landmarks (upper and lower lips)
    outer_upper_lip_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 55)]
    inner_upper_lip_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(64, 59,-1)]
    outer_lower_lip_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(54, 61)]
    inner_lower_lip_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(67, 63,-1)] 
    right_eyebrow_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 22)] 
    left_eyebrow_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(22, 27)] 
    right_eyeu_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 40)] 
    left_eyeu_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 46)] 
    right_eyel_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(39, 42)] 
    left_eyel_landmarks = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(45, 48)] 
    left_cheekbone = (landmarks.part(3).x, landmarks.part(3).y) 
    right_cheekbone = (landmarks.part(13).x, landmarks.part(13).y) 
    left_eye_corner = (landmarks.part(41).x, landmarks.part(41).y) 
    right_eye_corner = (landmarks.part(46).x, landmarks.part(46).y) 

    # print(right_eyebrow_landmarks)

    # print(right_eyeu_landmarks)
    # Launch color picker for each makeup type
    root = tk.Tk()
    root.withdraw()   

    msg1 = "Pick a color for lipstick"
    lipstick_color = pick_color(msg1)

    # Apply lipstick
    # lipstick_color = (0,255,0)
    img = apply_lipstick(outer_upper_lip_landmarks, inner_upper_lip_landmarks, img, lipstick_color, 0.8)
    img = apply_lipstick(outer_lower_lip_landmarks, inner_lower_lip_landmarks, img, lipstick_color, 0.8)
    cv2.imshow("Makeup Image", img)

    msg2 = "Pick a color for eyeshadow"
    eyeshadow_color = pick_color(msg2)

    img = apply_eyeshadow(right_eyeu_landmarks, right_eyel_landmarks, right_eyebrow_landmarks, img, eyeshadow_color, 0.3)
    img = apply_eyeshadow(left_eyeu_landmarks, left_eyel_landmarks, left_eyebrow_landmarks, img, eyeshadow_color, 0.3)
    
    cv2.imshow("Makeup Image", img)

    msg3 = "Pick a color for blush"
    blush_color = pick_color(msg3)
  

    img = apply_blush(left_cheekbone, left_eye_corner, img, blush_color, 0.5)
    img = apply_blush(right_cheekbone, right_eye_corner, img, blush_color, 0.5)

    cv2.imshow("Makeup Image", img)

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
