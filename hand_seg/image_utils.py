import cv2

def convert_opencv_input(image, color_space):
    """Convert images read using OpenCv (BGR order) to the chosen color space.
    Args:
      image: A Image object read using cv2 (BGR order).
      color_space: the chosen output color space.

    Returns:
      The image converted to the chosen color space.
    """
    if color_space.lower() == 'rgb':       
      out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    elif color_space.lower() == 'lab':
      out = cv2.cvtColor(image, cv2.COLOR_BGR2Lab) 
    elif color_space.lower() == 'l':
      lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
      L,A,B = cv2.split(lab_image)
      out = cv2.merge([L, L, L])
    return out