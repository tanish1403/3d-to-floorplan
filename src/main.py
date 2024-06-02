import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_edges(gray):
    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Apply Canny edge detector
    edges = cv2.Canny(blurred, 100, 150)
    return edges

def detect_lines(edges):
    # Detect lines using the Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=25, maxLineGap=7)
    return lines

def find_joints(lines, edges):
    # Create an empty image to draw lines
    line_img = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), (255, 255, 255), 2)
    # Find intersections
    gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
    return corners

def draw_floor_plan(corners, lines, edges):
    # Create an image to draw the final floor plan
    floor_plan = np.zeros((edges.shape[0], edges.shape[1], 3), dtype=np.uint8)
    if corners is not None:
        corners = np.int0(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(floor_plan, (x, y), 5, (255, 0, 0), -1)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(floor_plan, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return floor_plan

def main():
    image_path = 'data/test.jpg'
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    edges = detect_edges(gray)
    lines = detect_lines(edges)
    corners = find_joints(lines, edges)
    floor_plan = draw_floor_plan(corners, lines,edges)

    plt.imshow(cv2.cvtColor(floor_plan, cv2.COLOR_BGR2RGB))
    plt.title('2D Floor Plan')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()
