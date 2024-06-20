import cv2
import numpy as np


image_path = 'frame_1080720_resized.png'


frame = cv2.imread(image_path)


if frame is None:
    print("Error: Couldn't load the image.")
    exit()


def distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def draw_polygon(event, x, y, flags, param):
    global polygons, drawing, current_polygon
    if event == cv2.EVENT_LBUTTONDOWN:
        if not drawing:
            polygons.append([]) 
            drawing = True

        if len(polygons[-1]) > 0:

            if distance((x, y), polygons[-1][0]) < 10: 
                drawing = False 
                return
        polygons[-1].append((x, y)) 
    elif event == cv2.EVENT_RBUTTONDOWN:
        drawing = False 


cv2.namedWindow('Draw Polygons')
cv2.setMouseCallback('Draw Polygons', draw_polygon)


polygons = []
drawing = False
box_id = 1

while True:
  
    frame_resized = cv2.resize(frame, (1080, 720))


    frame_draw = frame_resized.copy()


    for polygon in polygons:
        if len(polygon) > 1:
            cv2.polylines(frame_draw, [np.array(polygon)], isClosed=True, color=(0, 255, 0), thickness=2)


    cv2.imshow('Draw Polygons', frame_draw)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):
        with open('coordinates.txt', 'w') as f:
            for polygon in polygons:
                f.write(f"a{box_id} {' '.join(str(coord) for point in polygon for coord in point)}\n")
                box_id += 1
        print("Polygon coordinates saved to 'coordinates.txt'")
 
    elif key == ord('q'):
        break


cv2.destroyAllWindows()
