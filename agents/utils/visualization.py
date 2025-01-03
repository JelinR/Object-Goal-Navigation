import cv2
import numpy as np

#Returns vertices for a pointer-shaped polygon 
#For visualizing current location
#Refer to check.ipynb for visualizations
def get_contour_points(pos, origin, size=20):
    x, y, o = pos

    #New origin point wrt the origin coords
    pt1 = (int(x) + origin[0],
           int(y) + origin[1])
    

    pt2 = (int(x + size / 1.5 * np.cos(o + np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o + np.pi * 4 / 3)) + origin[1])
    
    #pt3 is the diagonal point opposite to the origin (pt1), with diagonal length being 'size' and the orientation being the angle with x axis
    #This is the reason why size is multiplied by cos(o) and sin(o) to get the x_shift and y_shift
    pt3 = (int(x + size * np.cos(o)) + origin[0],
           int(y + size * np.sin(o)) + origin[1])
    

    pt4 = (int(x + size / 1.5 * np.cos(o - np.pi * 4 / 3)) + origin[0],
           int(y + size / 1.5 * np.sin(o - np.pi * 4 / 3)) + origin[1])

    return np.array([pt1, pt2, pt3, pt4])

#Draws a line of width (w) on mat from start till end with spaced steps
def draw_line(start, end, mat, steps=25, w=1):
    
    #Spaced steps or points
    for i in range(steps + 1):
        x = int(np.rint(start[0] + (end[0] - start[0]) * i / steps))
        y = int(np.rint(start[1] + (end[1] - start[1]) * i / steps))
        mat[x - w:x + w, y - w:y + w] = 1
    return mat

#Defines two boxes, one for Observation (live screen) and other for predicted Semantic Map
#Check example.gif from docs for reference
def init_vis_image(goal_name, legend):
    vis_image = np.ones((655, 1165, 3)).astype(np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 1
    color = (20, 20, 20)  # BGR
    thickness = 2

    text = "Observations (Goal: {})".format(goal_name)
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = (640 - textsize[0]) // 2 + 15
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    text = "Predicted Semantic Map"
    textsize = cv2.getTextSize(text, font, fontScale, thickness)[0]
    textX = 640 + (480 - textsize[0]) // 2 + 30
    textY = (50 + textsize[1]) // 2
    vis_image = cv2.putText(vis_image, text, (textX, textY),
                            font, fontScale, color, thickness,
                            cv2.LINE_AA)

    # draw outlines
    color = [100, 100, 100]

    #Observations box (480x640) - position top left
    vis_image[49, 15:655] = color         #Top margin
    vis_image[50:530, 14] = color         #Left margin
    vis_image[50:530, 655] = color        #Right margin
    vis_image[530, 15:655] = color        #Bottom margin

    #Predicted Semantic Map box (480x480) - position top right
    vis_image[49, 670:1150] = color       #Top margin
    vis_image[50:530, 669] = color        #Left margin
    vis_image[50:530, 1150] = color       #Right margin
    vis_image[530, 670:1150] = color      #Bottom margin

    # draw legend
    lx, ly, _ = legend.shape
    vis_image[537:537 + lx, 155:155 + ly, :] = legend

    return vis_image
