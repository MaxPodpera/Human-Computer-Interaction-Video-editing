import cv2
import numpy as np
from moviepy.editor import *

EDGE_COLOR = [0, 255, 0]
BLUR = 501
EDIT = True
ENHANCE = False
white = None


def edit_frame(img):
    edges = cv2.Canny(img.copy(), 100, 200)

    fin = cv2.GaussianBlur(img.copy(), (BLUR, BLUR), 0)

    fin = cv2.addWeighted(fin, 0.4, white, 0.1, 0)

    fin[np.where(edges == 255)] = EDGE_COLOR

    return fin


if __name__ == '__main__':
    in_vid = cv2.VideoCapture(sys.argv[1])
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    fps = in_vid.get(cv2.CAP_PROP_FPS)
    width, height, frames = int(in_vid.get(cv2.CAP_PROP_FRAME_WIDTH)), int(in_vid.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(in_vid.get(cv2.CAP_PROP_FRAME_COUNT))

    out_vid_edges = cv2.VideoWriter("out_edges_" + sys.argv[1], fourcc, fps, (width, height), 1)

    white = np.zeros((height, width, 3), np.uint8)
    white[:, :, :] = [220, 220, 220]

    i = 0

    while in_vid.isOpened():
        ret, frame = in_vid.read()
        if not ret:
            break

        edges = edit_frame(frame.copy())
        edges = cv2.cvtColor(edges.copy(), cv2.COLOR_RGB2BGR)

        out_vid_edges.write(edges)

        i += 1
        print(str(round((i / frames) * 100, 2)) + "%", end="\r")

    in_vid.release()
    out_vid_edges.release()
    cv2.destroyAllWindows()
    print("Done!  ")
