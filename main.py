import os
import numpy as np
import cv2
import copy
from datetime import datetime

def setup_video(path):
    cap = cv2.VideoCapture(path) # for video
    cap.set(cv2.CAP_PROP_FPS, 30)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    video_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    timestamp = datetime.now()
    filename = "outputs/output - " + str(timestamp) + ".avi"

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, fourcc, 30, (int(video_width), int(video_height)))

    return cap, out

def main():
    video_path = os.path.join('videos', 'persons.mp4')
    cap, out = setup_video(video_path)
    
    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

    first_iteration_indicator = 1
    while cap.isOpened():
        
        if (first_iteration_indicator == 1):
            ret, frame = cap.read()
            first_frame = copy.deepcopy(frame)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            accum_image = np.zeros((height, width), np.uint8)
            first_iteration_indicator = 0
        else:
            ret, frame = cap.read()  # read a frame
            if ret == False: 
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # convert to grayscale

            fgmask = fgbg.apply(gray)  # remove the background

            # apply a binary threshold only keeping pixels above thresh and setting the result to maxValue.  If you want
            # motion to be picked up more, increase the value of maxValue.  To pick up the least amount of motion over time, set maxValue = 1
            thresh = 2
            maxValue = 2
            ret, th1 = cv2.threshold(fgmask, thresh, maxValue, cv2.THRESH_BINARY)

            # add to the accumulated image
            accum_image = cv2.add(accum_image, th1) 
            # apply a color map
            # COLORMAP_PINK also works well, COLORMAP_BONE is acceptable if the background is dark
            color_image = cv2.applyColorMap(accum_image, cv2.COLORMAP_HOT)
            result_overlay = cv2.addWeighted(frame, 0.7, color_image, 0.7, 0)
            
            cv2.imshow('frame', result_overlay)
            out.write(result_overlay) 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cleanup
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__=='__main__':
    main()