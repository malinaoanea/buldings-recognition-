import cv2
import os

currentframe = 0
def take_pisc(cam, current_frame):
    while(True):
        
        # reading from frame
        ret,frame = cam.read()

        if ret:
            # if video is still left continue creating images
            name = './poze_din_videoclipuri/frame' + str(current_frame) + '.jpg'
            print ('Creating...' + name)
    
            # writing the extracted images
            cv2.imwrite(name, frame)
    
            # increasing counter so that it will
            # show how many frames are created
            current_frame += 1
        else:
            break
    
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()
    return current_frame

cam1 = cv2.VideoCapture("videos/cladire5togo.mp4")

try:
      
    # creating a folder named data
    if not os.path.exists('poze_din_videoclipuri'):
        os.makedirs('poze_din_videoclipuri')
  
# if not created then raise error
except OSError:
    print ('Error: Creating directory of data')
  
# frame=

currentframe =  take_pisc(cam1, currentframe)
currentframe = take_pisc(cam2, currentframe)

