from sre_parse import State
from vidgear.gears import CamGear
import cv2
import os
import uuid

options = {"STREAM_RESOLUTION": "720p"}

stream = CamGear(source='https://www.youtube.com/watch?v=LOUpPQ-iJIo', stream_mode = True, logging=True, **options).start()

IMAGES_PATH = os.path.join('./data/images/pol/1')

imagesCaptured = 0

while True:
    frame = stream.read()
    if frame is None:
        break

    cv2.imshow("LiveStream", frame)    
    cv2.waitKey(10)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        imagesCaptured +=1
        print('Images captured: ' + str(imagesCaptured))
        cv2.imwrite(os.path.join(IMAGES_PATH,'{}.jpg'.format(str(uuid.uuid1()))), frame)

    if key == ord("q"):
        break


cv2.destroyAllWindows()
stream.stop()