import cv2


class CustomVideoParser:
    '''
    Take videos from webcam
    '''

    def __init__(self):
        self.video = None

    def validate_video_stream(self):
        '''
        Validate if the webcam is available
        '''
        for i in range(-1,3):
            cap = cv2.VideoCapture(i)
            if cap is not None or cap.isOpened():
                print('[+] Available Camera Source Found :', i)
                return True, i
        return False,None

