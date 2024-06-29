import cv2
import mediapipe as mp
import util
import pyautogui
import random
from pynput.mouse import Button,Controller 

mouse=Controller()
screenWidth,screenHeight=pyautogui.size()
mouse=Controller()
mpHands=mp.solutions.hands
hands=mpHands.Hands(static_image_mode=False,model_complexity=1,min_detection_confidence=0.7,min_tracking_confidence=0.7,max_num_hands=1)

def find_finger_tip(processed):
    if processed.multi_hand_landmarks:
        hand_landmarks=processed.multi_hand_landmarks[0]
        return hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP]
    return None

def move_mouse(indexFingerTip):
    if indexFingerTip is not None:
        x=int(indexFingerTip.x * screenWidth)
        y=int(indexFingerTip.y * screenHeight)
        pyautogui.moveTo(x,y)

def isLeftCLick(landmarks_list,thumb_index_dist):
    return(util.getAngle(landmarks_list[5],landmarks_list[6],landmarks_list[8])<50 and
           util.getAngle(landmarks_list[9],landmarks_list[10],landmarks_list[12])>90 and
           thumb_index_dist>50
           )

def isRightCLick(landmarks_list,thumb_index_dist):
    return(util.getAngle(landmarks_list[9],landmarks_list[10],landmarks_list[12])<50 and
           util.getAngle(landmarks_list[5],landmarks_list[6],landmarks_list[8])>90 and
           thumb_index_dist>50
           )

def isDoubleCLick(landmarks_list,thumb_index_dist):
    return(util.getAngle(landmarks_list[5],landmarks_list[6],landmarks_list[12])<50 and
           util.getAngle(landmarks_list[9],landmarks_list[10],landmarks_list[8])<50 and
           thumb_index_dist>50
           )


def isScreenshot(landmarks_list,thumb_index_dist):
    return(util.getAngle(landmarks_list[9],landmarks_list[10],landmarks_list[12])<50 and
           util.getAngle(landmarks_list[5],landmarks_list[6],landmarks_list[8])<50 and
           thumb_index_dist<50
           )

def detectGestures(frame,landmarks_list,processed):
    if len(landmarks_list)>=21:
        indexFingerTip=find_finger_tip(processed)
        thumb_index_dist=util.getDistance([landmarks_list[4],landmarks_list[5]])

        if thumb_index_dist<50 and util.getAngle(landmarks_list[4],landmarks_list[6],landmarks_list[8])>90:
            move_mouse(indexFingerTip)
        #print(indexFingerTip)

        #left click
        elif isLeftCLick(landmarks_list,thumb_index_dist):
            mouse.press(Button.left)
            mouse.release(Button.left)
            cv2.putText(frame,"Left click",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        #Right click
        elif isRightCLick(landmarks_list,thumb_index_dist):
            mouse.press(Button.right)
            mouse.release(Button.right)
            cv2.putText(frame,"Right click",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

        # #Double click
        elif isDoubleCLick(landmarks_list,thumb_index_dist):
            pyautogui.doubleClick()
            cv2.putText(frame ,"Double Click",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,255),1)

        #Screen shot click
        elif isScreenshot(landmarks_list,thumb_index_dist):
            im1=pyautogui.screenshot()
            label=random.randint(1,1000)
            im1.save(f'my_screenshot_{label}.png')    
            cv2.putText(frame ,"Screenshot",(50,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),1)
            


def main():
    cap=cv2.VideoCapture(0)
    draw=mp.solutions.drawing_utils
    try:
        while cap.isOpened():
            ret,frame=cap.read()

            if not ret:
                break
            frame=cv2.flip(frame,1)
            frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            processed=hands.process(frameRGB)

            landmarks_list=[]

            if processed.multi_hand_landmarks:
                hand_landmarks=processed.multi_hand_landmarks[0]
                draw.draw_landmarks(frame,hand_landmarks,mpHands.HAND_CONNECTIONS)

                for lm in hand_landmarks.landmark:
                    landmarks_list.append((lm.x,lm.y))

            detectGestures(frame,landmarks_list,processed)
            #print(landmarks_list)

            cv2.imshow("Frame",frame)
            if cv2.waitKey(1) & 0xFF==ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__=='__main__':
    main()

