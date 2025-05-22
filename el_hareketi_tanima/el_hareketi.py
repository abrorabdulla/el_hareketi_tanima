import cv2
import mediapipe as mp
import time

# MediaPipe el tanıma modülü
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

# Video dosyası yolu
video_path = r"D:\Users\abror\PycharmProjects\PythonProject\el_hareketi_tanima\video.mp4"
cap = cv2.VideoCapture(video_path)

playing = True
like_given = False
last_action_time = 0
cooldown = 3  # saniye

# El hareketini tanıma fonksiyonu
def get_hand_gesture(landmarks):
    thumb_tip = landmarks[4].y
    thumb_ip = landmarks[3].y
    index_tip = landmarks[8].y
    middle_tip = landmarks[12].y
    ring_tip = landmarks[16].y
    pinky_tip = landmarks[20].y

    palm_base_y = landmarks[0].y

    if all(tip > palm_base_y for tip in [index_tip, middle_tip, ring_tip, pinky_tip]) and thumb_tip > palm_base_y:
        return "fist"
    if all(tip < palm_base_y for tip in [index_tip, middle_tip, ring_tip, pinky_tip]):
        return "open"
    if thumb_tip < thumb_ip:
        return "like"
    if thumb_tip > thumb_ip:
        return "dislike"
    return None

# Webcam başlat (el hareketi algılama için)
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Webcam açılamadı!")
    exit()

while cap.isOpened():
    if playing:
        ret, frame = cap.read()
        if not ret:
            print("Video bitti!")
            break
    else:
        ret, frame = cap.read()
        cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_POS_FRAMES) - 1)

    frame = cv2.resize(frame, (640, 480))
    cv2.imshow("Video", frame)

    ret_cam, cam_frame = cam.read()
    cam_frame = cv2.flip(cam_frame, 1)
    rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = None
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(cam_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            gesture = get_hand_gesture(hand_landmarks.landmark)

    current_time = time.time()
    if gesture and (current_time - last_action_time > cooldown):
        if gesture == "fist" and playing:
            print("✊ Video durduruldu!")
            playing = False
            last_action_time = current_time
        elif gesture == "open" and not playing:
            print("🖐 Video devam ediyor!")
            playing = True
            last_action_time = current_time
        elif gesture == "like" and not like_given:
            print("👍 Video beğenildi!")
            like_given = True
            last_action_time = current_time
        elif gesture == "dislike":
            print("👎 Video beğenilmedi!")
            like_given = False
            last_action_time = current_time

    cv2.putText(cam_frame, f"Gesture: {gesture if gesture else 'None'}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Webcam", cam_frame)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cam.release()
cv2.destroyAllWindows()