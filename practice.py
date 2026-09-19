import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=10)
draw = mp.solutions.drawing_utils

while True:
    success, frame = cap.read()

    if not success:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    total_fingers = 0

    if results.multi_hand_landmarks:
        for hand, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness
        ):
            draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            label = handedness.classification[0].label

            # Thumb
            if label == "Right":
                if hand.landmark[4].x < hand.landmark[3].x:
                    total_fingers += 1
            else:
                if hand.landmark[4].x > hand.landmark[3].x:
                    total_fingers += 1

            # Index, middle, ring and pinky
            for tip, joint in [(8, 6), (12, 10), (16, 14), (20, 18)]:
                if hand.landmark[tip].y < hand.landmark[joint].y:
                    total_fingers += 1

    cv2.putText(
        frame,
        "Fingers: " + str(total_fingers),
        (50, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        2,
        (255, 200, 255),
        3
    )

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
