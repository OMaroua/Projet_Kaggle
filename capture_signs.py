import cv2
import mediapipe as mp
import pandas as pd
import os
import zipfile

# Get the current working directory
current_working_directory = os.getcwd()
print(f"Current Working Directory: {current_working_directory}")

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic

def create_frame_landmark_df(frame_number, results):
    # Initialize DataFrames for each type of landmark with additional columns
    columns = ['frame', 'row_id', 'type', 'landmark_index', 'x', 'y', 'z']
    face = pd.DataFrame(columns=columns)
    left_hand = pd.DataFrame(columns=columns)
    pose = pd.DataFrame(columns=columns)
    right_hand = pd.DataFrame(columns=columns)

     
    expected_face_landmarks = 468
    expected_hand_landmarks = 21
    expected_pose_landmarks = 33

    
    def populate_df(df, landmarks, label, frame_number, expected_landmarks):
        if landmarks:
            for i, point in enumerate(landmarks.landmark):
                row_id = f"{frame_number}-{label}-{i}"
                df = pd.concat([df, pd.DataFrame([[frame_number, row_id, label, i, point.x, point.y, point.z]], columns=columns)], ignore_index=True)
        
        while len(df) < expected_landmarks:
            df = pd.concat([df, pd.DataFrame([[frame_number, f"{frame_number}-{label}-{len(df)}", label, len(df), None, None, None]], columns=columns)], ignore_index=True)
        return df

    # Process landmarks for each type
    face = populate_df(face, results.face_landmarks, 'face', frame_number, expected_face_landmarks)
    left_hand = populate_df(left_hand, results.left_hand_landmarks, 'left_hand', frame_number, expected_hand_landmarks)
    right_hand = populate_df(right_hand, results.right_hand_landmarks, 'right_hand', frame_number, expected_hand_landmarks)
    pose = populate_df(pose, results.pose_landmarks, 'pose', frame_number, expected_pose_landmarks)

    # Concatenate all DataFrames and handle NaN values
    landmarks = pd.concat([face, left_hand, pose, right_hand], ignore_index=True)
    landmarks.fillna(value=pd.NA, inplace=True)
    return landmarks

def do_capture_loop():
    all_landmarks = []

    cap = cv2.VideoCapture(0)
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        frame = 0
        while cap.isOpened():
            frame += 1
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)

            landmarks = create_frame_landmark_df(frame, results)

            all_landmarks.append(landmarks)

            # Draw landmark annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # Drawing the landmarks on image
            mp_drawing.draw_landmarks(
                image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            mp_drawing.draw_landmarks(
                image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
            mp_drawing.draw_landmarks(
                image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            cv2.imshow('MediaPipe Holistic', cv2.flip(image, 1))
            if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
                break
    cap.release()
    cv2.destroyAllWindows()
    print(all_landmarks)
    return all_landmarks

if __name__=="__main__":
    zf = zipfile.ZipFile('asl-signs.zip') 
    pq_file = zf.open('train_landmark_files/16069/10042041.parquet')
    xyz = pd.read_parquet(zf.open('train_landmark_files/16069/10042041.parquet'))   
    xyz_skel = xyz.query(' frame == 83')[['type', 'landmark_index']]. copy ()
    landmarks = do_capture_loop()
    landmarks = pd.concat(landmarks).reset_index(drop=True).to_parquet('output.parquet')


