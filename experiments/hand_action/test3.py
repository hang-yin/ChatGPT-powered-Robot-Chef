import argparse
import time

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
import pyrealsense2 as rs


def main():

    # setup camera loop
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands

    pose = mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)

    # create realsense pipeline
    pipeline = rs.pipeline()

    width, height = 640, 480

    config = rs.config()
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)

    profile = pipeline.start(config)

    prev_frame_time = 0

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if not color_frame:
                break

            image = np.asanyarray(color_frame.get_data())

            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = pose.process(image)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            current_time = time.time()
            fps = 1 / (current_time - prev_frame_time)
            prev_frame_time = current_time

            cv2.putText(image, "FPS: %.0f" % fps, (7, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow('RealSense Pose Detector', image)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        pose.close()
        pipeline.stop()


if __name__ == "__main__":
    main()