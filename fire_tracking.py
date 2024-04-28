import numpy as np
import cv2
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import util  # Import the util module containing motion control functions
import RPi.GPIO as GPIO  # Import the RPi.GPIO module

# Set pin numbering mode
GPIO.setmode(GPIO.BCM)

# Initialize GPIO pins
util.init_gpio()

# Define the conversion factor (pixels per centimeter)
conversion_factor = 10  # Adjust this value based on your calibration

# Initialize camera
with PiCamera() as camera:
    camera.resolution = (960, 540)
    camera.framerate = 30
    rawCapture = PiRGBArray(camera, size=(960, 540))

    # Allow the camera to warm up
    time.sleep(0.1)

    # Initialize variables
    prev_fire_state = False
    robot_position = [0, 0]  # Initial position of the robot
    target_distance = 50  # Distance from the fire for the robot to maintain
    tolerance = 0.1  # Tolerance for robot movement

    for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
        frame = frame.array

        # Convert BGR image to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define the lower and upper bounds for the color red (adjust as needed)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([20, 255, 255])

        # Threshold the HSV image to get only red colors
        mask = cv2.inRange(hsv, lower_red, upper_red)

        # Apply morphological operations to remove noise
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Check if fire is detected
        fire_detected = False
        for contour in contours:
            # Calculate contour area
            area = cv2.contourArea(contour)

            # If the area is large enough, consider it as fire
            if area > 1000:
                fire_detected = True

                # Draw a bounding box around the fire
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Calculate centroid of the bounding box
                fire_center_x = x + w // 2
                fire_center_y = y + h // 2

                # Calculate the distance between the robot and the fire (in pixels)
                distance_to_fire = np.sqrt((robot_position[0] - fire_center_x) ** 2 + (robot_position[1] - fire_center_y) ** 2)

                # Convert distance to real-world units (centimeters)
                distance_to_fire_cm = distance_to_fire / conversion_factor

                # Update robot's position based on the relative distance to the fire
                if distance_to_fire_cm > target_distance:
                    # Calculate deviation from the center
                    x_deviation = (fire_center_x - 480) / 480
                    y_deviation = (fire_center_y - 270) / 270

                    # Determine movement direction based on deviation
                    if abs(x_deviation) > tolerance:
                        if x_deviation > 0:
                            util.left()
                            cv2.putText(frame, "Move Left", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        else:
                            util.right()
                            cv2.putText(frame, "Move Right", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    elif abs(y_deviation) > tolerance:
                        if y_deviation > 0:
                            util.forward()
                            cv2.putText(frame, "Move Forward", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                        else:
                            util.backward()
                            cv2.putText(frame, "Move Backward", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    # Stop the robot if the fire is close enough
                    util.stop()

                # Display the distance on the frame
                cv2.putText(frame, f"Distance to fire: {distance_to_fire_cm:.2f} cm", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                break

        # Print message if fire detection state changes
        if fire_detected != prev_fire_state:
            if fire_detected:
                print("Fire detected!")
            else:
                print("No fire detected!")
            prev_fire_state = fire_detected

        # Display the frame
        cv2.imshow("Frame", frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Clear the stream for the next frame
        rawCapture.truncate(0)

    # Close all OpenCV windows
    cv2.destroyAllWindows()

# Cleanup GPIO pins
GPIO.cleanup()
