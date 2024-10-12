import pygame
import keypress as kp
from djitellopy import Tello
from time import sleep
import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO("yolo11n.pt")

# Initialize Pygame
pygame.init()

# Set up the display
WINDOW_WIDTH, WINDOW_HEIGHT = 960, 720
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Tello Drone Control")

# Initialize the Tello drone
kp.init()
tello = Tello()
tello.connect()
tello.streamon()

print(f"Battery: {tello.get_battery()}%")

def get_keyboard_input():
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        lr = -speed
    elif keys[pygame.K_RIGHT]:
        lr = speed

    if keys[pygame.K_UP]:
        fb = speed
    elif keys[pygame.K_DOWN]:
        fb = -speed

    if keys[pygame.K_w]:
        ud = speed
    elif keys[pygame.K_s]:
        ud = -speed

    if keys[pygame.K_a]:
        yv = -speed
    elif keys[pygame.K_d]:
        yv = speed

    if keys[pygame.K_q]:
        tello.land()
        sleep(3)
    if keys[pygame.K_e]:
        tello.takeoff()

    return [lr, fb, ud, yv]

# Main loop for both control and object detection
running = True
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Get keyboard input for drone control
    vals = get_keyboard_input()
    tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    # Get the current frame from the Tello camera
    frame_read = tello.get_frame_read()
    frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))

    # Perform object detection using YOLO
    results = model.predict(frame, conf=0.5)

    # Draw bounding boxes and labels on the frame
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{result.names[int(box.cls[0])]} {box.conf[0]:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    # Convert the frame to a Pygame surface
    frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))

    # Display the frame on the Pygame window
    screen.blit(frame_surface, (0, 0))
    pygame.display.flip()

    clock.tick(30)  # Limit the frame rate to 30 FPS

# Clean up
tello.land()
tello.streamoff()
pygame.quit()