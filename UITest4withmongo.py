import pygame
import cv2
import numpy as np
from djitellopy import Tello
from ultralytics import YOLO
import time
from mon import insert_into_db 

# Initialize YOLO model
model = YOLO("yolo11n.pt")

# Define classes we want to detect (humans and animals)
target_classes = ['person', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe']

# People counter
num_people = 0

# Initialize Pygame
pygame.init()

# Set up the display
WINDOW_WIDTH, WINDOW_HEIGHT = 1280, 720
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Tello Drone Control - Forager Edition")

# Initialize fonts (using default font as a fallback for pixel font)
try:
    font = pygame.font.Font("PixeloidSans-Bold.ttf", 18)
    small_font = pygame.font.Font("PixeloidSans-Bold.ttf", 14)
except:
    font = pygame.font.Font(None, 24)
    small_font = pygame.font.Font(None, 20)

# Colors (Forager inspired palette)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BEIGE = (242, 233, 225)
LIGHT_BLUE = (155, 214, 255)
DARK_BLUE = (52, 109, 201)
GREEN = (99, 199, 16)
YELLOW = (252, 191, 73)
RED = (226, 88, 87)

# Initialize the Tello drone
tello = Tello()
tello.connect()
tello.streamon()

print(f"Battery: {tello.get_battery()}%")

# Initialize drone position
drone_x, drone_y = 0, 0
map_scale = 5  # 1 unit of movement = 5 pixels on the map

def draw_forager_rect(surface, color, rect, border_radius=10):
    x, y, w, h = rect

    # Create a surface with per-pixel alpha
    shape_surf = pygame.Surface((w, h), pygame.SRCALPHA, 32)
    pygame.draw.rect(shape_surf, color, (0, 0, w, h), border_radius=border_radius)

    # Draw the main shape
    surface.blit(shape_surf, (x, y))

    # Draw the border
    pygame.draw.rect(surface, BLACK, rect, width=2, border_radius=border_radius)

def draw_button(text, x, y, w, h, color, text_color=BLACK):
    draw_forager_rect(screen, color, (x, y, w, h))
    text_surface = font.render(text, True, text_color)
    text_rect = text_surface.get_rect(center=(x + w / 2, y + h / 2))
    screen.blit(text_surface, text_rect)

def get_keyboard_input():
    global drone_x, drone_y
    lr, fb, ud, yv = 0, 0, 0, 0
    speed = 50

    keys = pygame.key.get_pressed()

    if keys[pygame.K_LEFT]:
        lr = -speed
        drone_x -= 1
    elif keys[pygame.K_RIGHT]:
        lr = speed
        drone_x += 1

    if keys[pygame.K_UP]:
        fb = speed
        drone_y += 1
    elif keys[pygame.K_DOWN]:
        fb = -speed
        drone_y -= 1

    if keys[pygame.K_w]:
        ud = speed
    elif keys[pygame.K_s]:
        ud = -speed

    if keys[pygame.K_a]:
        yv = -speed
    elif keys[pygame.K_d]:
        yv = speed

    return [lr, fb, ud, yv]

def draw_map(surface, drone_x, drone_y):
    map_width, map_height = 280, 280
    map_x, map_y = 980, 400

    # Calculate the map's center position
    center_x = map_x + map_width // 2
    center_y = map_y + map_height // 2

    # Draw map background
    draw_forager_rect(surface, WHITE, (map_x, map_y, map_width, map_height))

    # Draw grid lines (adjusted to move with the drone)
    for i in range(-14, 15):
        x = center_x + (i * 20) - (drone_x * map_scale) % 20
        y = center_y - (i * 20) + (drone_y * map_scale) % 20
        pygame.draw.line(surface, LIGHT_BLUE, (x, map_y), (x, map_y + map_height))
        pygame.draw.line(surface, LIGHT_BLUE, (map_x, y), (map_x + map_width, y))

    # Draw drone position (always at the center)
    pygame.draw.circle(surface, RED, (center_x, center_y), 5)

    # Draw detected people
    for person_x, person_y in detected_people:
        person_map_x = center_x + (person_x - drone_x) * map_scale
        person_map_y = center_y - (person_y - drone_y) * map_scale
        if map_x <= person_map_x < map_x + map_width and map_y <= person_map_y < map_y + map_height:
            pygame.draw.circle(surface, GREEN, (int(person_map_x), int(person_map_y)), 3)

    # Draw coordinates
    coord_text = small_font.render(f"X: {drone_x}, Y: {drone_y}", True, BLACK)
    surface.blit(coord_text, (map_x + 10, map_y + map_height + 10))

def is_close_to_existing_person(x, y, detected_people, threshold=100):
    for px, py in detected_people:
        if ((x - px) ** 2 + (y - py) ** 2) ** 0.5 < threshold:
            return True
    return False

# Main loop
running = True
clock = pygame.time.Clock()
is_flying = False
detected_objects = []
detected_people = []

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            if 1040 < x < 1240 and 20 < y < 70:
                if is_flying:
                    tello.land()
                    time.sleep(100)
                    insert_into_db(num_people)
                    is_flying = False
                else:
                    tello.takeoff()
                    is_flying = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q and is_flying:
                tello.land()
                is_flying = False
                mon.insert_into_db(num_people)
            elif event.key == pygame.K_e and not is_flying:
                tello.takeoff()
                is_flying = True

    # Get keyboard input for drone control
    vals = get_keyboard_input()
    tello.send_rc_control(vals[0], vals[1], vals[2], vals[3])

    # Get the current frame from the Tello camera
    frame_read = tello.get_frame_read()
    frame = frame_read.frame
    frame = cv2.resize(frame, (960, 720))

    # Perform object detection using YOLO
    results = model.predict(frame, conf=0.5)
    detected_objects = []
    new_detected_people = []  # Temporary list to store newly detected people

    # Process detected objects
    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]

            if class_name in target_classes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {box.conf[0]:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                detected_objects.append(label)

                # If a person is detected, check proximity before adding to the list
                if class_name == 'person':
                    num_people += 1
    
                    # Estimate the person's position relative to the drone
                    person_x = drone_x + (x1 + x2) / 2 - 480  # Assuming the frame width is 960
                    person_y = drone_y - (y1 + y2) / 2 + 360  # Assuming the frame height is 720
                    scaled_x, scaled_y = person_x / 20, person_y / 20

                    if not is_close_to_existing_person(scaled_x, scaled_y, new_detected_people):
                        new_detected_people.append((scaled_x, scaled_y))



    # Update the detected_people list
    detected_people = new_detected_people

    # Convert the frame to a Pygame surface
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = np.rot90(frame)
    frame = pygame.surfarray.make_surface(frame)
    screen.blit(frame, (0, 0))

    # Draw UI elements
    draw_forager_rect(screen, BEIGE, (960, 0, 320, 720))

    # Battery status
    battery = tello.get_battery()
    battery_text = font.render(f"Battery: {battery}%", True, BLACK)
    screen.blit(battery_text, (980, 90))
    draw_forager_rect(screen, WHITE, (980, 120, 200, 20))
    draw_forager_rect(screen, YELLOW, (980, 120, battery * 2, 20))

    # Flight status and takeoff/land button
    status_text = font.render("Status: " + ("Flying" if is_flying else "Landed"), True, BLACK)
    screen.blit(status_text, (980, 150))
    draw_button("Land (Q)" if is_flying else "Take Off (E)", 1040, 20, 200, 50, RED if is_flying else GREEN)

    # Control instructions
    controls = [
        "Controls:",
        "Arrows: Move",
        "W/S: Up/Down",
        "A/D: Rotate",
        "Q: Land",
        "E: Take Off"
    ]
    for i, text in enumerate(controls):
        control_text = small_font.render(text, True, DARK_BLUE)
        screen.blit(control_text, (980, 200 + i * 30))

    # Draw the map
    draw_map(screen, drone_x, drone_y)

    # Detected objects
    objects_text = font.render("Detected Objects:", True, BLACK)
    screen.blit(objects_text, (980, 690))
    for i, obj in enumerate(detected_objects[:2]):  # Display up to 2 objects
        obj_text = small_font.render(obj, True, DARK_BLUE)
        screen.blit(obj_text, (990, 720 + i * 30))

    pygame.display.flip()
    clock.tick(30)  # Limit the frame rate to 30 FPS

# Clean up
tello.land()
tello.streamoff()
pygame.quit()