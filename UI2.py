import pygame
import math
import cv2
from djitellopy import tello
from ultralytics import YOLO

# Initialize Pygame
pygame.init()

# Constants for the display
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
DRONE_SIZE = 10
ARROW_SIZE = 20
THRESHOLD_DISTANCE = 30  # Minimum distance to consider a new dot

# Set up Pygame window
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Drone Moving Map")

# Load YOLO model
model = YOLO("yolo11n.pt")

# Initialize DJI Tello
tello = tello.Tello()
tello.connect()
tello.streamon()

# Starting position of the drone (center of the screen)
drone_position = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]
initial_position = drone_position.copy()
drone_direction = 0  # Angle in degrees, 0 is facing right
path = [drone_position.copy()]  # List to store the path

# List to store detected persons' positions (relative to the map)
person_positions = []

# Speed and turn rate
move_speed = 30  # Adjust speed for testing
turn_speed = 15  # Degrees per frame when turning

# Function to calculate new position based on movement
def move_drone(position, direction, speed):
    # Calculate new position using basic trigonometry (movement in direction)
    new_x = position[0] + math.cos(math.radians(direction)) * speed
    new_y = position[1] + math.sin(math.radians(direction)) * speed
    return [new_x, new_y]

# Function to draw the arrow representing the direction of the drone
def draw_arrow(surface, position, direction, size):
    # Calculate the end point of the arrow based on drone's direction
    end_x = position[0] + math.cos(math.radians(direction)) * size
    end_y = position[1] + math.sin(math.radians(direction)) * size
    pygame.draw.line(surface, BLUE, position, (end_x, end_y), 5)  # Draw the direction line
    pygame.draw.circle(surface, BLUE, (end_x, end_y), 5)  # Draw the tip of the arrow

# Function to map the bounding box (x1, y1, x2, y2) to a position on the 2D map
def map_person_position(x1, y1, x2, y2, frame_width, frame_height, zoom_factor):
    # Mapping person bounding box center to the map with zoom factor
    person_x = int((x1 + x2) / 2 / frame_width * SCREEN_WIDTH * zoom_factor)
    person_y = int((y1 + y2) / 2 / frame_height * SCREEN_HEIGHT * zoom_factor)
    return person_x, person_y

# Function to check if a new position is too close to any existing positions
def is_too_close(new_pos, existing_positions, threshold):
    for pos in existing_positions:
        # Calculate Euclidean distance between the new position and existing position
        distance = math.sqrt((new_pos[0] - pos[0]) ** 2 + (new_pos[1] - pos[1]) ** 2)
        if distance < threshold:
            return True  # The new position is too close
    return False  # The new position is far enough

# Function to adjust zoom based on distance from the initial position
def calculate_zoom_factor(drone_position, initial_position):
    # Calculate distance from the initial position
    distance = math.sqrt((drone_position[0] - initial_position[0]) ** 2 + (drone_position[1] - initial_position[1]) ** 2)
    # Use distance to adjust zoom (e.g., the further the drone moves, the more zoomed-out the map is)
    zoom_factor = max(1.0, min(2.5, 1 + distance / 200))  # Adjust zoom range
    return zoom_factor

# Smooth movement of the drone
def smooth_move(current_pos, target_pos, speed):
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    distance = math.sqrt(dx**2 + dy**2)
    if distance < speed:
        return target_pos  # Reached the target
    else:
        ratio = speed / distance
        new_x = current_pos[0] + dx * ratio
        new_y = current_pos[1] + dy * ratio
        return [new_x, new_y]

# Main loop
running = True
while running:
    # Get the current frame from the drone
    frame_read = tello.get_frame_read()
    frame = cv2.cvtColor(frame_read.frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (960, 720))

    # Perform object detection
    results = model.predict(frame, show=True, conf=0.5)

    # Iterate over the detections
    for result in results:
        for box in result.boxes:
            # YOLO box format: (x1, y1, x2, y2) and class id
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cls = int(box.cls)

            # Check if the detected object is a "person" (typically class id = 0)
            if cls == 0:
                # Map the bounding box to a position on the map with zoom factor
                zoom_factor = calculate_zoom_factor(drone_position, initial_position)
                person_position = map_person_position(x1, y1, x2, y2, frame.shape[1], frame.shape[0], zoom_factor)
                
                # Check if the new position is too close to any existing position
                if not is_too_close(person_position, person_positions, THRESHOLD_DISTANCE):
                    person_positions.append(person_position)

    # Handle Pygame events (such as quitting the app or moving the drone)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:  # Move forward
                tello.move_forward(move_speed)  # Correct method to move forward
                target_position = move_drone(drone_position, drone_direction, move_speed)
                drone_position = smooth_move(drone_position, target_position, move_speed)
                path.append(drone_position.copy())
            elif event.key == pygame.K_LEFT:  # Turn left
                drone_direction += turn_speed
                tello.rotate_counter_clockwise(turn_speed)  # Rotate drone counter-clockwise
            elif event.key == pygame.K_RIGHT:  # Turn right
                drone_direction -= turn_speed
                tello.rotate_clockwise(turn_speed)  # Rotate drone clockwise
            elif event.key == pygame.K_DOWN:  # Move backward
                tello.move_backward(move_speed)  # Correct method to move backward
                target_position = move_drone(drone_position, drone_direction, -move_speed)
                drone_position = smooth_move(drone_position, target_position, move_speed)
                path.append(drone_position.copy())

    # Calculate zoom factor based on drone's movement
    zoom_factor = calculate_zoom_factor(drone_position, initial_position)

    # Draw the UI
    screen.fill(WHITE)  # Clear the screen with white

    # Draw the path (drone movement history)
    if len(path) > 1:
        for i in range(1, len(path)):
            pygame.draw.line(screen, GREEN, path[i-1], path[i], 2)

    # Draw the drone's position (represented by a blue circle)
    pygame.draw.circle(screen, BLUE, (int(drone_position[0]), int(drone_position[1])), DRONE_SIZE)

    # Draw the arrow showing the drone's direction
    draw_arrow(screen, drone_position, drone_direction, ARROW_SIZE)

    # Draw all the detected persons on the map as red circles (fixed in place)
    for pos in person_positions:
        pygame.draw.circle(screen, RED, pos, 5)

    # Update the display
    pygame.display.flip()

    # Control the frame rate
    pygame.time.Clock().tick(30)

# Quit Pygame
tello.streamoff()
pygame.quit()
