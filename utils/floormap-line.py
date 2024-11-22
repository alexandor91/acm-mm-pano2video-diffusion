import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from sklearn.cluster import DBSCAN

# # Normalize the depth map (scale values between 0 and 1)
# depth_map_normalized = depth_map.astype(np.float32) / 255.0  # Assuming 8-bit depth map image

# # Extract the horizontal depth line (e.g., the middle row of the depth map)
# # This assumes the horizon is located in the middle of the image
# middle_row = depth_map_normalized.shape[0] // 3
# depth_values = depth_map_normalized[middle_row, :]  # Extract middle row depth values

# # Parameters for depth-to-floor map conversion
# image_width = depth_map_normalized.shape[1]
# fov = 360  # field of view is 360 degrees for panorama
# angles = np.linspace(0, 2 * np.pi, image_width)  # Angle for each pixel

# # Convert depth and angles (polar coordinates) to Cartesian coordinates
# x = depth_values * np.cos(angles)  # x-coordinate
# y = depth_values * np.sin(angles)  # y-coordinate

# # Normalize the depth values for better visualization
# x = x - np.min(x)
# y = y - np.min(y)

# # Create the floor map boundary from the converted Cartesian coordinates
# plt.figure(figsize=(8, 8))
# plt.plot(x, y, color='blue')  # Plot the boundary

# plt.title('Floor Map Boundary from Panorama Depth')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.axis('equal')  # Ensure the aspect ratio is equal for accurate representation
# plt.show()

# plt.savefig('floormap.png', dpi=300, bbox_inches='tight', pad_inches=0)
# Load the depth map image (grayscale)
# Normalize the depth map (scale values between 0 and 1)

base_dir = "/home/student./anonymous/"
folder = "PanaromaSamples"
filename = "c8112f69d34d476fbb29e6b3909deba2.jpg"

folder = "Depth-Anything-V2"
filename = "magma_depth_0e92a69a50414253a23043758f111cec.png"
input_image_path = os.path.join(base_dir, folder, filename)    

# Load the panorama depth image
depth_map = cv2.imread(input_image_path, cv2.IMREAD_GRAYSCALE)
depth_map = cv2.flip(depth_map, 1)

# Normalize and invert depth values
depth_map_normalized = depth_map.astype(np.float32) / 255.0
depth_map_normalized = 1.0 - depth_map_normalized

# Extract the middle row depth values (horizontal line)
middle_row = depth_map_normalized.shape[0] // 3 * 2
depth_values = depth_map_normalized[middle_row, :]

# Define image width and number of quadrants
image_width = depth_values.shape[0]
quarter_length = image_width // 4

# Ensure depth values can be split evenly
remaining_pixels = image_width % 4
if remaining_pixels != 0:
    quarter_length = image_width // 4
else:
    quarter_length = image_width // 4

radius = 1.0

# Create angles for four quadrants, mapping them as straight lines
angles_1 = np.linspace(0, np.pi/2, quarter_length)
angles_2 = np.linspace(np.pi/2, np.pi, quarter_length)
angles_3 = np.linspace(np.pi, 3*np.pi/2, quarter_length)
angles_4 = np.linspace(3*np.pi/2, 2*np.pi, quarter_length + remaining_pixels)

# Extract depth values for each quadrant
depth_q1 = radius * depth_values[:quarter_length]
depth_q2 = radius * depth_values[quarter_length:2*quarter_length]
depth_q3 = radius * depth_values[2*quarter_length:3*quarter_length]
depth_q4 = radius * depth_values[3*quarter_length:]

# Convert depth values to Cartesian coordinates for straight-line mapping
x_q1 = depth_q1 * np.cos(angles_1)
y_q1 = depth_q1 * np.sin(angles_1)

x_q2 = depth_q2 * np.cos(angles_2)
y_q2 = depth_q2 * np.sin(angles_2)

x_q3 = depth_q3 * np.cos(angles_3)
y_q3 = depth_q3 * np.sin(angles_3)

x_q4 = depth_q4 * np.cos(angles_4)
y_q4 = depth_q4 * np.sin(angles_4)

# Combine Cartesian coordinates
x_total = np.concatenate((x_q1, x_q2, x_q3, x_q4))
y_total = np.concatenate((y_q1, y_q2, y_q3, y_q4))

# Normalize x and y for visualization
x_total = (x_total - np.min(x_total)) / (np.max(x_total) - np.min(x_total))
y_total = (y_total - np.min(y_total)) / (np.max(y_total) - np.min(y_total))

def fit_bounding_box(x, y):
    points = np.column_stack((x, y))
    db = DBSCAN(eps=0.05, min_samples=5).fit(points)
    labels = db.labels_
    
    core_samples_mask = np.zeros_like(labels, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    
    unique_labels = set(labels)
    
    max_area = 0
    best_bbox = None
    
    for k in unique_labels:
        if k == -1:
            continue
        class_member_mask = (labels == k)
        xy = points[class_member_mask & core_samples_mask]
        
        if len(xy) < 4:
            continue
        
        x_min, y_min = xy.min(axis=0)
        x_max, y_max = xy.max(axis=0)
        width = x_max - x_min
        height = y_max - y_min
        area = width * height
        
        if area > max_area:
            max_area = area
            best_bbox = (x_min, y_min, x_max, y_max)
    
    return best_bbox

def generate_waypoints(start, end, num_points, center, depths, safety_margin=0.1):
    direction = np.array(end) - np.array(start)
    unit_direction = direction / np.linalg.norm(direction)
    
    max_distance = np.linalg.norm(direction)
    
    waypoints = []
    for i in range(num_points):
        t = i / (num_points - 1)
        point = np.array(start) + t * direction
        
        # Calculate the angle of the waypoint relative to the center
        angle = np.arctan2(point[1] - center[1], point[0] - center[0])
        if angle < 0:
            angle += 2 * np.pi  # Ensure angle is between 0 and 2π
        
        # Find the index in the depth array that corresponds to this angle
        index = int(angle / (2 * np.pi) * len(depths))
        
        # Check if the waypoint is within the valid depth minus the safety margin
        distance_to_center = np.linalg.norm(point - center)
        if distance_to_center <= depths[index] * (1 - safety_margin):
            waypoints.append(tuple(point))
        else:
            # Add the last safe point
            safe_distance = depths[index] * (1 - safety_margin)
            safe_point = center + safe_distance * unit_direction
            waypoints.append(tuple(safe_point))
            break  # Stop generating waypoints if we've reached the safety margin
    
    return waypoints

def generate_curved_waypoints(start, end, num_points, center, depths, curvature=0.5, convex=True, safety_margin=0.1):
    direction = (np.array(end) - np.array(start))
    max_distance = np.linalg.norm(direction)
    unit_direction = direction / max_distance
    
    # Vector perpendicular to the direction
    perpendicular = np.array([-unit_direction[1], unit_direction[0]])
    
    waypoints = []
    for i in range(num_points):
        t = i / (num_points - 1)
        
        # Linear interpolation
        linear_point = np.array(start) + t * direction
        
        # Calculate the offset for the curve
        if convex:
            offset = curvature * np.sin(np.pi * t)
        else:
            offset = curvature * (- np.sin(np.pi * t))
        
        # Apply the offset
        curved_point = linear_point + offset * max_distance * perpendicular
        
        # Calculate the angle of the waypoint relative to the center
        # print("$$$$$$$$$$$$ @@@@@@@  $$$")
        # print(center[0])
        # print(center[1])
        angle = np.arctan2(curved_point[1] - center[1], curved_point[0] - center[0])
        if angle < 0:
            angle += 2 * np.pi  # Ensure angle is between 0 and 2π
        
        # Find the index in the depth array that corresponds to this angle
        index = int(angle / (2 * np.pi) * len(depths))
        # Check if the waypoint is within the valid depth minus the safety margin
        distance_to_center = np.linalg.norm(curved_point - center)
        if distance_to_center <= depths[index] * (1 - safety_margin):
            waypoints.append(tuple(curved_point))
        else:
            # Find the intersection point with the safety margin
            direction_to_point = curved_point - center
            scale = (depths[index] * (1 - safety_margin)) / distance_to_center
            safe_point = center + scale * direction_to_point
            waypoints.append(tuple(safe_point))
            break  # Stop generating waypoints if we've reached the safety margin

    return waypoints

# Function to generate circular waypoints
def generate_circle_waypoints(radius, num_points, start_angle, end_angle, center):
    angles = np.linspace(start_angle, end_angle, num_points)
    waypoints = [(center[0] + radius * np.cos(angle), center[1] + radius * np.sin(angle)) for angle in angles]
    return waypoints

def check_rooms(x, y, bbox):
    x_min, y_min, x_max, y_max = bbox
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    center = (center_x, center_y)
    
    width = x_max - x_min
    height = y_max - y_min
    
    half_width = width / 2
    half_height = height / 2
    
    directions = [
        (0, "East"),
        (90, "North"),
        (180, "West"),
        (270, "South")
    ]
    
    rooms = []
    room_lines = []
    
    for angle, direction in directions:
        depths = []
        for i in range(-10, 11):  # Sample from -10 to +10 degrees
            sample_angle = np.radians(angle + i)
            sample_x = center_x + np.cos(sample_angle)
            sample_y = center_y + np.sin(sample_angle)
            
            distances = np.sqrt((x - sample_x)**2 + (y - sample_y)**2)
            depths.append(np.max(distances))
        
        avg_depth = np.mean(depths)
        max_depth = np.max(depths)
        threshold = max(half_width, half_height)
        
        if avg_depth > threshold:
            rooms.append(direction)
            
            # Generate line for the room using the maximum depth in that direction
            end_x = center_x + max_depth * np.cos(np.radians(angle))
            end_y = center_y + max_depth * np.sin(np.radians(angle))
            
            room_lines.append((center, (end_x, end_y), depths))
    
    return rooms, room_lines

# Fit bounding box
bbox = fit_bounding_box(x_total, y_total)

if bbox is None:
    print("No suitable bounding box found.")
else:
    # Check for rooms
    rooms, room_lines = check_rooms(x_total, y_total, bbox)

    # Plotting
    plt.figure(figsize=(10, 10))
    # plt.plot(x_total, y_total, color='blue', label='Room Layout')
    
    x_min, y_min, x_max, y_max = bbox
    # plt.plot([x_min, x_max, x_max, x_min, x_min],
    #          [y_min, y_min, y_max, y_max, y_min], 'r-', label='Bounding Box')
    bbx_half_width = (x_max - x_min) / 2.0
    bbx_half_height = (y_max - y_min) / 2.0

    center_x = 0.0
    center_y = 0.0
    # In the main plotting section:
    # Plot room lines and waypoints
    # Generate straight waypoints
    # In the main plotting section:
    i = 0
    for start, end, depths in room_lines:
        if  i == 3 or i ==0:
            safety_margin = 0.1  # 10% safety margin
            print(start[0])
            center_x = start[0]
            center_y = start[1]
            plt.plot([start[0], end[0]], [start[1], end[1]], 'k-', label='Room Line')
            
            print("####### start end #######")
            print(start)
            print(end)
            # Generate straight waypoints
            straight_waypoints = generate_waypoints(start, end, 15, start, depths, safety_margin)
            if straight_waypoints:
                waypoints_x, waypoints_y = zip(*straight_waypoints)
                plt.scatter(waypoints_x, waypoints_y, color='green', marker='o', label='Straight Waypoints')
            
            # Generate convex curved waypoints
            convex_waypoints = generate_curved_waypoints(start, end, 15, start, depths, curvature=0.15, convex=True, safety_margin=safety_margin)
            if convex_waypoints:
                waypoints_x, waypoints_y = zip(*convex_waypoints)
                plt.scatter(waypoints_x, waypoints_y, color='orange', marker='o', label='Convex Waypoints')
            
            # Generate concave curved waypoints
            concave_waypoints = generate_curved_waypoints(start, end, 15, start, depths, curvature=0.15, convex=False, safety_margin=safety_margin)
            if concave_waypoints:
                waypoints_x, waypoints_y = zip(*concave_waypoints)
                plt.scatter(waypoints_x, waypoints_y, color='red', marker='o', label='Concave Waypoints')
        i = i + 1
    center = (center_x, center_y)
    plt.title("Floor Map Boundary with Minimal Bounding Box and Room Detection")
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.axis('equal')
    # plt.legend()
    plt.grid(False)
    
    if rooms:
        print(f"Potential rooms found in directions: {', '.join(rooms)}")
    else:
        print("No potential rooms found in the sampled directions.")

    # Parameters
    radius = 1.0  # Radius of the circle
    num_points = 20  # Number of waypoints
    start_angle = np.pi/2.0  # Starting angle (0 radians)
    end_angle = np.pi * 3.0/2.0  # End angle (π/4 radians)
      # Center of the circle

    # Generate waypoints for π/4 trajectory
    waypoints = generate_circle_waypoints(radius, num_points, start_angle, end_angle, center)
    # Extract starting and ending points
    start_point = (center[0] + radius * np.cos(start_angle), center[1] + radius * np.sin(start_angle))
    end_point = (center[0] + radius * np.cos(end_angle), center[1] + radius * np.sin(end_angle))

    # Plotting the circle trajectory
    waypoints_np = np.array(waypoints)
    plt.plot(waypoints_np[:, 0], waypoints_np[:, 1], 'k-', label='π/4 Trajectory')  # Green circle trajectory

    # Mark waypoints
    plt.scatter(waypoints_np[:, 0], waypoints_np[:, 1], color='green')

    # Draw the radius lines in black from center to start and end points
    plt.plot([center[0], start_point[0]], [center[1], start_point[1]], 'k-', label='Start Radius')  # Black line for start radius
    plt.plot([center[0], end_point[0]], [center[1], end_point[1]], 'k-', label='End Radius')  # Black line for end radius

    # Draw the circle center for reference
    plt.scatter(center[0], center[1], color='red', marker='x', label='Center')

    # Adding labels and legend
    # plt.title('π/4 Circle Trajectory with Waypoints')

    plt.savefig("room_detection_with_waypoints.png", dpi=300, bbox_inches='tight', pad_inches=0)
    plt.show()