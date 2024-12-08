import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.cluster import DBSCAN
from sklearn.linear_model import LinearRegression
from matplotlib.lines import Line2D

df = pd.read_csv("log_2020_04_11_06_19_44_805891.csv")

# Initialize the position and lists to hold the path coordinates
x, y = 0, 0
positions = [(x, y)]
wall_positions_f = []
wall_positions_l = []
wall_positions_r = []

# Convert angles from degrees to radians for trigonometric functions
df['angle_rad'] = np.deg2rad(df[' angle'])

# Calculate the robot's path and detect walls
for i in range(1, len(df)):
    # Calculate the change in distance
    delta_dist = df.loc[i, ' dist'] - df.loc[i - 1, ' dist']

    # Calculate the new position
    angle = df.loc[i, 'angle_rad']
    x += delta_dist * np.cos(angle)
    y += delta_dist * np.sin(angle)

    # Append the new position to the list
    positions.append((x, y))

    # Detect walls using ultrasound sensor data
    f_uh = df.loc[i, ' f_uh']
    l_uh = df.loc[i, ' l_uh']
    r_uh = df.loc[i, ' r_uh']

    # Calculate the wall positions relative to the robot's current position and angle
    front_wall_x = x + f_uh * np.cos(angle)
    front_wall_y = y + f_uh * np.sin(angle)
    left_wall_x = x + l_uh * np.cos(angle + np.pi / 2)
    left_wall_y = y + l_uh * np.sin(angle + np.pi / 2)
    right_wall_x = x + r_uh * np.cos(angle - np.pi / 2)
    right_wall_y = y + r_uh * np.sin(angle - np.pi / 2)

    # Append the wall positions to the respective lists
    wall_positions_f.append((front_wall_x, front_wall_y))
    wall_positions_l.append((left_wall_x, left_wall_y))
    wall_positions_r.append((right_wall_x, right_wall_y))

# Convert the positions list to a numpy array for easier plotting
positions = np.array(positions)
wall_positions_f = np.array(wall_positions_f)
wall_positions_l = np.array(wall_positions_l)
wall_positions_r = np.array(wall_positions_r)

# Prepare the figure and axis for animation
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(-1000, 2000)
ax.set_ylim(-1000, 2000)
path_line, = ax.plot([], [], 'bo-', label='Robot Path')
wall_points_f = ax.scatter([], [], s=10, c='r', label='Measured Front UH Points')
wall_points_l = ax.scatter([], [], s=10, c='g', label='Measured Left UH Points')
wall_points_r = ax.scatter([], [], s=10, c='orange', label='Measured Right UH Points')

# Define wall_lines variable
wall_lines = Line2D([], [], color='b', linestyle='-', label='Wall Lines')
ax.add_line(wall_lines)


# Initialize the plot
def init():
    path_line.set_data([], [])
    wall_points_f.set_offsets(np.array([[], []]).T)
    wall_points_l.set_offsets(np.array([[], []]).T)
    wall_points_r.set_offsets(np.array([[], []]).T)
    wall_lines.set_data([], [])
    return path_line, wall_points_f, wall_points_l, wall_points_r, wall_lines


# Update the plot for each frame
def update(frame):
    # Update the path line
    path_line.set_data(positions[:frame + 1, 0], positions[:frame + 1, 1])

    # Update the wall points for each ultrasound sensor
    if frame > 0:
        wall_points_f.set_offsets(wall_positions_f[:frame, :])
        wall_points_l.set_offsets(wall_positions_l[:frame, :])
        wall_points_r.set_offsets(wall_positions_r[:frame, :])

    return path_line, wall_points_f, wall_points_l, wall_points_r, wall_lines


# Create the animation
ani = FuncAnimation(fig, update, frames=len(positions), init_func=init, blit=True, repeat=False)


# Finalize function to cluster wall points and fit lines
def finalize():
    global wall_positions_f, wall_positions_l, wall_positions_r
    wall_positions = np.concatenate([wall_positions_f, wall_positions_l, wall_positions_r])
    db = DBSCAN(eps=100, min_samples=2).fit(wall_positions)
    labels = db.labels_
    unique_labels = set(labels)
    all_lines_x = []
    all_lines_y = []
    for label in unique_labels:
        if label == -1:
            continue
        class_member_mask = (labels == label)
        xy = wall_positions[class_member_mask]
        if len(xy) > 1:
            X = xy[:, 0].reshape(-1, 1)
            y = xy[:, 1]
            lr = LinearRegression().fit(X, y)
            line_x = np.linspace(X.min(), X.max(), 100)
            line_y = lr.predict(line_x.reshape(-1, 1))
            all_lines_x.extend(line_x)
            all_lines_y.extend(line_y)
    wall_lines.set_data(all_lines_x, all_lines_y)


# Execute the finalize function after the animation is finished
# ani.event_source.add_callback(finalize)

# Show the plot
plt.title('Progressive Robot Path and Detected Walls')
plt.xlabel('X Position (mm)')
plt.ylabel('Y Position (mm)')
plt.legend()
plt.grid(True)
plt.show()