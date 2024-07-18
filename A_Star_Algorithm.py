import heapq
import math
from matplotlib.animation import FuncAnimation
from matplotlib.animation import PillowWriter
import matplotlib.pyplot as plt
import numpy as np

class Node:
    def __init__(self, state=None, parent=None, cost=0, heuristic=0, step_cost=1, last_move=None, last_direction=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.cost = cost
        self.heuristic = heuristic
        self.step_cost = step_cost
        self.last_move = last_move
        self.last_direction = last_direction

    def addChildren(self, children):
        self.children.extend(children)

    def total_cost(self):
        return self.step_cost + self.heuristic

    def __lt__(self, other):
        return self.total_cost() < other.total_cost()

# Constants to represent different types of cells
EMPTY = "Empty"
TRAP1 = "Trap 1"
TRAP2 = "Trap 2"
TRAP3 = "Trap 3"
TRAP4 = "Trap 4"
REWARD1 = "Reward 1"
REWARD2 = "Reward 2"
TREASURE = "Treasure"
OBSTACLE = "Obstacle"

# Define the hexagonal grid as a dictionary to include neighbors
state_space = {
    (1, 6): EMPTY, (2, 6): EMPTY, (3, 6): EMPTY, (4, 6): EMPTY, (5, 6): REWARD1, (6, 6): EMPTY, (7, 6): EMPTY, (8, 6): EMPTY, (9, 6): EMPTY, (10, 6): EMPTY,
    (1, 5): EMPTY, (2, 5): TRAP2, (3, 5): EMPTY, (4, 5): TRAP4, (5, 5): TREASURE, (6, 5): EMPTY, (7, 5): TRAP3, (8, 5): EMPTY, (9, 5): OBSTACLE, (10, 5): EMPTY,
    (1, 4): EMPTY, (2, 4): EMPTY, (3, 4): OBSTACLE, (4, 4): EMPTY, (5, 4): OBSTACLE, (6, 4): EMPTY, (7, 4): EMPTY, (8, 4): REWARD2, (9, 4): TRAP1, (10, 4): EMPTY,
    (1, 3): OBSTACLE, (2, 3): REWARD1, (3, 3): EMPTY, (4, 3): OBSTACLE, (5, 3): EMPTY, (6, 3): TRAP3, (7, 3): OBSTACLE, (8, 3): TREASURE, (9, 3): EMPTY, (10, 3): TREASURE,
    (1, 2): EMPTY, (2, 2): EMPTY, (3, 2): TRAP2, (4, 2): TREASURE, (5, 2): OBSTACLE, (6, 2): EMPTY, (7, 2): OBSTACLE, (8, 2): OBSTACLE, (9, 2): EMPTY, (10, 2): EMPTY,
    (1, 1): EMPTY, (2, 1): EMPTY, (3, 1): EMPTY, (4, 1): EMPTY, (5, 1): EMPTY, (6, 1): REWARD2, (7, 1): EMPTY, (8, 1): EMPTY, (9, 1): EMPTY, (10, 1): EMPTY
}

# Define directions for movement in a hexagonal grid
DIRECTIONS_EVEN = {
    "N": (0, 1),
    "NE": (1, 1),
    "SE": (1, 0),
    "S": (0, -1),
    "SW": (-1, 0),
    "NW": (-1, 1)
}

DIRECTIONS_ODD = {
    "N": (0, 1),
    "NE": (1, 0),
    "SE": (1, -1),
    "S": (0, -1),
    "SW": (-1, -1),
    "NW": (-1, 0)
}

def get_neighbors(node):
    (x, y) = node
    directions = DIRECTIONS_EVEN if x % 2 == 0 else DIRECTIONS_ODD
    possible_neighbors = [(x + dx, y + dy) for (dx, dy) in directions.values()]
    return [neighbor for neighbor in possible_neighbors if neighbor in state_space]

def get_direction(dx, dy, x):
    directions = DIRECTIONS_EVEN if x % 2 == 0 else DIRECTIONS_ODD
    for direction, (dir_dx, dir_dy) in directions.items():
        if (dx, dy) == (dir_dx, dir_dy):
            return direction
    return None

# Function defining heuristic with Euclidean distance
def heuristic(node, goal):
    return math.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

def trap_cost(node):
    node.step_cost *= 2
    return node.step_cost

def reward_cost(node):
    node.step_cost /= 2
    return node.step_cost

# Function for Trap 3
def move_two_cells(node, state_space):
    if node.last_direction:
        new_position = node.state
        for _ in range(1):  # move two cells
            directions = DIRECTIONS_EVEN if new_position[0] % 2 == 0 else DIRECTIONS_ODD
            dx, dy = directions[node.last_direction]
            new_position = (new_position[0] + dx, new_position[1] + dy)
            if new_position not in state_space or state_space[new_position] == OBSTACLE:
                return
        print(" ---------------------------------------------------------------------------------")
        print(f"|{node.parent} is Trap 3: It move us two cells away following the last movement direction|")
        print(" ---------------------------------------------------------------------------------")
        print(f"Node last direction: {node.last_direction}")
        print(f"-> The cell move from node {node.parent} to {new_position}!!!!\n")
        node.state = new_position


def handle_traps_and_rewards(node, state_space):
    cell_type = state_space[node.parent]
    if cell_type == TRAP1:
        node.step_cost = trap_cost(node)
        print("\n -----------------------------------------------------------------")
        print(f"| {node.parent} is Trap 1: The step cost to {node.state} will be multiply by 2 |")
        print(" -----------------------------------------------------------------")
    elif cell_type == TRAP2:
        node.step_cost = trap_cost(node)
        print("\n -----------------------------------------------------------------")
        print(f"| {node.parent} is Trap 2: The step cost to {node.state} will be multiply by 2 |")
        print(" -----------------------------------------------------------------")
    elif cell_type == REWARD1:
        node.step_cost = reward_cost(node)
        print("\n ------------------------------------------------------------")
        print(f"| {node.parent} is Reward 1: The step cost to {node.state} be divide by 2 |")
        print(" -----------------------------------------------------------")
    elif cell_type == REWARD2:
        node.step_cost = reward_cost(node)
        print("\n ------------------------------------------------------------")
        print(f"| {node.parent} is Reward 2: The step cost to {node.state} be divide by 2 |")
        print(" ------------------------------------------------------------")
    elif cell_type == TRAP3:
        move_two_cells(node,state_space)
    elif cell_type == TRAP4:
        # Game over: logic to avoid trap 4
        print("\n ------------------------------------------------------------")
        print(f"| {node.parent} is Trap 4: All uncollected treasures will be gone!! |")
        print(" ------------------------------------------------------------")
        return False
    return True

def expandAndReturnChildren(state_space, node, goal_state):
  children = []
  for neighbor in get_neighbors(node.state):
    if state_space[neighbor] != OBSTACLE:
        h = heuristic(neighbor, goal_state)
        step_cost = 1  # inherit the step cost
        child_cost = node.cost + step_cost
      # Check if the current node is Trap 3
        if state_space[node.state] == TRAP3:
            # If Trap 3, calculate new position based on last_direction
            new_position = node.state
            if node.last_direction:
                directions = DIRECTIONS_EVEN if new_position[0] % 2 == 0 else DIRECTIONS_ODD
                dx, dy = directions[node.last_direction]
                new_position = (new_position[0] + dx, new_position[1] + dy)

                # Check if new position is valid and not an obstacle
                if new_position not in state_space or state_space[new_position] == OBSTACLE:
                    continue

                child_node = Node(new_position, node.state, child_cost, h, step_cost, None, node.last_direction)  # Don't update last_move here
        else:
            # For other nodes, determine last_move and last_direction based on neighbor
            last_move = (neighbor[0] - node.state[0], neighbor[1] - node.state[1])
            last_direction = get_direction(last_move[0], last_move[1], node.state[0])
            child_node = Node(neighbor, node.state, child_cost, h, step_cost, last_move, last_direction)

        if handle_traps_and_rewards(child_node, state_space):
            children.append(child_node)
  return children


def a_star(state_space, initial_state, goal_state):
    frontier = []
    explored = []
    found_goal = False
    goalie = Node()
    solution = []
    
    # Add initial state to frontier
    initial_node = Node(initial_state, None, 0, heuristic(initial_state, goal_state))
    heapq.heappush(frontier, initial_node)
    
    while not found_goal and frontier:
        # Goal test at expansion
        current_node = heapq.heappop(frontier)
        if  current_node.state==goal_state:
            found_goal = True
            goalie = current_node
            print(f"!!!Goal found at {current_node.state}!!!")
            print("--------------------------------------------------------------------------------------")
            

        # Expand the current node
        children = expandAndReturnChildren(state_space, current_node, goal_state)
        print(f"-> Expanding node {current_node.state} \nNeighboring node: {[child.state for child in children]}")
        # Add children list to the expanded node
        current_node.addChildren(children)
        # Add to the explored list
        explored.append(current_node)
        # Add children to the frontier
        for child in children:
            # print(f"Heuristic from {child.state} to goal: {child.heuristic}")
            # print(f"Path cost from {current_node.state} to {child.state}: {child.step_cost}")
            print(f"Total cost from {current_node.state} to {child.state} : {child.total_cost()}")

            if not any(child.state == e.state for e in explored):
                heapq.heappush(frontier, child)
        print("--------------------------------------------------------------------------------------")
    
    solution = [goalie.state]
    path_cost = goalie.cost
    while goalie.parent is not None:
        solution.insert(0, goalie.parent)
        for e in explored:
            if e.state == goalie.parent:
                goalie = e
                break
        
    return solution, path_cost


def find_nearest_treasure(state_space, start, treasures):
    nearest_treasure = None
    min_distance = float('inf')
    for treasure in treasures:
        distance = heuristic(start, treasure)
        if distance < min_distance:
            min_distance = distance
            nearest_treasure = treasure
    return nearest_treasure



#====================#
# Codes for plotting #
#====================#

# Function to calculate the points of a hexagon given its center, rotated correctly
def hexagon_points(x, y, size=1):
    angle_offset = np.pi / 1
    points = [
        (x + size * np.cos(angle_offset + np.pi / 3 * i), 
         y + size * np.sin(angle_offset + np.pi / 3 * i)) 
        for i in range(6)
    ]
    return points

# Function to plot the state space
def plot_state_space(state_space, initial_state, treasures):
    fig, ax = plt.subplots()
    colors = {
        EMPTY: 'white',
        TRAP1: 'purple',
        TRAP2: 'violet',
        TRAP3: 'magenta',
        TRAP4: 'darkviolet',
        REWARD1: 'cyan',
        REWARD2: 'lightseagreen',
        TREASURE: 'gold',
        OBSTACLE: 'gray'
    }
    
    size = 1
    for (x, y), cell_type in state_space.items():
        hx = (x - 1) * 1.5 * size  # Align with x-axis
        hy = (y - 1) * np.sqrt(3) * size - (np.sqrt(3) / 2 * size if x % 2 else 0)  # Align with y-axis, rearrange rows
        hexagon = plt.Polygon(hexagon_points(hx, hy, size), color=colors[cell_type], ec='black')
        ax.add_patch(hexagon)
        ax.text(hx, hy, f'({x},{y})', ha='center', va='center', fontsize=8)

    for (x, y) in treasures:
        hx = (x - 1) * 1.5 * size
        hy = (y - 1) * np.sqrt(3) * size - (np.sqrt(3) / 2 * size if x % 2 else 0)
        hexagon = plt.Polygon(hexagon_points(hx, hy, size), color=colors[TREASURE], ec='black')
        ax.add_patch(hexagon)
    
    hx = (initial_state[0] - 1) * 1.5 * size
    hy = (initial_state[1] - 1) * np.sqrt(3) * size - (np.sqrt(3) / 2 * size if initial_state[0] % 2 else 0)
    hexagon = plt.Polygon(hexagon_points(hx, hy, size), color='blue', ec='black')
    ax.add_patch(hexagon)

    # Create legend
    handles = [plt.Line2D([0], [0], marker='o', color='black', markerfacecolor=color, markersize=10, label=label) 
               for label, color in colors.items()]
    handles.append(plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='blue', markersize=10, label='Initial State'))
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.15, 1))

    ax.autoscale_view()
    ax.set_aspect('equal')
    plt.axis('off')  # Turn off the axis
    plt.title('State Space')
    plt.show()

# Function to animate the path taken
def animate_path(state_space, path, treasures):
    fig, ax = plt.subplots()
    colors = {
        EMPTY: 'white',
        TRAP1: 'purple',
        TRAP2: 'violet',
        TRAP3: 'magenta',
        TRAP4: 'darkviolet',
        REWARD1: 'cyan',
        REWARD2: 'lightseagreen',
        TREASURE: 'gold',
        OBSTACLE: 'gray'
    }
    
    size = 1
    for (x, y), cell_type in state_space.items():
        hx = (x - 1) * 1.5 * size  # Align with x-axis
        hy = (y - 1) * np.sqrt(3) * size - (np.sqrt(3) / 2 * size if x % 2 else 0)  # Align with y-axis, rearrange rows
        hexagon = plt.Polygon(hexagon_points(hx, hy, size), color=colors[cell_type], ec='black')
        ax.add_patch(hexagon)
        ax.text(hx, hy, f'({x},{y})', ha='center', va='center', fontsize=8)
    
    collected_treasures = []
    text_prompt = ax.text(0.5, 0.001, '', transform=ax.transAxes, ha='center', fontsize=12, color='red')

    handles = [plt.Line2D([0], [0], marker='o', color='black', markerfacecolor=color, markersize=10, label=label) 
               for label, color in colors.items()]
    handles.append(plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='blue', markersize=10, label='Initial State'))
    ax.legend(handles=handles, loc='upper right', bbox_to_anchor=(1.15, 1))

    def update(num):
        patches = []
        if num < len(path):
            hx = (path[num][0] - 1) * 1.5 * size
            hy = (path[num][1] - 1) * np.sqrt(3) * size - (np.sqrt(3) / 2 * size if path[num][0] % 2 else 0)
            hexagon = plt.Polygon(hexagon_points(hx, hy, size), color='blue', ec='black')
            patches.append(hexagon)
            ax.add_patch(hexagon)

            # Check if a treasure is collected
            if path[num] in treasures and path[num] not in collected_treasures:
                collected_treasures.append(path[num])
                text_prompt.set_text(f'Treasure collected at {path[num]}')

        if num == len(path) - 1 and len(collected_treasures) == len(treasures):
            text_prompt.set_text('All treasures collected')
        return patches + [text_prompt]

    ani = FuncAnimation(fig, update, frames=len(path), repeat=False, interval=500, blit=True)

    ax.autoscale_view()
    ax.set_aspect('equal')
    plt.axis('off')  # Turn off the axis
    plt.title('Path Taken')
    ani.save("path_animation.gif", writer=PillowWriter(fps=2))
    plt.show()


# Main execution
if __name__ == "__main__":
    initial_state = (1, 6)
    treasures = [(5, 5), (8, 3), (10, 3), (4, 2)]

    total_solution = []
    total_cost = 0
    current_state = initial_state

    # loop while there are treasures in the list
    while treasures:
        nearest_treasure = find_nearest_treasure(state_space, current_state, treasures) # Find the closest treasure to the current state
        solution, cost = a_star(state_space, current_state, nearest_treasure) # Find the path to the closest treasure using a_star
        total_solution.extend(solution[:-1])  # add all but the last to avoid duplication
        total_cost += cost
        current_state = nearest_treasure
        treasures.remove(nearest_treasure)

    total_solution.append(current_state)  # add the final goal state
    print("Solution:", total_solution)
    print("Total Path Cost:", total_cost)

    # Plotting the state space of the virtual world map out
    plot_state_space(state_space, initial_state, [(5, 5), (8, 3), (10, 3), (4, 2)])
    # Close the window to show the animate solution path
    animate_path(state_space, total_solution, [(5, 5), (8, 3), (10, 3), (4, 2)])
    

