import tkinter as tk
import random
import math
import matplotlib # type: ignore
matplotlib.use("TkAgg")
from matplotlib.figure import Figure # type: ignore
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # type: ignore
from collections import defaultdict

# COMMON VARIABLES
SYS_FIELD_WIDTH = 2400
SYS_FIELD_HEIGHT = 2400
SYS_CELL_SIZE = 200         # The field is divided into cells to save processing power. This is the size of individual cells. Should be at least the vision lengths of carnivores. More cells (smaller cell size) is most efficient
SYS_START_PLANT_NUM = 1000
SYS_START_HERB_NUM = 80
SYS_START_CARN_NUM = 20
SYS_GRAPH_MEMORY = 100000   # Amount of ticks visible on the graph
SYS_SPEED_LEVELS = [0, 1, 2, 4, 8, 16, 32]

# NEURAL NETWORK VARIABLES
NN_MUTATION_RATE = 0.05     # Amount each weight is allowed to fluctuate per generation

# PLANT VARIABLES
PLANT_START_COLOR_R_0 = 0
PLANT_START_COLOR_R_1 = 0
PLANT_START_COLOR_G_0 = 192
PLANT_START_COLOR_G_1 = 255
PLANT_START_COLOR_B_0 = 0
PLANT_START_COLOR_B_1 = 0
PLANT_COLOR_MUTATE_RAND = 16            # Amount (in decimal) an organism's color components are allowed to fluctuate per generation
PLANT_SPREAD_MIN = 18                   # Shortest distance between any two plants
PLANT_SPREAD_MAX = 30                   # Largest distance a child plant can spawn
PLANT_SPREAD_TRY_NUM = 2                # Amount of spawning positions children plants attempt
PLANT_REPRODUCTION_FRAME_MIN = 200
PLANT_REPRODUCTION_FRAME_MAX = 1000
PLANT_REPRODUCTION_START_FRAME_MIN = 0
PLANT_REPRODUCTION_START_FRAME_MAX = 1000

# HERBIVORE VARIABLES
HERB_START_COLOR_R_0 = 0
HERB_START_COLOR_R_1 = 0
HERB_START_COLOR_G_0 = 0
HERB_START_COLOR_G_1 = 0
HERB_START_COLOR_B_0 = 192
HERB_START_COLOR_B_1 = 255
HERB_COLOR_MUTATE_RAND = 24             # Amount (in decimal) an organism's color components are allowed to fluctuate per generation
HERB_RADIUS_START = 16                  # Size at the beginning of the sim
HERB_NN_HIDDEN_SIZE = 6                 # Number of hidden neurons
HERB_VISION_CONE_LENGTH = 120           # Length of vision cones, in units
HERB_VISION_CONE_WIDTH = math.pi/5      # Width of vision cones, in radians
HERB_METABOLISM = 0.035                 # Energy units lost per tick
HERB_METABOLISM_SPEED_ADD_INV = 160     # Additional energy units lost per tick when moving (as a divisor)
HERB_METABOLISM_ROTATE_ADD_INV = 240    # Additional energy units lost per tick when rotating (as a divisor)
HERB_ROTATE_THRESHOLD = 0.3             # Minimum a neuron has to be set to to rotate an organism
HERB_SPEED_THRESHOLD = 0.1              # Minimum a neuron has to be set to cause an organism to move
HERB_ROTATE_MUL = 1.4                   # Degrees per frame an organism rotates, multiplied by their rotate neuron strength
HERB_SPEED_MUL = 2.8                    # Units per frame an organism moves when moving foward, multiplied by their speed neuron strength
HERB_SPEED_MUL_REV = 1.2                # Units per frame an organism moves when moving backward, multiplied by their speed neuron strength
HERB_ENERGY_GAIN = 20                   # Energy units to gain when eating a plant
HERB_REPRODUCTION_THRESHOLD = 300       # Energy units required to reproduce
HERB_REPRODUCTION_RETURN = 70           # Energy units to return to after reproduction (plus any excess)
HERB_BORN_ENERGY = 70                   # Energy units new organisms start with
HERB_ENERGY_START_MIN = 40              # Minimum energy units organisms start with on simulation start
HERB_ENERGY_START_MAX = 80              # Maximum energy units organisms start with on simulation start
HERB_LIFESPAN_MIN = 4000                # Minimum ticks to live for
HERB_LIFESPAN_MAX = 8000                # Maximum ticks to live for

# CARNIVORE VARIABLES
CARN_START_COLOR_R_0 = 192
CARN_START_COLOR_R_1 = 255
CARN_START_COLOR_G_0 = 0
CARN_START_COLOR_G_1 = 0
CARN_START_COLOR_B_0 = 0
CARN_START_COLOR_B_1 = 0
CARN_COLOR_MUTATE_RAND = 32             # Amount (in decimal) an organism's color components are allowed to fluctuate per generation
CARN_RADIUS_START = 20                  # Size of at the beginning of the sim
CARN_NN_HIDDEN_SIZE = 10                # Number of hidden neurons
CARN_VISION_CONE_LENGTH = 200           # Length of vision cones, in units
CARN_VISION_CONE_WIDTH = math.pi/9      # Width of vision cones, in radians
CARN_METABOLISM = 0.038                 # Energy units lost per tick
CARN_METABOLISM_SPEED_ADD_INV = 300     # Additional energy units lost per tick when moving (as a divisor)
CARN_METABOLISM_ROTATE_ADD_INV = 400    # Additional energy units lost per tick when rotating (as a divisor)
CARN_ROTATE_THRESHOLD = 0.1             # Minimum a neuron has to be set to to rotate an organism
CARN_SPEED_THRESHOLD = 0.2              # Minimum a neuron has to be set to cause an organism to move
CARN_ROTATE_MUL = 2.4                   # Degrees per frame an organism rotates, multiplied by their rotate neuron strength
CARN_SPEED_MUL = 3.8                    # Units per frame an organism moves when moving foward, multiplied by their speed neuron strength
CARN_SPEED_MUL_REV = 1.8                # Units per frame an organism moves when moving backward, multiplied by their speed neuron strength
CARN_ENERGY_GAIN = 35                   # Energy units to gain when eating a herbivore
CARN_REPRODUCTION_THRESHOLD = 300       # Energy units required to reproduce
CARN_REPRODUCTION_RETURN = 120          # Energy units to return to after reproduction (plus any excess)
CARN_BORN_ENERGY = 120                  # Energy units new organisms start with
CARN_ENERGY_START_MIN = 70              # Minimum energy units organisms start with on simulation start
CARN_ENERGY_START_MAX = 140             # Maximum energy units organisms start with on simulation start
CARN_LIFESPAN_MIN = 7000                # Minimum ticks to live for
CARN_LIFESPAN_MAX = 9000                # Maximum ticks to live for


# ----------------------
# Neural Network
# ----------------------
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, w1=None, b1=None, w2=None, b2=None):
        if w1 and b1 and w2 and b2:
            self.w1 = [[w + random.uniform(-NN_MUTATION_RATE, NN_MUTATION_RATE) for w in row] for row in w1]
            self.b1 = [b + random.uniform(-NN_MUTATION_RATE, NN_MUTATION_RATE) for b in b1]
            self.w2 = [[w + random.uniform(-NN_MUTATION_RATE, NN_MUTATION_RATE) for w in row] for row in w2]
            self.b2 = [b + random.uniform(-NN_MUTATION_RATE, NN_MUTATION_RATE) for b in b2]
        else:
            self.w1 = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
            self.b1 = [random.uniform(-1, 1) for _ in range(hidden_size)]
            self.w2 = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]
            self.b2 = [random.uniform(-1, 1) for _ in range(output_size)]

    def activate(self, x):
        return 1 / (1 + math.exp(-x))

    def forward(self, inputs):
        hidden = []
        for i in range(len(self.w1)):
            val = sum(w * inp for w, inp in zip(self.w1[i], inputs)) + self.b1[i]
            hidden.append(self.activate(val))
        outputs = []
        for i in range(len(self.w2)):
            val = sum(w * h for w, h in zip(self.w2[i], hidden)) + self.b2[i]
            outputs.append(self.activate(val) * 2 - 1)
        return outputs


# ----------------------
# Plant
# ----------------------
class Plant:
    _id_counter = 1

    def __init__(self, canvas, x, y, size=10):
        self.id = Plant._id_counter
        Plant._id_counter += 1
        self.x, self.y, self.size = x, y, size
        self.color = (
            random.randint(PLANT_START_COLOR_R_0, PLANT_START_COLOR_R_1), 
            random.randint(PLANT_START_COLOR_G_0, PLANT_START_COLOR_G_1), 
            random.randint(PLANT_START_COLOR_B_0, PLANT_START_COLOR_B_1), 
        )
        hex_color = f'#{self.color[0]:02x}{self.color[1]:02x}{self.color[2]:02x}'
        self.shape = canvas.create_rectangle(
            x - size//2, y - size//2, x + size//2, y + size//2,
            fill=hex_color, outline=""
        )
        self.duplication_timer = random.randint(PLANT_REPRODUCTION_START_FRAME_MIN, PLANT_REPRODUCTION_START_FRAME_MAX)

    def remove(self, canvas):
        canvas.delete(self.shape)

    def try_duplicate(self, canvas, plants, world_w, world_h):
        self.duplication_timer -= 1
        if self.duplication_timer <= 0:
            self.duplication_timer = random.randint(PLANT_REPRODUCTION_FRAME_MIN, PLANT_REPRODUCTION_FRAME_MAX)

            for _ in range(PLANT_SPREAD_TRY_NUM):
                new_x = (self.x + random.randint(-PLANT_SPREAD_MAX, PLANT_SPREAD_MAX)) % world_w
                new_y = (self.y + random.randint(-PLANT_SPREAD_MAX, PLANT_SPREAD_MAX)) % world_h

                too_close = False
                for p in plants:
                    dx = new_x - p.x
                    dy = new_y - p.y
                    if dx*dx + dy*dy < PLANT_SPREAD_MIN*PLANT_SPREAD_MIN:
                        too_close = True
                        break

                if not too_close:
                    r = min(max(self.color[0] + random.randint(-PLANT_COLOR_MUTATE_RAND, PLANT_COLOR_MUTATE_RAND), 0), 255)
                    g = min(max(self.color[1] + random.randint(-PLANT_COLOR_MUTATE_RAND, PLANT_COLOR_MUTATE_RAND), 0), 255)
                    b = min(max(self.color[2] + random.randint(-PLANT_COLOR_MUTATE_RAND, PLANT_COLOR_MUTATE_RAND), 0), 255)

                    child = Plant(canvas, new_x, new_y, self.size)
                    child.color = (r, g, b)
                    canvas.itemconfig(child.shape, fill=f'#{r:02x}{g:02x}{b:02x}')
                    plants.append(child)
                    return # Successfully reproduced


# ----------------------
# Herbivore
# ----------------------
class Herbivore:
    _id_counter = 1

    def __init__(self, canvas, x, y, radius=HERB_RADIUS_START, parent=None):
        self.id = Herbivore._id_counter
        Herbivore._id_counter += 1

        if parent:
            pr = int(parent.color[1:3], 16)
            pg = int(parent.color[3:5], 16)
            pb = int(parent.color[5:7], 16)
            pr = min(max(pr + random.randint(-HERB_COLOR_MUTATE_RAND, HERB_COLOR_MUTATE_RAND), 0), 255)
            pg = min(max(pg + random.randint(-HERB_COLOR_MUTATE_RAND, HERB_COLOR_MUTATE_RAND), 0), 255)
            pb = min(max(pb + random.randint(-HERB_COLOR_MUTATE_RAND, HERB_COLOR_MUTATE_RAND), 0), 255)
            self.color = f"#{pr:02x}{pg:02x}{pb:02x}"
        else:
            self.color = "#{:02x}{:02x}{:02x}".format(
                random.randint(HERB_START_COLOR_R_0, HERB_START_COLOR_R_1), 
                random.randint(HERB_START_COLOR_G_0, HERB_START_COLOR_G_1), 
                random.randint(HERB_START_COLOR_B_0, HERB_START_COLOR_B_1), 
            )

        self.radius = radius
        self.x, self.y = x, y
        self.rotation = random.uniform(0, 2*math.pi)
        self.speed = 0

        if parent:
            self.energy = random.uniform(HERB_ENERGY_START_MIN, HERB_ENERGY_START_MAX)
            self.generation = parent.generation + 1
        else:
            self.energy = HERB_BORN_ENERGY
            self.generation = 1
            
        self.alive = True

        self.age = 0
        self.lifespan = random.randint(HERB_LIFESPAN_MIN, HERB_LIFESPAN_MAX)

        input_size = 4 # R, G, B, Energy
        if parent:
            self.nn = NeuralNetwork(input_size, HERB_NN_HIDDEN_SIZE, 2,
                                    w1=parent.nn.w1, b1=parent.nn.b1,
                                    w2=parent.nn.w2, b2=parent.nn.b2)
        else:
            self.nn = NeuralNetwork(input_size, HERB_NN_HIDDEN_SIZE, 2)

        self.shape = canvas.create_oval(0,0,0,0, fill=self.color, outline="")
        self.facing_line = canvas.create_line(0,0,0,0, fill="black")
        self.vision_poly = canvas.create_polygon(0,0,0,0,0,0, fill="blue", stipple="gray25", outline="")

    def get_inputs(self, carnivores, plants, field_w, field_h, plant_grid=None, carn_grid=None):
        rgb = [-1, -1, -1]
        min_dist2 = float('inf')
        cx, cy = int(self.x // SYS_CELL_SIZE), int(self.y // SYS_CELL_SIZE)

        neighbor_cells = [
            (cx+dx, cy+dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        ]

        nearby_plants = []
        nearby_carns = []
        if plant_grid:
            for cell in neighbor_cells:
                nearby_plants.extend(plant_grid.get(cell, []))
        if carn_grid:
            for cell in neighbor_cells:
                nearby_carns.extend(carn_grid.get(cell, []))

        for obj in nearby_plants + [c for c in nearby_carns if c.alive]:
            dx, dy = obj.x - self.x, obj.y - self.y
            dist2 = dx*dx + dy*dy
            if dist2 < HERB_VISION_CONE_LENGTH**2:
                angle_to = math.atan2(dy, dx)
                diff = (angle_to - self.rotation + math.pi) % (2*math.pi) - math.pi
                if abs(diff) < HERB_VISION_CONE_WIDTH:
                    if dist2 < min_dist2:
                        min_dist2 = dist2
                        if isinstance(obj, Plant):
                            rgb = [c / 255 for c in obj.color]
                        else:
                            r, g, b = int(obj.color[1:3], 16), int(obj.color[3:5], 16), int(obj.color[5:7], 16)
                            rgb = [r/255, g/255, b/255]

        energy_norm = max(0.0, min(self.energy / HERB_REPRODUCTION_THRESHOLD, 1.0))
        return rgb + [energy_norm]

    def update(self, canvas, herbivores, plants, carnivores, field_w, field_h, plant_grid, carn_grid):
        if not self.alive:
            return
        
        # Age
        self.age += 1
        if self.age >= self.lifespan:
            self.die(canvas, cause="old_age")
            return

        # Metabolism
        self.energy -= HERB_METABOLISM
        if self.energy <= 0:
            self.die(canvas)
            return

        # Neural network
        inputs = self.get_inputs(carnivores, plants, field_w, field_h, plant_grid, carn_grid)
        rotate_out, move_out = self.nn.forward(inputs)

        # Rotation
        if rotate_out > HERB_ROTATE_THRESHOLD:
            self.rotation += rotate_out / HERB_ROTATE_MUL
            self.energy -= (rotate_out / HERB_ROTATE_MUL) / HERB_METABOLISM_ROTATE_ADD_INV
        elif rotate_out < -HERB_ROTATE_THRESHOLD:
            self.rotation += rotate_out / HERB_ROTATE_MUL
            self.energy -= (rotate_out / HERB_ROTATE_MUL) / -HERB_METABOLISM_ROTATE_ADD_INV

        # Movement
        if move_out > HERB_SPEED_THRESHOLD:
            self.speed = move_out * HERB_SPEED_MUL
            self.energy -= self.speed / HERB_METABOLISM_SPEED_ADD_INV
        elif move_out < -HERB_SPEED_THRESHOLD:
            self.speed = move_out * HERB_SPEED_MUL_REV
            self.energy -= (move_out * HERB_SPEED_MUL) / -HERB_METABOLISM_SPEED_ADD_INV
        else:
            self.speed = 0

        # Update pos
        self.x = (self.x + math.cos(self.rotation) * self.speed) % field_w
        self.y = (self.y + math.sin(self.rotation) * self.speed) % field_h

        cell_x = int(self.x // SYS_CELL_SIZE)
        cell_y = int(self.y // SYS_CELL_SIZE)

        # Collect nearby plants
        nearby_plants = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = cell_x + dx, cell_y + dy
                key = (nx, ny)
                if key in plant_grid:
                    nearby_plants.extend(plant_grid[key])

        # Eating plants
        for plant in nearby_plants[:]:
            dx, dy = plant.x - self.x, plant.y - self.y
            dist = dx * dx + dy * dy
            if dist < (self.radius + plant.size / 2) ** 2:
                self.energy += HERB_ENERGY_GAIN
                if self.energy > HERB_REPRODUCTION_THRESHOLD:
                    self.energy -= (HERB_REPRODUCTION_THRESHOLD - HERB_REPRODUCTION_RETURN)
                    child = Herbivore(canvas,
                                      self.x + random.randint(-20, 20),
                                      self.y + random.randint(-20, 20),
                                      parent=self)
                    herbivores.append(child)

                if plant in plants:
                    plant.remove(canvas)
                    plants.remove(plant)

        # Draw
        canvas.coords(self.shape,
                      self.x - self.radius, self.y - self.radius,
                      self.x + self.radius, self.y + self.radius)
        canvas.coords(self.facing_line,
                      self.x, self.y,
                      self.x + math.cos(self.rotation) * self.radius,
                      self.y + math.sin(self.rotation) * self.radius)

        left_angle = self.rotation - HERB_VISION_CONE_WIDTH
        right_angle = self.rotation + HERB_VISION_CONE_WIDTH
        x1, y1 = self.x + math.cos(left_angle) * HERB_VISION_CONE_LENGTH, self.y + math.sin(left_angle) * HERB_VISION_CONE_LENGTH
        x2, y2 = self.x + math.cos(right_angle) * HERB_VISION_CONE_LENGTH, self.y + math.sin(right_angle) * HERB_VISION_CONE_LENGTH
        canvas.coords(self.vision_poly, self.x, self.y, x1, y1, x2, y2)

    def die(self, canvas, cause="starvation"):
        self.alive = False
        self.death_cause = cause
        canvas.delete(self.shape)
        canvas.delete(self.facing_line)
        canvas.delete(self.vision_poly)


# ----------------------
# Carnivore
# ----------------------
class Carnivore:
    _id_counter = 1

    def __init__(self, canvas, x, y, radius=CARN_RADIUS_START, parent=None):
        self.id = Carnivore._id_counter
        Carnivore._id_counter += 1

        if parent:
            pr = int(parent.color[1:3], 16)
            pg = int(parent.color[3:5], 16)
            pb = int(parent.color[5:7], 16)
            pr = min(max(pr + random.randint(-CARN_COLOR_MUTATE_RAND, CARN_COLOR_MUTATE_RAND), 0), 255)
            pg = min(max(pg + random.randint(-CARN_COLOR_MUTATE_RAND, CARN_COLOR_MUTATE_RAND), 0), 255)
            pb = min(max(pb + random.randint(-CARN_COLOR_MUTATE_RAND, CARN_COLOR_MUTATE_RAND), 0), 255)
            self.color = f"#{pr:02x}{pg:02x}{pb:02x}"
        else:
            self.color = "#{:02x}{:02x}{:02x}".format(
                random.randint(CARN_START_COLOR_R_0, CARN_START_COLOR_R_1), 
                random.randint(CARN_START_COLOR_G_0, CARN_START_COLOR_G_1), 
                random.randint(CARN_START_COLOR_B_0, CARN_START_COLOR_B_1), 
            )

        self.radius = radius
        self.x, self.y = x, y
        self.rotation = random.uniform(0, 2 * math.pi)
        self.speed = 0

        if parent:
            self.energy = random.uniform(CARN_ENERGY_START_MIN, CARN_ENERGY_START_MAX)
            self.generation = parent.generation + 1
        else: 
            self.energy = CARN_BORN_ENERGY
            self.generation = 1
        
        self.alive = True

        self.age = 0
        self.lifespan = random.randint(CARN_LIFESPAN_MIN, CARN_LIFESPAN_MAX)

        input_size = 4 # R, G, B, Energy
        if parent:
            self.nn = NeuralNetwork(input_size, CARN_NN_HIDDEN_SIZE, 2,
                                    w1=parent.nn.w1, b1=parent.nn.b1,
                                    w2=parent.nn.w2, b2=parent.nn.b2)
        else:
            self.nn = NeuralNetwork(input_size, CARN_NN_HIDDEN_SIZE, 2)

        self.shape = canvas.create_oval(0, 0, 0, 0, fill=self.color, outline="")
        self.facing_line = canvas.create_line(0, 0, 0, 0, fill="black")
        self.vision_poly = canvas.create_polygon(0, 0, 0, 0, 0, 0, fill="red", stipple="gray25", outline="")

    def get_inputs(self, herbivores, carnivores, field_w, field_h, herb_grid=None, carn_grid=None):
        rgb = [-1, -1, -1]
        min_dist2 = float('inf')
        cx, cy = int(self.x // SYS_CELL_SIZE), int(self.y // SYS_CELL_SIZE)

        neighbor_cells = [
            (cx+dx, cy+dy)
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
        ]

        nearby_herbs = []
        if herb_grid:
            for cell in neighbor_cells:
                nearby_herbs.extend(herb_grid.get(cell, []))

        for prey in [h for h in nearby_herbs if h.alive]:
            dx, dy = prey.x - self.x, prey.y - self.y
            dist2 = dx*dx + dy*dy
            if dist2 < CARN_VISION_CONE_LENGTH**2:
                angle_to = math.atan2(dy, dx)
                diff = (angle_to - self.rotation + math.pi) % (2*math.pi) - math.pi
                if abs(diff) < CARN_VISION_CONE_WIDTH:
                    if dist2 < min_dist2:
                        min_dist2 = dist2
                        r, g, b = int(prey.color[1:3], 16), int(prey.color[3:5], 16), int(prey.color[5:7], 16)
                        rgb = [r/255, g/255, b/255]

        energy_norm = max(0.0, min(self.energy / CARN_REPRODUCTION_THRESHOLD, 1.0))
        return rgb + [energy_norm]

    def update(self, canvas, herbivores, carnivores, field_w, field_h, herb_grid, carn_grid):
        if not self.alive:
            return

        # Age
        self.age += 1
        if self.age >= self.lifespan:
            self.die(canvas)
            return

        # Metabolism
        self.energy -= CARN_METABOLISM
        if self.energy <= 0:
            self.die(canvas)
            return

        # Neural network
        inputs = self.get_inputs(herbivores, carnivores, field_w, field_h, herb_grid, carn_grid)
        rotate_out, move_out = self.nn.forward(inputs)

        # Rotation
        if rotate_out > CARN_ROTATE_THRESHOLD:
            self.rotation += rotate_out / CARN_ROTATE_MUL
            self.energy -= (rotate_out / CARN_ROTATE_MUL) / CARN_METABOLISM_ROTATE_ADD_INV
        elif rotate_out < -CARN_ROTATE_THRESHOLD:
            self.rotation += rotate_out / CARN_ROTATE_MUL
            self.energy -= (rotate_out / CARN_ROTATE_MUL) / -CARN_METABOLISM_ROTATE_ADD_INV

        # Movement
        if move_out > CARN_SPEED_THRESHOLD:
            self.speed = move_out * CARN_SPEED_MUL
            self.energy -= self.speed / CARN_METABOLISM_SPEED_ADD_INV
        elif move_out < -CARN_SPEED_THRESHOLD:
            self.speed = move_out * CARN_SPEED_MUL_REV
            self.energy -= (move_out * CARN_SPEED_MUL) / -CARN_METABOLISM_SPEED_ADD_INV
        else:
            self.speed = 0

        # Update position
        self.x = (self.x + math.cos(self.rotation) * self.speed) % field_w
        self.y = (self.y + math.sin(self.rotation) * self.speed) % field_h

        cell_x = int(self.x // SYS_CELL_SIZE)
        cell_y = int(self.y // SYS_CELL_SIZE)

        # Collect nearby herbivores
        nearby_herbs = []
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                nx, ny = cell_x + dx, cell_y + dy
                key = (nx, ny)
                if key in herb_grid:
                    nearby_herbs.extend(herb_grid[key])

        # Eating herbivores
        for prey in nearby_herbs[:]:
            if not prey.alive:
                continue
            dx, dy = prey.x - self.x, prey.y - self.y
            dist = dx * dx + dy * dy
            if dist < (self.radius + prey.radius / 1.2) ** 2: # divisor of 1 is a big hitbox, 2 is a small hitbox
                prey.die(canvas, cause="eaten")
                self.energy += CARN_ENERGY_GAIN
                if self.energy > CARN_REPRODUCTION_THRESHOLD:
                    self.energy -= (CARN_REPRODUCTION_THRESHOLD - CARN_REPRODUCTION_RETURN)
                    child = Carnivore(canvas,
                                      self.x + random.randint(-20, 20),
                                      self.y + random.randint(-20, 20),
                                      parent=self)
                    carnivores.append(child)

        # Draw
        canvas.coords(self.shape,
                      self.x - self.radius, self.y - self.radius,
                      self.x + self.radius, self.y + self.radius)
        canvas.coords(self.facing_line,
                      self.x, self.y,
                      self.x + math.cos(self.rotation) * self.radius,
                      self.y + math.sin(self.rotation) * self.radius)
        left_angle = self.rotation - CARN_VISION_CONE_WIDTH
        right_angle = self.rotation + CARN_VISION_CONE_WIDTH
        x1, y1 = self.x + math.cos(left_angle) * CARN_VISION_CONE_LENGTH, self.y + math.sin(left_angle) * CARN_VISION_CONE_LENGTH
        x2, y2 = self.x + math.cos(right_angle) * CARN_VISION_CONE_LENGTH, self.y + math.sin(right_angle) * CARN_VISION_CONE_LENGTH
        canvas.coords(self.vision_poly, self.x, self.y, x1, y1, x2, y2)

    def die(self, canvas):
        self.alive = False
        canvas.delete(self.shape)
        canvas.delete(self.facing_line)
        canvas.delete(self.vision_poly)


# ----------------------
# Evolution Simulator
# ----------------------
class EvolutionSimulator:
    def __init__(self, root):
        self.root = root
        self.root.title("Evolution Simulator")

        # Field
        self.field_w, self.field_h = SYS_FIELD_WIDTH, SYS_FIELD_HEIGHT

        # Main layout frame
        self.frame = tk.Frame(root)
        self.frame.pack(fill="both", expand=True)

        # LEFT PANEL
        self.left_panel = tk.Frame(self.frame, width=400, bg="#eee")
        self.left_panel.pack(side="left", fill="y")

        # Speed controls
        self.speed_frame = tk.Frame(self.left_panel, bg="#eee")
        self.speed_frame.pack(pady=10)
        self.speed_label = tk.Label(self.speed_frame, text="Speed: 1x", bg="#eee", font=("Arial", 10, "bold"))
        self.speed_label.pack()
        tk.Button(self.speed_frame, text=" << ", command=self.decrease_speed).pack(side="left", padx=5, pady=5)
        tk.Button(self.speed_frame, text=" >> ", command=self.increase_speed).pack(side="right", padx=5, pady=5)

        # Graph
        self.tick_count = 0
        self.sim_data = []
        self.fig = Figure(figsize=(3.5, 2.2), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Population")
        self.ax.set_xlabel("Ticks")
        self.ax.set_ylabel("Count")
        self.line_plants, = self.ax.plot([], [], label="Plants (x10)", color="green")
        self.line_herbs, = self.ax.plot([], [], label="Herbivores", color="blue")
        self.line_carns, = self.ax.plot([], [], label="Carnivores", color="red")
        self.ax.legend(loc="upper left", fontsize=8)

        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=self.left_panel)
        self.canvas_graph.get_tk_widget().pack(pady=10)

        # CENTER CANVAS
        self.canvas_w, self.canvas_h = 800, 800
        self.canvas = tk.Canvas(self.frame, width=self.canvas_w, height=self.canvas_h, bg="white")
        self.canvas.pack(side="left", fill="both", expand=True)

        # RIGHT PANEL
        self.right_panel = tk.Frame(self.frame, width=400, bg="#eee")
        self.right_panel.pack(side="right", fill="y")

        self.info_label = tk.Label(self.right_panel, text="Select an organism", bg="#eee", justify="left", font=("Arial", 10))
        self.info_label.pack(pady=10)

        # Neural network view
        self.nn_canvas = tk.Canvas(self.right_panel, width=380, height=280, bg="white")
        self.nn_canvas.pack(pady=10)

        # State
        self.sim_speed = 1
        self.selected_organism = None
        self.carnivores, self.herbivores, self.plants = [], [], []

        # Camera
        self.camera_x, self.camera_y = 0, 0
        self.scale = 1.0
        self.drag_start = None

        # Create objects
        self.create_random_plants(SYS_START_PLANT_NUM)
        self.create_random_herbivores(SYS_START_HERB_NUM)
        self.create_random_carnivores(SYS_START_CARN_NUM)

        # Bindings
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<ButtonPress-2>", self.start_drag)
        self.canvas.bind("<B2-Motion>", self.do_drag)
        self.canvas.bind("<MouseWheel>", self.do_zoom)

        self.auto_center_and_zoom()

        self.update_loop()

    # ------------------ Camera ------------------
    def start_drag(self, event):
        self.drag_start = (event.x, event.y)

    def do_drag(self, event):
        dx = (event.x - self.drag_start[0]) / self.scale
        dy = (event.y - self.drag_start[1]) / self.scale
        self.camera_x -= dx
        self.camera_y -= dy
        self.drag_start = (event.x, event.y)

    def do_zoom(self, event):
        factor = 1.1 if event.delta > 0 or getattr(event, "num", 0) == 4 else 0.9
        self.scale *= factor
        self.scale = max(0.1, min(self.scale, 5))

    def auto_center_and_zoom(self):
        canvas_ratio = self.canvas_w / self.canvas_h
        field_ratio = self.field_w / self.field_h

        if field_ratio > canvas_ratio:
            self.scale = self.canvas_w / self.field_w * 0.95
        else:
            self.scale = self.canvas_h / self.field_h * 0.95

        self.camera_x = self.field_w / 2 - (self.canvas_w / 2) / self.scale
        self.camera_y = self.field_h / 2 - (self.canvas_h / 2) / self.scale

    # ------------------ Speed ------------------
    def increase_speed(self):
        current_index = SYS_SPEED_LEVELS.index(self.sim_speed) if self.sim_speed in SYS_SPEED_LEVELS else 1
        if current_index < len(SYS_SPEED_LEVELS) - 1:
            self.sim_speed = SYS_SPEED_LEVELS[current_index + 1]
        self.speed_label.config(text=f"Speed: {self.sim_speed}x")

    def decrease_speed(self):
        current_index = SYS_SPEED_LEVELS.index(self.sim_speed) if self.sim_speed in SYS_SPEED_LEVELS else 1
        if current_index > 0:
            self.sim_speed = SYS_SPEED_LEVELS[current_index - 1]
        self.speed_label.config(text=f"Speed: {self.sim_speed}x" if self.sim_speed > 0 else "Paused")

    # ------------------ Organisms & Plants ------------------
    def create_random_herbivores(self, count):
        for _ in range(count):
            x = random.randint(50, self.field_w-50)
            y = random.randint(50, self.field_h-50)
            self.herbivores.append(Herbivore(self.canvas, x, y))

    def create_random_carnivores(self, count):
        for _ in range(count):
            x = random.randint(50, self.field_w-50)
            y = random.randint(50, self.field_h-50)
            self.carnivores.append(Carnivore(self.canvas, x, y))

    def create_random_plants(self, count):
        for _ in range(count):
            x = random.randint(20, self.field_w-20)
            y = random.randint(20, self.field_h-20)
            self.plants.append(Plant(self.canvas, x, y))

    # ------------------ Click & Info ------------------
    def on_click(self, event):
        wx = (event.x / self.scale) + self.camera_x
        wy = (event.y / self.scale) + self.camera_y
        for herb in self.herbivores:
            if herb.alive:
                dx = wx - herb.x
                dy = wy - herb.y
                if dx*dx + dy*dy <= herb.radius*herb.radius:
                    self.selected_organism = herb
                    break
        for carn in self.carnivores:
            if carn.alive:
                dx = wx - carn.x
                dy = wy - carn.y
                if dx*dx + dy*dy <= carn.radius*carn.radius:
                    self.selected_organism = carn
                    break

    def display_info(self, organism):
        rot_deg = math.degrees(organism.rotation) % 360

        inputs = None
        if isinstance(organism, Herbivore):
            inputs = organism.get_inputs(
                self.carnivores,
                self.plants,
                self.field_w,
                self.field_h,
                plant_grid=getattr(self, "plant_grid", None),
                carn_grid=getattr(self, "carn_grid", None)
            )
        elif isinstance(organism, Carnivore):
            inputs = organism.get_inputs(
                self.herbivores,
                self.carnivores,
                self.field_w,
                self.field_h,
                herb_grid=getattr(self, "herb_grid", None),
                carn_grid=getattr(self, "carn_grid", None)
            )

        if hasattr(organism, "nn"):
            self.draw_nn(organism.nn, inputs=inputs)

        info_text = (
            f"{'Herbivore' if isinstance(organism, Herbivore) else 'Carnivore'} #{organism.id} (Generation {organism.generation})\n"
            f"Color: {organism.color}\n"
            f"Pos: ({int(organism.x)}, {int(organism.y)})\n"
            f"Rotation: {round(rot_deg, 1)}°\n"
            f"Speed: {round(organism.speed, 2)}\n"
            f"Energy: {round(organism.energy, 1)}\n"
            f"Age: {organism.age}/{organism.lifespan}\n"
        )

        self.info_label.config(text=info_text)

    def draw_nn(self, nn, inputs=None):
        self.nn_canvas.delete("all")

        if inputs is None:
            inputs = [0.0] * len(nn.w1[0])

        hidden = [0.0 for _ in range(len(nn.w1))]
        for j in range(len(nn.w1)):
            s = nn.b1[j]
            for i in range(len(inputs)):
                s += inputs[i] * nn.w1[j][i]
            hidden[j] = max(0, s)

        outputs = [0.0 for _ in range(len(nn.w2))]
        for j in range(len(nn.w2)):
            s = nn.b2[j]
            for i in range(len(hidden)):
                s += hidden[i] * nn.w2[j][i]
            outputs[j] = math.tanh(s)

        layers = [inputs, hidden, outputs]
        sizes = [len(layer) for layer in layers]

        # Layout
        x_spacing = 100
        y_spacing = 40
        positions = []
        for li, size in enumerate(sizes):
            px = 80 + li * x_spacing
            py_start = 40
            layer_pos = []
            for j in range(size):
                py = py_start + j * y_spacing
                layer_pos.append((px, py))
            positions.append(layer_pos)

        # Draw weights (input & hidden)
        for i, (x1, y1) in enumerate(positions[0]):
            for j, (x2, y2) in enumerate(positions[1]):
                w = nn.w1[j][i]
                color = "blue" if w > 0 else "red"
                width = max(1, int(abs(w) * 2))
                self.nn_canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

        # Draw weights (hidden & output)
        for i, (x1, y1) in enumerate(positions[1]):
            for j, (x2, y2) in enumerate(positions[2]):
                w = nn.w2[j][i]
                color = "blue" if w > 0 else "red"
                width = max(1, int(abs(w) * 2))
                self.nn_canvas.create_line(x1, y1, x2, y2, fill=color, width=width)

        # Draw neurons
        for li, layer in enumerate(layers):
            for j, val in enumerate(layer):
                x, y = positions[li][j]
                intensity = int((val + 1) / 2 * 255) if li == 2 else int(max(0, min(1, val)) * 255)
                fill_color = f"#{255-intensity:02x}{255-intensity:02x}{255-intensity:02x}"
                neuron_size = 16
                self.nn_canvas.create_oval(x-neuron_size, y-neuron_size, x+neuron_size, y+neuron_size, fill=fill_color, outline="black")
                brightness = (255-intensity)
                text_color = "black" if brightness > 128 else "white"
                self.nn_canvas.create_text(x, y, text=f"{val:.2f}", font=("Arial", 8), fill=text_color)

        # Labels
        input_labels = ["R", "G", "B", "Energy"][:len(positions[0])]
        output_labels = ["Turn", "Move"][:len(positions[2])]

        # Input labels
        for i, (x, y) in enumerate(positions[0]):
            label = input_labels[i] if i < len(input_labels) else f"In{i+1}"
            self.nn_canvas.create_text(x - 20, y, text=label, font=("Arial", 9, "bold"), fill="black", anchor="e")

        # Output labels
        for i, (x, y) in enumerate(positions[2]):
            label = output_labels[i] if i < len(output_labels) else f"Out{i+1}"
            self.nn_canvas.create_text(x + 20, y, text=label, font=("Arial", 9, "bold"), fill="black", anchor="w")

    # ------------------ Graph & Data ------------------
    def update_data(self):
        if not self.plants and not self.herbivores and not self.carnivores:
            return

        total_plants = len(self.plants)
        total_herbs = sum(1 for h in self.herbivores if h.alive)
        total_carns = sum(1 for c in self.carnivores if c.alive)

        herb_starve_deaths = sum(1 for h in self.herbivores if not h.alive and getattr(h, "death_cause", "") == "starvation")
        herb_eaten_deaths = sum(1 for h in self.herbivores if not h.alive and getattr(h, "death_cause", "") == "eaten")

        self.sim_data.append([
            total_plants,
            total_herbs,
            total_carns,
            herb_starve_deaths,
            herb_eaten_deaths
        ])

        if len(self.sim_data) > SYS_GRAPH_MEMORY:
            self.sim_data.pop(0)

        self.tick_count += 1

    def update_graphs(self):
        if not self.sim_data:
            return

        data = list(zip(*self.sim_data))
        ticks = range(len(self.sim_data))

        plants = [v / 10 for v in data[0]]
        herbs = list(data[1])
        carns = list(data[2])
        starve_deaths = list(data[3]) # todo
        eaten_deaths = list(data[4])

        self.line_plants.set_data(ticks, plants)
        self.line_herbs.set_data(ticks, herbs)
        self.line_carns.set_data(ticks, carns)

        self.ax.set_xlim(0, len(self.sim_data))
        ymax = max(max(plants + herbs + carns), 1) + 5
        self.ax.set_ylim(0, ymax)

        self.canvas_graph.draw()

    # ------------------ Main Loop ------------------
    def update_loop(self):
        # Skip if paused
        if self.sim_speed == 0:
            self.root.after(100, self.update_loop)
            return

        for _ in range(self.sim_speed):
            # Build spatial grids
            self.plant_grid = defaultdict(list)
            self.herb_grid = defaultdict(list)
            self.carn_grid = defaultdict(list)

            for p in self.plants:
                cell = (int(p.x // SYS_CELL_SIZE), int(p.y // SYS_CELL_SIZE))
                self.plant_grid[cell].append(p)

            for h in self.herbivores:
                cell = (int(h.x // SYS_CELL_SIZE), int(h.y // SYS_CELL_SIZE))
                self.herb_grid[cell].append(h)

            for c in self.carnivores:
                cell = (int(c.x // SYS_CELL_SIZE), int(c.y // SYS_CELL_SIZE))
                self.carn_grid[cell].append(c)

            # Update all organisms
            for plant in self.plants[:]:
                plant.try_duplicate(self.canvas, self.plants, self.field_w, self.field_h)
            for herb in self.herbivores:
                herb.update(self.canvas, self.herbivores, self.plants, self.carnivores,
                            self.field_w, self.field_h,
                            self.plant_grid, self.carn_grid)
            for carn in self.carnivores:
                carn.update(self.canvas, self.herbivores, self.carnivores,
                            self.field_w, self.field_h,
                            self.herb_grid, self.carn_grid)
                
            # Update data
            self.update_data()

        # Redraw objects
        for plant in self.plants:
            x = (plant.x - self.camera_x) * self.scale
            y = (plant.y - self.camera_y) * self.scale
            r = plant.size / 2 * self.scale
            self.canvas.coords(plant.shape, x - r, y - r, x + r, y + r)

        for herb in self.herbivores:
            if herb.alive:
                x = (herb.x - self.camera_x) * self.scale
                y = (herb.y - self.camera_y) * self.scale
                r = herb.radius * self.scale

                # Vision cone parameters
                cone_length = HERB_VISION_CONE_LENGTH * self.scale
                half_angle = HERB_VISION_CONE_WIDTH

                # Calculate vision cone
                left_angle = herb.rotation - half_angle
                right_angle = herb.rotation + half_angle

                x1 = x + math.cos(left_angle) * cone_length
                y1 = y + math.sin(left_angle) * cone_length
                x2 = x + math.cos(right_angle) * cone_length
                y2 = y + math.sin(right_angle) * cone_length

                # Body & facing line
                self.canvas.coords(herb.shape, x - r, y - r, x + r, y + r)
                self.canvas.coords(herb.facing_line, x, y,
                                   x + math.cos(herb.rotation) * r,
                                   y + math.sin(herb.rotation) * r)

                # Update vision cone
                self.canvas.coords(herb.vision_poly, x, y, x1, y1, x2, y2)
            else:
                self.canvas.itemconfig(herb.shape, state='hidden')
                self.canvas.itemconfig(herb.facing_line, state='hidden')
                self.canvas.itemconfig(herb.vision_poly, state='hidden')

        for carn in self.carnivores:
            if carn.alive:
                x = (carn.x - self.camera_x) * self.scale
                y = (carn.y - self.camera_y) * self.scale
                r = carn.radius * self.scale

                # Vision cone parameters
                cone_length = CARN_VISION_CONE_LENGTH * self.scale
                half_angle = CARN_VISION_CONE_WIDTH

                # Calculate vision cone
                left_angle = carn.rotation - half_angle
                right_angle = carn.rotation + half_angle

                x1 = x + math.cos(left_angle) * cone_length
                y1 = y + math.sin(left_angle) * cone_length
                x2 = x + math.cos(right_angle) * cone_length
                y2 = y + math.sin(right_angle) * cone_length

                # Body & facing line
                self.canvas.coords(carn.shape, x - r, y - r, x + r, y + r)
                self.canvas.coords(carn.facing_line, x, y,
                                   x + math.cos(carn.rotation) * r,
                                   y + math.sin(carn.rotation) * r)

                # Update vision cone
                self.canvas.coords(carn.vision_poly, x, y, x1, y1, x2, y2)
            else:
                self.canvas.itemconfig(carn.shape, state='hidden')
                self.canvas.itemconfig(carn.facing_line, state='hidden')
                self.canvas.itemconfig(carn.vision_poly, state='hidden')

        if self.selected_organism and self.selected_organism.alive:
            self.display_info(self.selected_organism)
        else:
            self.selected_organism = None
            self.info_label.config(text="Select an organism")

        # Rerender graph
        if not any(h.alive for h in self.herbivores) or not any(c.alive for c in self.carnivores) or len(self.plants) == 0:
            return
        else:
            self.update_graphs()

        # Next frame
        self.root.after(int(100 / self.sim_speed), self.update_loop)


if __name__ == "__main__":
    root = tk.Tk()
    app = EvolutionSimulator(root)
    root.mainloop()
