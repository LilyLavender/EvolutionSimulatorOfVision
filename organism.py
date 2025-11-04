import random
from variables import *
from neural_network import NeuralNetwork

# Base class, should not be instantiated
class Organism:
    _id_counter = 1
    
    def __init__(self, canvas, x, y, radius, color_range, color_mutate_rand, energy_start, born_energy,
                 lifespan_range, nn_hidden_size, reproduction_threshold, reproduction_return,
                 rotate_threshold, rotate_mul, metabolism, metabolism_rot_add_inv,
                 speed_threshold, speed_mul, speed_mul_rev, metabolism_speed_add_inv,
                 vision_length, vision_width, vision_color, parent=None):

        # Assign ID
        cls = type(self)
        if not hasattr(cls, "_id_counter"):
            cls._id_counter = 1
        self.id = cls._id_counter
        cls._id_counter += 1

        # Color
        if parent:
            pr = int(parent.color[1:3], 16)
            pg = int(parent.color[3:5], 16)
            pb = int(parent.color[5:7], 16)
            pr = min(max(pr + random.randint(-color_mutate_rand, color_mutate_rand), 0), 255)
            pg = min(max(pg + random.randint(-color_mutate_rand, color_mutate_rand), 0), 255)
            pb = min(max(pb + random.randint(-color_mutate_rand, color_mutate_rand), 0), 255)
            self.color = f"#{pr:02x}{pg:02x}{pb:02x}"
        else:
            self.color = "#{:02x}{:02x}{:02x}".format(
                random.randint(color_range[0][0], color_range[0][1]),
                random.randint(color_range[1][0], color_range[1][1]),
                random.randint(color_range[2][0], color_range[2][1]),
            )

        self.x, self.y = x, y
        self.radius = radius
        self.rotation = random.uniform(0, 2 * math.pi)
        self.speed = 0
        self.alive = True
        self.age = 0
        self.lifespan = random.randint(*lifespan_range)
        self.gestating = False
        self.gestation_timer = 0
        self.child_class = None
        self.energy = born_energy if parent else random.uniform(*energy_start)
        self.generation = parent.generation + 1 if parent else 1

        # Neural network
        input_size = 4
        if parent:
            self.nn = NeuralNetwork(input_size, nn_hidden_size, 2,
                                    w1=parent.nn.w1, b1=parent.nn.b1,
                                    w2=parent.nn.w2, b2=parent.nn.b2)
        else:
            self.nn = NeuralNetwork(input_size, nn_hidden_size, 2)

        # Vision cone + body
        self.shape = canvas.create_oval(0, 0, 0, 0, fill=self.color, outline="")
        self.facing_line = canvas.create_line(0, 0, 0, 0, fill="black")
        self.vision_poly = canvas.create_polygon(0, 0, 0, 0, 0, 0, fill=vision_color, stipple="gray25", outline="")

        self.rotate_threshold = rotate_threshold
        self.rotate_mul = rotate_mul
        self.metabolism = metabolism
        self.metabolism_rot_add_inv = metabolism_rot_add_inv
        self.speed_threshold = speed_threshold
        self.speed_mul = speed_mul
        self.speed_mul_rev = speed_mul_rev
        self.metabolism_speed_add_inv = metabolism_speed_add_inv
        self.reproduction_threshold = reproduction_threshold
        self.reproduction_return = reproduction_return
        self.vision_length = vision_length
        self.vision_width = vision_width

    def update(self, canvas, field_w, field_h, plant_grid, herb_grid, carn_grid, plants, herbivores, carnivores):
        if not self.alive:
            return

        # Die of old age
        self.age += 1
        if self.age >= self.lifespan:
            self.die(canvas)
            return

        # Die when energy runs out
        self.energy -= self.metabolism
        if self.energy <= 0:
            self.die(canvas)
            return

        # Gestation
        if self.gestating:
            self.gestation_timer -= 1
            if self.gestation_timer <= 0:
                # Give birth
                self.gestating = False
                self.gestation_timer = 0
                if self.child_class:
                    child = self.child_class(canvas,
                                             self.x + random.randint(-20, 20),
                                             self.y + random.randint(-20, 20),
                                             parent=self)
                    
                    from herbivore import Herbivore
                    from carnivore import Carnivore

                    if isinstance(self, Herbivore):
                        herbivores.append(child)
                    elif isinstance(self, Carnivore):
                        carnivores.append(child)

        # Brain
        inputs = self.get_inputs(plant_grid, herb_grid, carn_grid)
        rotate_out, move_out = self.nn.forward(inputs)

        # Rotation
        if abs(rotate_out) > self.rotate_threshold:
            self.rotation += rotate_out / self.rotate_mul
            self.energy -= abs(rotate_out / self.rotate_mul) / self.metabolism_rot_add_inv

        # Movement
        if abs(move_out) > self.speed_threshold:
            mul = self.speed_mul if move_out > 0 else self.speed_mul_rev
            self.speed = move_out * mul
            self.energy -= abs(self.speed) / self.metabolism_speed_add_inv
        else:
            self.speed = 0

        self.x = (self.x + math.cos(self.rotation) * self.speed) % field_w
        self.y = (self.y + math.sin(self.rotation) * self.speed) % field_h

        # Eating
        self.eat_targets(canvas, plant_grid, herb_grid, carn_grid, plants, herbivores, carnivores)

        # Draw
        self.draw(canvas)

    def draw(self, canvas):
        canvas.coords(self.shape,
                      self.x - self.radius, self.y - self.radius,
                      self.x + self.radius, self.y + self.radius)
        canvas.coords(self.facing_line,
                      self.x, self.y,
                      self.x + math.cos(self.rotation) * self.radius,
                      self.y + math.sin(self.rotation) * self.radius)
        left_angle = self.rotation - self.vision_width
        right_angle = self.rotation + self.vision_width
        x1, y1 = self.x + math.cos(left_angle) * self.vision_length, self.y + math.sin(left_angle) * self.vision_length
        x2, y2 = self.x + math.cos(right_angle) * self.vision_length, self.y + math.sin(right_angle) * self.vision_length
        canvas.coords(self.vision_poly, self.x, self.y, x1, y1, x2, y2)

    def die(self, canvas, cause="starvation"):
        self.alive = False
        self.death_cause = cause
        canvas.delete(self.shape)
        canvas.delete(self.facing_line)
        canvas.delete(self.vision_poly)
