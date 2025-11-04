import random
from organism import Organism
from variables import *

class Carnivore(Organism):
    def __init__(self, canvas, x, y, parent=None):
        super().__init__(
            canvas, x, y,
            radius=CARN_RADIUS_START,
            color_range=((CARN_START_COLOR_R_0, CARN_START_COLOR_R_1),
                         (CARN_START_COLOR_G_0, CARN_START_COLOR_G_1),
                         (CARN_START_COLOR_B_0, CARN_START_COLOR_B_1)),
            color_mutate_rand=CARN_COLOR_MUTATE_RAND,
            energy_start=(CARN_ENERGY_START_MIN, CARN_ENERGY_START_MAX),
            born_energy=CARN_BORN_ENERGY,
            lifespan_range=(CARN_LIFESPAN_MIN, CARN_LIFESPAN_MAX),
            nn_hidden_size=CARN_NN_HIDDEN_SIZE,
            reproduction_threshold=CARN_REPRODUCTION_THRESHOLD,
            reproduction_return=CARN_REPRODUCTION_RETURN,
            rotate_threshold=CARN_ROTATE_THRESHOLD,
            rotate_mul=CARN_ROTATE_MUL,
            metabolism=CARN_METABOLISM,
            metabolism_rot_add_inv=CARN_METABOLISM_ROTATE_ADD_INV,
            speed_threshold=CARN_SPEED_THRESHOLD,
            speed_mul=CARN_SPEED_MUL,
            speed_mul_rev=CARN_SPEED_MUL_REV,
            metabolism_speed_add_inv=CARN_METABOLISM_SPEED_ADD_INV,
            vision_length=CARN_VISION_CONE_LENGTH,
            vision_width=CARN_VISION_CONE_WIDTH,
            vision_color="red",
            parent=parent
        )

    def get_inputs(self, plant_grid=None, herb_grid=None, carn_grid=None):
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

    def eat_targets(self, canvas, plant_grid, herb_grid, carn_grid, plants, herbivores, carnivores):
        # Gestation
        if self.gestating:
            return
        
        # Calculate cells
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
                gained_energy = prey.energy * CARN_ENERGY_GAIN_PERCENT
                self.energy += gained_energy
                prey.die(canvas, cause="eaten")
                if self.energy > CARN_REPRODUCTION_THRESHOLD and not self.gestating:
                    self.energy -= (CARN_REPRODUCTION_THRESHOLD - CARN_REPRODUCTION_RETURN)
                    self.gestating = True
                    self.gestation_timer = CARN_GESTATION_PERIOD
                    self.child_class = Carnivore
