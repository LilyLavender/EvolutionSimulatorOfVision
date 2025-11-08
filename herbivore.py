import random
from organism import Organism
from plant import Plant
from variables import *

class Herbivore(Organism):
    def __init__(self, canvas, x, y, parent=None):
        super().__init__(
            canvas, x, y,
            radius=HERB_RADIUS_START,
            color_range=((HERB_START_COLOR_R_0, HERB_START_COLOR_R_1),
                         (HERB_START_COLOR_G_0, HERB_START_COLOR_G_1),
                         (HERB_START_COLOR_B_0, HERB_START_COLOR_B_1)),
            color_mutate_rand=HERB_COLOR_MUTATE_RAND,
            energy_start=(HERB_ENERGY_START_MIN, HERB_ENERGY_START_MAX),
            born_energy=HERB_BORN_ENERGY,
            lifespan_range=(HERB_LIFESPAN_MIN, HERB_LIFESPAN_MAX),
            nn_hidden_size=HERB_NN_HIDDEN_SIZE,
            reproduction_threshold=HERB_REPRODUCTION_THRESHOLD,
            reproduction_return=HERB_REPRODUCTION_RETURN,
            rotate_threshold=HERB_ROTATE_THRESHOLD,
            rotate_mul=HERB_ROTATE_MUL,
            metabolism=HERB_METABOLISM,
            metabolism_rot_add_inv=HERB_METABOLISM_ROTATE_ADD_INV,
            speed_threshold=HERB_SPEED_THRESHOLD,
            speed_mul=HERB_SPEED_MUL,
            speed_mul_rev=HERB_SPEED_MUL_REV,
            metabolism_speed_add_inv=HERB_METABOLISM_SPEED_ADD_INV,
            vision_length=HERB_VISION_CONE_LENGTH,
            vision_width=HERB_VISION_CONE_WIDTH,
            vision_color="blue",
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

        nearby_plants = []
        nearby_carns = []
        if plant_grid:
            for cell in neighbor_cells:
                nearby_plants.extend(plant_grid.get(cell, []))
        if carn_grid:
            for cell in neighbor_cells:
                nearby_carns.extend([c for c in carn_grid.get(cell, []) if c.alive])

        # Carnivores
        for carn in nearby_carns:
            dx, dy = carn.x - self.x, carn.y - self.y
            dist2 = dx*dx + dy*dy
            if dist2 < HERB_VISION_CONE_LENGTH**2:
                angle_to = math.atan2(dy, dx)
                diff = (angle_to - self.rotation + math.pi) % (2*math.pi) - math.pi
                if abs(diff) < HERB_VISION_CONE_WIDTH:
                    if dist2 < min_dist2:
                        min_dist2 = dist2
                        r, g, b = int(carn.color[1:3], 16), int(carn.color[3:5], 16), int(carn.color[5:7], 16)
                        rgb = [r/255, g/255, b/255]

        # Plants
        if min_dist2 == float('inf'):
            for plant in nearby_plants:
                dx, dy = plant.x - self.x, plant.y - self.y
                dist2 = dx*dx + dy*dy
                if dist2 < HERB_VISION_CONE_LENGTH**2:
                    angle_to = math.atan2(dy, dx)
                    diff = (angle_to - self.rotation + math.pi) % (2*math.pi) - math.pi
                    if abs(diff) < HERB_VISION_CONE_WIDTH:
                        if dist2 < min_dist2:
                            min_dist2 = dist2
                            rgb = [c / 255 for c in plant.color]

        energy_norm = max(0.0, min(self.energy / HERB_REPRODUCTION_THRESHOLD, 1.0))
        return rgb + [energy_norm]

    def eat_targets(self, canvas, plant_grid, herb_grid, carn_grid, plants, herbivores, carnivores):
        # Gestation
        if self.gestating:
            return

        # Calculate cells
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
                if self.energy > HERB_REPRODUCTION_THRESHOLD and not self.gestating:
                    self.energy -= (HERB_REPRODUCTION_THRESHOLD - HERB_REPRODUCTION_RETURN)
                    self.gestating = True
                    self.gestation_timer = HERB_GESTATION_PERIOD
                    self.child_class = Herbivore

                if plant in plants:
                    plant.remove(canvas)
                    plants.remove(plant)
