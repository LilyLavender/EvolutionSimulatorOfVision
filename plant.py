import random
from variables import *

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
