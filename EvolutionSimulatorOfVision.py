import tkinter as tk
import random
import math
import matplotlib # type: ignore
matplotlib.use("TkAgg")
from matplotlib.figure import Figure # type: ignore
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # type: ignore
from collections import defaultdict
from variables import *
from plant import Plant
from herbivore import Herbivore
from carnivore import Carnivore

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
        self.fig = Figure(figsize=(3.0, 6.5), dpi=100)

        # Population graph (1)
        self.ax = self.fig.add_subplot(311)
        self.ax.set_title("Population")
        self.ax.set_xlabel("Ticks")
        self.ax.set_ylabel("Count")
        self.line_plants, = self.ax.plot([], [], label="Plants (x10)", color="green")
        self.line_herbs, = self.ax.plot([], [], label="Herbivores", color="blue")
        self.line_carns, = self.ax.plot([], [], label="Carnivores", color="red")
        self.ax.legend(loc="upper left", fontsize=8)

        # Death cause graph (2)
        self.ax2 = self.fig.add_subplot(312)
        self.ax2.set_title("Herbivore Cause of Death (%)")
        self.ax2.set_xlabel("Ticks")
        self.ax2.set_ylabel("Percent of Deaths")
        self.line_starve, = self.ax2.plot([], [], label="Starvation", color="green")
        self.line_eaten, = self.ax2.plot([], [], label="Predation", color="red")
        self.line_oldage, = self.ax2.plot([], [], label="Old Age", color="blue")
        self.ax2.set_ylim(0, 100)
        self.ax2.legend(loc="upper left", fontsize=8)

        # Predator-prey phase plot (3)
        self.ax3 = self.fig.add_subplot(313)
        self.ax3.set_title("Predator–Prey Cycle")
        self.ax3.set_xlabel("Carnivores")
        self.ax3.set_ylabel("Herbivores")
        self.phase_line, = self.ax3.plot([], [], color="black", linewidth=0.75)
        self.ax3.grid(True, linestyle="--", alpha=0.5)
        
        self.fig.subplots_adjust(hspace=0.75)

        self.canvas_graph = FigureCanvasTkAgg(self.fig, master=self.left_panel)
        self.canvas_graph.get_tk_widget().pack(pady=10)

        # CENTER CANVAS
        self.canvas_w, self.canvas_h = 800, 800
        self.canvas = tk.Canvas(self.frame, width=self.canvas_w, height=self.canvas_h, bg="white")
        self.canvas.pack(side="left", fill="both", expand=True)

        # RIGHT PANEL
        self.right_panel = tk.Frame(self.frame, width=400, bg="#eee")
        self.right_panel.pack(side="right", fill="y")

        # Info box
        self.info_box = tk.Text(self.right_panel, width=40, height=7, bg="#eee", relief="flat", font=("Arial", 10))
        self.info_box.tag_configure("bold", font=("Arial", 10, "bold"))
        self.info_box.pack(pady=10)
        self.info_box.insert("end", "Select an organism")
        self.info_box.config(state="disabled")

        # Neural network view
        self.nn_canvas = tk.Canvas(self.right_panel, width=360, height=420, bg="white")
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

        nearest = None
        nearest_dist_sq = float('inf')
        max_click_dist = SYS_MAX_CLICK_DIST / self.scale

        # Get all organisms
        for herb in self.herbivores:
            if herb.alive:
                dx = wx - herb.x
                dy = wy - herb.y
                dist_sq = dx*dx + dy*dy
                if dist_sq <= (herb.radius + max_click_dist)**2 and dist_sq < nearest_dist_sq:
                    nearest = herb
                    nearest_dist_sq = dist_sq
        for carn in self.carnivores:
            if carn.alive:
                dx = wx - carn.x
                dy = wy - carn.y
                dist_sq = dx*dx + dy*dy
                if dist_sq <= (carn.radius + max_click_dist)**2 and dist_sq < nearest_dist_sq:
                    nearest = carn
                    nearest_dist_sq = dist_sq

        # Assign selection
        if nearest is not None:
            self.selected_organism = nearest
        else:
            self.selected_organism = None

    def display_info(self, organism):
        rot_deg = math.degrees(organism.rotation) % 360

        inputs = None
        if isinstance(organism, Herbivore) or isinstance(organism, Carnivore):
            inputs = organism.get_inputs(
                plant_grid=getattr(self, "plant_grid", None),
                herb_grid=getattr(self, "herb_grid", None),
                carn_grid=getattr(self, "carn_grid", None),
            )

        if hasattr(organism, "nn"):
            self.draw_nn(organism.nn, inputs=inputs)

        is_herb = isinstance(organism, Herbivore)
        label_value_pairs = [
            ("", f"{'Herbivore' if is_herb else 'Carnivore'} #{organism.id} (Gen {organism.generation})"),
            ("Color: ", organism.color),
            ("Pos: ", f"({int(organism.x)}, {int(organism.y)})"),
            ("Rotation: ", f"{round(rot_deg, 1)}°"),
            ("Speed: ", f"{round(organism.speed, 2)}"),
            ("Energy: ", f"{round(organism.energy, 1)}"),
            ("Age: ", f"{organism.age}/{organism.lifespan}"),
        ]

        self.info_box.config(state="normal")
        self.info_box.delete("1.0", "end")

        for label, value in label_value_pairs:
            self.info_box.insert("end", label)
            self.info_box.insert("end", value, "bold")
            self.info_box.insert("end", "\n")

        self.info_box.config(state="disabled")

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
        herb_oldage_deaths = sum(1 for h in self.herbivores if not h.alive and getattr(h, "death_cause", "") == "old_age")

        self.sim_data.append([
            total_plants,
            total_herbs,
            total_carns,
            herb_starve_deaths,
            herb_eaten_deaths,
            herb_oldage_deaths,
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
        starve_deaths = list(data[3])
        eaten_deaths = list(data[4])
        oldage_deaths = list(data[5])

        # Population graph (1)
        self.line_plants.set_data(ticks, plants)
        self.line_herbs.set_data(ticks, herbs)
        self.line_carns.set_data(ticks, carns)
        self.ax.set_xlim(0, len(self.sim_data))
        ymax = max(max(plants + herbs + carns), 1) + 5
        self.ax.set_ylim(0, ymax)

        # Death cause graph (2)
        recent_window = SYS_DEATH_WINDOW_SIZE

        # Predator–prey phase plot (3)
        if len(carns) > 0 and len(herbs) > 0:
            self.phase_line.set_data(carns, herbs)
            self.ax3.set_xlim(0, max(carns) + 5)
            self.ax3.set_ylim(0, max(herbs) + 5)

        perc_starve = []
        perc_eaten = []
        perc_oldage = []

        for i in range(len(starve_deaths)):
            start = max(0, i - recent_window)
            s_sum = sum(starve_deaths[start:i+1])
            e_sum = sum(eaten_deaths[start:i+1])
            o_sum = sum(oldage_deaths[start:i+1])
            total = s_sum + e_sum + o_sum

            if total > 0:
                perc_starve.append(s_sum / total * 100)
                perc_eaten.append(e_sum / total * 100)
                perc_oldage.append(o_sum / total * 100)
            else:
                perc_starve.append(0)
                perc_eaten.append(0)
                perc_oldage.append(0)

        self.line_starve.set_data(ticks, perc_starve)
        self.line_eaten.set_data(ticks, perc_eaten)
        self.line_oldage.set_data(ticks, perc_oldage)
        self.ax2.set_xlim(0, len(self.sim_data))
        self.ax2.set_ylim(0, 100)

        # Draw graphs
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
                herb.update(self.canvas, self.field_w, self.field_h, self.plant_grid, self.herb_grid, self.carn_grid, self.plants, self.herbivores, self.carnivores)
            for carn in self.carnivores:
                carn.update(self.canvas, self.field_w, self.field_h, self.plant_grid, self.herb_grid, self.carn_grid, self.plants, self.herbivores, self.carnivores)
                
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

                # Darker when spectated
                stipple = "gray75" if herb is self.selected_organism else "gray25"
                self.canvas.itemconfig(herb.vision_poly, stipple=stipple)
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

                # Darker when spectated
                stipple = "gray75" if carn is self.selected_organism else "gray25"
                self.canvas.itemconfig(carn.vision_poly, stipple=stipple)
            else:
                self.canvas.itemconfig(carn.shape, state='hidden')
                self.canvas.itemconfig(carn.facing_line, state='hidden')
                self.canvas.itemconfig(carn.vision_poly, state='hidden')

        if self.selected_organism and self.selected_organism.alive:
            self.display_info(self.selected_organism)
        else:
            self.selected_organism = None
            self.info_box.config(state="normal")
            self.info_box.delete("1.0", "end")
            self.info_box.insert("end", "Select an organism")
            self.info_box.config(state="disabled")

        # Rerender graph
        if not any(h.alive for h in self.herbivores) or not any(c.alive for c in self.carnivores) or len(self.plants) == 0:
            return
        else:
            if self.tick_count % 1 == 0:
                self.update_graphs()

        # Next frame
        self.root.after(int(100 / self.sim_speed), self.update_loop)


if __name__ == "__main__":
    root = tk.Tk()
    app = EvolutionSimulator(root)
    root.mainloop()
