import math

# COMMON VARIABLES
SYS_FIELD_WIDTH = 2400
SYS_FIELD_HEIGHT = 2400
SYS_CELL_SIZE = 200         # The field is divided into cells to save processing power. This is the size of individual cells. Should be at least the vision lengths of carnivores. More cells (smaller cell size) is most efficient
SYS_START_PLANT_NUM = 2000
SYS_START_HERB_NUM = 100
SYS_START_CARN_NUM = 40
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
PLANT_SPREAD_MIN = 20                   # Shortest distance between any two plants
PLANT_SPREAD_MAX = 32                   # Largest distance a child plant can spawn
PLANT_SPREAD_TRY_NUM = 1                # Amount of spawning positions children plants attempt
PLANT_REPRODUCTION_FRAME_MIN = 300
PLANT_REPRODUCTION_FRAME_MAX = 1200
PLANT_REPRODUCTION_START_FRAME_MIN = 600
PLANT_REPRODUCTION_START_FRAME_MAX = 1800

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
HERB_VISION_CONE_LENGTH = 140           # Length of vision cones, in units
HERB_VISION_CONE_WIDTH = math.pi/4.5    # Width of vision cones, in radians
HERB_METABOLISM = 0.032                 # Energy units lost per tick
HERB_METABOLISM_SPEED_ADD_INV = 160     # Additional energy units lost per tick when moving (as a divisor)
HERB_METABOLISM_ROTATE_ADD_INV = 240    # Additional energy units lost per tick when rotating (as a divisor)
HERB_ROTATE_THRESHOLD = 0.3             # Minimum a neuron has to be set to to rotate an organism
HERB_SPEED_THRESHOLD = 0.1              # Minimum a neuron has to be set to cause an organism to move
HERB_ROTATE_MUL = 1.4                   # Degrees per frame an organism rotates, multiplied by their rotate neuron strength
HERB_SPEED_MUL = 2.8                    # Units per frame an organism moves when moving foward, multiplied by their speed neuron strength
HERB_SPEED_MUL_REV = 1.2                # Units per frame an organism moves when moving backward, multiplied by their speed neuron strength
HERB_ENERGY_GAIN = 20                   # Energy units to gain when eating a plant
HERB_REPRODUCTION_THRESHOLD = 300       # Energy units required to reproduce
HERB_REPRODUCTION_RETURN = 80           # Energy units to return to after reproduction (plus any excess)
HERB_BORN_ENERGY = 70                   # Energy units new organisms start with
HERB_ENERGY_START_MIN = 40              # Minimum energy units organisms start with on simulation start
HERB_ENERGY_START_MAX = 80              # Maximum energy units organisms start with on simulation start
HERB_LIFESPAN_MIN = 4000                # Minimum ticks to live for
HERB_LIFESPAN_MAX = 8000                # Maximum ticks to live for
HERB_GESTATION_PERIOD = 250             # Ticks before being able to give birth again

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
CARN_ENERGY_GAIN_PERCENT = 0.275        # Percentage of an herbivore's energy carnivores gain upon eating
CARN_REPRODUCTION_THRESHOLD = 300       # Energy units required to reproduce
CARN_REPRODUCTION_RETURN = 120          # Energy units to return to after reproduction (plus any excess)
CARN_BORN_ENERGY = 120                  # Energy units new organisms start with
CARN_ENERGY_START_MIN = 70              # Minimum energy units organisms start with on simulation start
CARN_ENERGY_START_MAX = 140             # Maximum energy units organisms start with on simulation start
CARN_LIFESPAN_MIN = 9000                # Minimum ticks to live for
CARN_LIFESPAN_MAX = 15000               # Maximum ticks to live for
CARN_GESTATION_PERIOD = 200             # Ticks before being able to give birth again
