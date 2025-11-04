# Evolution Simulator of Vision
> A small evolution simulator project to practice machine learning and to explain the concepts of natural selection and genes to others

### About & Customization
- **This project was made in an afternoon and isn't currently being updated anymore. It was more of an experiment of sorts. Nonetheless, a short overview:**
  - When the simulation starts, randomly generated plants, herbivores, and carnivores are placed on the field
  - Herbivores eat plants, and carnivores eat herbivores
  - Organisms display a cone that represents their vision. Their neural network is given the RGB values of the closest thing in their vision and then decides whether to and at what speed to move forward or rotate
  - If an organism's energy value is high enough, it will give birth to another creature with slightly different attributes
  - This simulates real-world natural selection, where organisms with more fit genes will tend to live longer and therefore reproduce, slowly over the course of the simulation creating organisms more fit to live in the environment
- **This evolution simulator was made specifically to test vision and camouflage**
  - Since organisms' colors can slowly drift over the course of many generations and organisms can only see which color is closest in their vision cone, this simulation lends itself well to the fittest organisms being the ones that happen to evolve colors different than those around them
- **This sim also has many variables for the end user to change**
  - In the `variables.py` file, you'll find a list of uppercase variables
  - These decide many different values from the rate of mutations to organisms' max speed
  - They are by default set to values I've found prolong the simulation as long as possible
  - Feel free to change them to customize the simulation to your liking

### Todo
- **Simulation**
  - Make organism size affected by evolution
  - Tie organism size to metabolism
  - Allow herbivores to see other herbivores and carnivores to see plants and other carnivores
  - Omnivore?
- **Data**
  - Automatically download info from all graphs once a simulation is over
  - Graph to show herbivore causes of death (herbivores already track this, needs to be put into a graph)
  - Graph to show average color of plants, herbivores, or carnivores over time
  - Display the number of children an organism has
  - Add a way to see which organism is being spectated (only organism to show vision cone?)
  - Option to automatically spectate another organism once one dies
- **Bugs**
  - Zooming the field currently zooms relative to the top left rather than where the mouse is
  - Plants display above some carnivores. Vision cone z-indexes are also possibly messed up

### Install
1. Download the repository's code
    - Click the green (or blue) `<> Code` button
    - Click `Download ZIP`
    - Unzip the folder into the desired location
2. [Install Matplotlib](https://matplotlib.org/stable/install/index.html) via the command line if you haven't already
3. Run `python EvolutionSimulatorOfVision.py`