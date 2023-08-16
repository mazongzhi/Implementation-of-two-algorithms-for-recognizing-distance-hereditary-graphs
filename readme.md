## Environment Setup

### Primary Dependencies

1. **Python**: Ensure that Python is installed on your system. Given the complexity of your code, it is recommended to use Python 3.6 or above.

2. **networkx**: Used for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
   - Installation: `pip install networkx`

3. **matplotlib**: Provides a way to visually draw graphics and charts.
   - Installation: `pip install matplotlib`

4. **Other standard libraries**:
   - `random`: For generating random numbers.
   - `time`: For measuring runtime.
   - `math`: Provides mathematical functions and constants.
   - `itertools`: Provides iterator functions for efficient looping.
   - `collections`: Contains specialized container datatypes for creating collections.

### Optional Dependencies

1. **memory_profiler**: Used to monitor memory usage of Python scripts. Although not necessary, if you wish to perform memory usage analysis, you can opt to install.
   - Installation: `pip install memory-profiler`

   Usage: Prefix the function you wish to measure with the `@profile` decorator. For example:

   ```python
   @profile
   def my_function():
       pass
   ```

### Setup Steps

1. Ensure you have Python installed.
2. Install the required third-party libraries using pip: `pip install networkx matplotlib`
3. For the optional memory analysis functionality, install: `pip install memory-profiler`
4. Place your code files along with `methond1.py`, `method2.py`, and `generate_random_distance_hereditary_graph.py` in the same directory.


Both algorithms currently require modifications to the original code to change the input graph, which might be a bit inconvenient. Both `method1` and `method2` use `test_G` as their input graph, where `test_G` is defined by the networkX library's graph structure. You can refer to the networkX documentation to modify and create the desired graph.

The `generate_graph.py` can be used to produce random distance-hereditary graphs, generate random non-distance-hereditary graphs, and create some random graphs. These functions are already defined within, and there are also some functions comparing the time complexities of the two algorithms. Users can execute them as needed.

The two algorithms mainly reference the paper "A simple paradigm for graph recognition: application to cographs and distance hereditary graphs." written by Guillaume Damiand, Michel Habib, and Christophe Paul, as well as the paper "New Approach to Distance-Hereditary Graphs" authored by Shin-ichi Nakano, Ryuhei Uehara, and Takeaki Uno.
