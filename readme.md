Both algorithms currently require modifications to the original code to change the input graph, which might be a bit inconvenient. Both `method1` and `method2` use `test_G` as their input graph, where `test_G` is defined by the networkX library's graph structure. You can refer to the networkX documentation to modify and create the desired graph.

The `generate_graph.py` can be used to produce random distance-hereditary graphs, generate random non-distance-hereditary graphs, and create some random graphs. These functions are already defined within, and there are also some functions comparing the time complexities of the two algorithms. Users can execute them as needed.

The two algorithms mainly reference the paper "A simple paradigm for graph recognition: application to cographs and distance hereditary graphs." written by Guillaume Damiand, Michel Habib, and Christophe Paul, as well as the paper "New Approach to Distance-Hereditary Graphs" authored by Shin-ichi Nakano, Ryuhei Uehara, and Takeaki Uno.