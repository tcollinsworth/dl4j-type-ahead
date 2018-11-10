# RNN type-ahead using dl4j


43 inputs -L0-> 200 hidden -L1-> 200 hidden -L2-> 200 hidden -L3-> 43 outputs
Number of parameters in layer 0: 195800
Number of parameters in layer 1: 321400
Number of parameters in layer 2: 321400
Number of parameters in layer 3: 8643
Total number of network parameters: 847243
features: 43, classifications: 43

System.out.println(timeMs + " ms");
~200 ms on 4 core/hyperthreaded not using GPU, no TBTT, ~2 ms / char