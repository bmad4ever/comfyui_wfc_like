# comfyui_wfc_like
An "opinionated" Wave Function Collapse implementation with a set of nodes for comfyui

### Important Technicality 
This implementation is not a pure-to-form implementation of the wave collapse algorithm.

The “wave” of possibilities is not kept and updated on the entire grid; instead, only the boundary of the collapsed nodes is evaluated, expanding the boundary at each iteration, and validating only the states of cells adjacent to the expanded ones. In this sense, it would be fair to name it something else, since instead of a wave of possibilities the algorithm only satisfies local constraints until reaching an impossible state, at which point it backtracks to a previous valid state. Nevertheless, the wave-function-collapse captures, and helps clarify, the potential applications of this algorithm.

Additionally, in the spirit of being used as a visual tool, there is no way to specify global constraints, and all local constraints are inferred from the given samples. Although this makes some sets of rules hard to specify, the envisioned application is not to necessarily arrive at a complete solution, but rather a partial one, which can be completed using diffusion. 

This implementation searches for possible states using a best-first search which also considers the node’s depth to make the search greedy towards already deep paths as to speed up the generation towards a partially acceptable state, i.e. a state that hasn’t collapsed all the cells but should be somewhat complete, provided the constraints are not very intricate. 


TODO: Add documentation & sample workflows
