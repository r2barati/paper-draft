# Digital Twin Environment (`environment.py`)

The environment abstracts supply chain physics into a modular, highly configurable system fully compatible with the Gymnasium API. This permits the training of Reinforcement Learning agents or the testing of Operational Research models in a standardized sandbox.

## Core Modules

### 1. `SupplyChainNetwork` (Topology Mapping)
This class maps out the physical directed graph of the supply chain. It creates nodes indicating echelons (e.g., Factory -> Distributor -> Wholesaler -> Retailer) and links depicting transit routes.
* **Configurations**: Can easily toggle between a simple `serial` line and a complex `base` topology linking multiple distinct branches together. Handles adjacency mapping automatically to feed bounds to the RL action spaces.

### 2. `DemandEngine` (Customer Behaviors)
The engine that generates non-stationary sequences to simulate dynamic, unpredictable markets.
* **Profiles Supported**:
    * `stationary`: Normal distribution around a fixed mean.
    * `trend`: Mean slowly drifts upward or downward over time.
    * `seasonal`: Generates cyclical sinusodial waves representing holiday or seasonal fluctuations.
    * `shock`: An abrupt, massive spike or drop simulating a black-swan event.
* **Endogenous Goodwill**: If `goodwill=True`, the environment tracks "Customer Patience". Missed fulfillments degrade goodwill, which feeds back into the DemandEngine to structurally lower future baseline demand—simulating customers permanently leaving for competitors.

### 3. `NetInvMgmtMasterEnv_New` (The Gym Environment)
The main orchestrator uniting topologies and demand into a step-by-step observable state-space simulation.

* **State / Observation Space**: Returns multidimensional NumPy arrays providing local inventory levels, pipeline orders (units in transit), current and historical demand, and node-specific unit limits.
* **Action Space**: Accepts multidimensional arrays dictating exactly how many units each node should order from its respective upstream supplier. Accounts for holding capacities and maximum possible delivery speeds.
* **Fulfillment Mechanics**: 
    * `backlog=True`: Unmet orders are queued and must be paid off in future periods (accumulating penalty costs over time).
    * `backlog=False`: Unmet orders are immediately lost (acting as lost sales, incurring a one-time penalty limit).
* **Cost Function**: Evaluates an intricate profit function = *(Revenue from met demand) - (Holding costs of idle inventory) - (Ordering costs) - (Penalty costs for stockouts)*. Returns the current operational profit as the standard Reinforcement Learning `reward` signal.
