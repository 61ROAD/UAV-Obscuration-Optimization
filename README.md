# **🚁 UAV Swarm Anti-Missile Obscuration Optimization Engine**

## **📌 Project Overview**

In modern defensive operations, deploying Unmanned Aerial Vehicles (UAVs) to create smoke screens is a highly cost-effective countermeasure against guided missiles. However, precisely calculating the trajectory, release timing, and detonation delay in a dynamic 3D environment is a computationally complex challenge.

This project is a sophisticated **3D Kinematics Simulation and Trajectory Optimization Engine**. It mathematically models the physical behavior of missiles, UAVs, and expanding smoke clouds. By combining **Meta-Heuristic Optimization (Differential Evolution)** with **Mixed-Integer Linear Programming (MILP)**, the engine dynamically generates optimal flight paths and deployment strategies for a UAV swarm to maximize the continuous Line-of-Sight (LoS) obscuration time, successfully protecting ground targets.

## **🧠 Simulation Engine & 3D Physics Model**

To ensure real-world applicability, I built a custom 3D physics engine from scratch to simulate the battlefield environment:

* **Kinematics & Gravity:** Modeled the parabolic free-fall of unguided smoke grenades and the constant-velocity sinking of aerosol clouds.  
* **Line-of-Sight (LoS) Mathematics:** Engineered complex spatial geometry algorithms (Cone-Sphere intersections) using NumPy vectorization to compute whether the expanding smoke spheres successfully intersect the dynamic visual cone between the incoming missile and the ground target.  
* **Continuous Time Simulation:** Implemented discrete-time stepping evaluation (DT \= 0.05s) to accurately measure the total effective obscuration time across multiple moving entities.

## **🚀 Optimization Architecture**

The optimization pipeline scales from single-agent continuous parameter tuning to multi-agent combinatorial optimization:

### **1\. High-Dimensional Continuous Optimization (Differential Evolution)**

* **Challenge:** Optimizing UAV flight speed, heading angle, release timing, and detonation delay creates a highly non-linear, non-convex reward landscape.  
* **Solution:** Utilized SciPy's differential\_evolution algorithm. It iteratively mutates and recombines deployment strategies, successfully escaping local optima to find configurations that increased shelter time from a baseline of 1.39s to over 6.10s for single-UAV multi-drop scenarios.

### **2\. Multi-Agent Swarm Allocation (MILP & Column Generation)**

* **Challenge:** Coordinating multiple UAVs against multiple incoming missiles requires solving a massive combinatorial assignment problem preventing trajectory overlap and maximizing global cover.  
* **Solution:** Framed the swarm coordination as a **Set Covering Problem**.  
* **Tech:** Engineered a Column Generation approach and solved the Mixed-Integer Linear Program (MILP) using the PuLP library. The system selects the absolute best combination of pre-calculated flight paths to guarantee global optimal obscuration for all targets simultaneously.

## **🛠️ Tech Stack**

* **Language:** Python  
* **Vectorized Math & Physics:** NumPy  
* **Heuristic Solvers:** SciPy (scipy.optimize.differential\_evolution)  
* **Linear Programming Solver:** PuLP  
* **Data Persistence:** Pandas, Pickle (for storing pre-computed path columns)

## **📁 Project Structure**

.  
├── utils.py                 \# Core 3D geometry engine (Cone/Sphere intersection algorithms)  
├── get\_pos.py               \# Kinematics engine (Missile & Smoke trajectory calculators)  
├── 1.py \- 4.py              \# Iterative Differential Evolution solvers (from simple to complex scenarios)  
├── 5\_get\_paths.py           \# Path generator building the candidate pool for the swarm  
├── 5\_merge.py               \# Data merging utility for generated trajectories  
├── 5\_final\_solver.py        \# MILP / Column Generation solver using PuLP for multi-UAV assignment  
└── README.md                \# Project documentation

## **⚙️ Installation & Usage**

1. **Clone the repository:**  
   git clone \[https://github.com/yourusername/UAV-Obscuration-Optimization.git\](https://github.com/yourusername/UAV-Obscuration-Optimization.git)  
   cd UAV-Obscuration-Optimization

2. **Install dependencies:**  
   pip install numpy scipy pandas pulp tqdm

3. **Run the Physics Simulation & Heuristic Optimization:**  
   *Execute 2.py or 3.py to watch the Differential Evolution algorithm optimize a single UAV's trajectory in real-time.*  
   python 3.py

4. **Run the Multi-Agent Swarm Solver:**  
   *First generate the path columns, then run the final MILP solver.*  
   python 5\_get\_paths.py  
   python 5\_final\_solver.py

## 