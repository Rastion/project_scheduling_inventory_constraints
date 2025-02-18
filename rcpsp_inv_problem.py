from qubots.base_problem import BaseProblem
import random, os

class RCPSPInventoryProblem(BaseProblem):
    """
    Project Scheduling Problem with Inventory Constraints.

    Instance file format (Patterson format):
      - First line: three integers:
          • number of tasks (nb_tasks),
          • number of renewable resources (nb_resources),
          • number of inventory resources (nb_inventories).
      - Second line: first nb_resources integers are the maximum capacities 
          for the renewable resources, followed by nb_inventories integers 
          representing the initial levels for the inventory resources.
      - Then, for each task (nb_tasks lines), the following data is provided:
          • Duration of the task (integer)
          • Renewable resource requirements (weights) for each resource 
            (nb_resources integers)
          • Inventory resource consumption at start for each inventory 
            resource (nb_inventories integers)
          • Inventory resource production at end for each inventory 
            resource (nb_inventories integers)
          • Number of successors (integer)
          • List of successor task IDs (1-indexed, which will be converted to 0-indexed)

    A candidate solution is represented as a list of start times (integers) for each task.

    Objective:
      - Minimize the makespan (i.e. the maximum finish time over all tasks) 
        while satisfying:
          • Precedence constraints: each task must finish before its successors start.
          • Renewable resource constraints: at every time unit t, the total resource usage 
            (from tasks active at t) must not exceed the resource capacity.
          • Inventory constraints: at every time t, for each inventory resource, the initial level 
            plus the sum of productions from tasks that have finished minus the sum of consumptions 
            from tasks that have started must be non-negative.

    Infeasible solutions incur heavy penalties.
    """
    
    def __init__(self, instance_file):
        self.instance_file = instance_file
        self._load_instance(instance_file)
        self.penalty_multiplier = 1000000  # A large number to penalize constraint violations

    def _load_instance(self, filename):

        # Resolve relative path with respect to this module’s directory.
        if not os.path.isabs(filename):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(base_dir, filename)

        with open(filename, "r") as f:
            lines = f.readlines()
        
        # First line: nb_tasks, nb_resources, nb_inventories
        tokens = lines[0].split()
        self.nb_tasks = int(tokens[0])
        self.nb_resources = int(tokens[1])
        self.nb_inventories = int(tokens[2])
        
        # Second line: renewable capacities followed by initial inventory levels
        tokens = lines[1].split()
        self.capacity = [int(tokens[r]) for r in range(self.nb_resources)]
        self.init_level = [int(tokens[r + self.nb_resources]) for r in range(self.nb_inventories)]
        
        # Initialize lists for task data
        self.duration = [0] * self.nb_tasks
        self.weight = [[] for _ in range(self.nb_tasks)]
        self.start_cons = [[] for _ in range(self.nb_tasks)]
        self.end_prod = [[] for _ in range(self.nb_tasks)]
        self.nb_successors = [0] * self.nb_tasks
        self.successors = [[] for _ in range(self.nb_tasks)]
        
        # For each task (starting at line 3)
        for i in range(self.nb_tasks):
            tokens = lines[i + 2].split()
            self.duration[i] = int(tokens[0])
            # Renewable resource requirements (weights)
            self.weight[i] = [int(tokens[r + 1]) for r in range(self.nb_resources)]
            # Inventory consumption at task start (for each inventory resource)
            self.start_cons[i] = [int(tokens[2 * r + self.nb_resources + 1]) for r in range(self.nb_inventories)]
            # Inventory production at task end (for each inventory resource)
            self.end_prod[i] = [int(tokens[2 * r + self.nb_resources + 2]) for r in range(self.nb_inventories)]
            # Number of successors
            idx = 2 * self.nb_inventories + self.nb_resources + 1
            self.nb_successors[i] = int(tokens[idx])
            # Successor task IDs (convert from 1-indexed to 0-indexed)
            self.successors[i] = [int(tokens[idx + 1 + s]) - 1 for s in range(self.nb_successors[i])]
        
        # Trivial horizon: the sum of durations of all tasks
        self.horizon = sum(self.duration)

    def evaluate_solution(self, solution) -> float:
        """
        Evaluates a candidate solution.

        Parameters:
          solution: A list of start times (one per task).

        Returns:
          The makespan plus a heavy penalty for any constraint violations.
        """
        if len(solution) != self.nb_tasks:
            raise ValueError("Solution must have a start time for each task.")
        
        penalty = 0
        
        # Precedence constraints: each task must finish before its successors start.
        for i in range(self.nb_tasks):
            finish_i = solution[i] + self.duration[i]
            for succ in self.successors[i]:
                if solution[succ] < finish_i:
                    penalty += (finish_i - solution[succ])
        
        # Determine the makespan as the maximum finish time.
        makespan = max(solution[i] + self.duration[i] for i in range(self.nb_tasks))
        
        # Renewable resource constraints: for each time unit t and each renewable resource.
        for t in range(makespan):
            for r in range(self.nb_resources):
                usage = 0
                for i in range(self.nb_tasks):
                    # Task i is active at time t if start <= t < start+duration.
                    if solution[i] <= t < solution[i] + self.duration[i]:
                        usage += self.weight[i][r]
                if usage > self.capacity[r]:
                    penalty += (usage - self.capacity[r])
        
        # Inventory constraints: for each time unit t and for each inventory resource.
        for t in range(makespan):
            for r in range(self.nb_inventories):
                # Inventory level at time t: initial level + productions from finished tasks - consumptions from started tasks.
                inventory = self.init_level[r]
                for i in range(self.nb_tasks):
                    if solution[i] + self.duration[i] <= t:
                        inventory += self.end_prod[i][r]
                    if solution[i] <= t:
                        inventory -= self.start_cons[i][r]
                if inventory < 0:
                    penalty += abs(inventory)
        
        cost = makespan + self.penalty_multiplier * penalty
        return cost

    def random_solution(self):
        """
        Generates a random candidate solution: a list of start times (between 0 and the horizon)
        for each task.
        """
        return [random.randint(0, self.horizon) for _ in range(self.nb_tasks)]
