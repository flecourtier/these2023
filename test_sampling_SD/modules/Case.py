from modules.utils import read_config
from modules.Poisson2D import *
from modules.Problem import *

class Case:
    def __init__(self,case_file="case.json"):
        self.case_file = case_file

        # read config file
        dict = read_config(self.case_file)

        # Boundary condition
        if dict["Boundary_condition"] == "exact_bc":
            self.impose_exact_bc = True
        elif dict["Boundary_condition"] == "approach_bc":
            self.impose_exact_bc = False
        else:
            raise ValueError("Boundary condition not recognized")
        
        # PDE type
        if dict["Class_PDE"] == "SingleProblem":
            self.class_PDE = SingleProblem
        elif dict["Class_PDE"] == "VariedSolution_S":
            self.class_PDE = VariedSolution_S
        else:
            raise ValueError("Class_PDE not recognized")
        
        # Sampling_on
        if dict["Sampling_on"] == "Omega":
            self.sampling_on = "Omega"
        elif dict["Sampling_on"] == "O_cal":
            self.sampling_on = "O_cal"
        else:
            raise ValueError("Sampling_on not recognized")
        
        # Problem
        if dict["Class_Problem"] == "Circle":
            self.class_Problem = Circle
        elif dict["Class_Problem"] == "Square":
            self.class_Problem = Square
        else:
            raise ValueError("Class_Problem not recognized")

        self.Problem = self.class_Problem()
        self.PDE = self.class_PDE(self.Problem, sampling_on=self.sampling_on, impose_exact_bc=self.impose_exact_bc)
        
        self.dir_name = "networks/"+dict["Boundary_condition"]+"/"+dict["Class_PDE"]+"/"+dict["Class_Problem"]+"/"+dict["Sampling_on"]+"_training/"

