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
        num_pb = dict["Problem"]
        if dict["Geometry"] == "Circle":
            if num_pb == 1:
                self.class_Problem = Circle_Solution1
            elif num_pb == 2:
                self.class_Problem = Circle_Solution2
            else:
                raise ValueError("Problem not recognized")
        elif dict["Geometry"] == "Square":
            if num_pb == 1:
                self.class_Problem = Square_Solution1
            else:
                raise ValueError("Problem not recognized")
        elif dict["Geometry"] == "Random_domain":
            if num_pb == 1:
                self.class_Problem = Random_domain_Solution1
            else:
                raise ValueError("Problem not recognized")
        else:
            raise ValueError("Geometry not recognized")
        
        # Correction
        self.corr_type = dict["Correction"]
        assert self.corr_type in ["add","mult"]

        self.Problem = self.class_Problem()
        self.PDE = self.class_PDE(self.Problem, sampling_on=self.sampling_on, impose_exact_bc=self.impose_exact_bc)
        
        subdir_name = dict["Geometry"]+"_Solution"+str(dict["Problem"])
        self.dir_name = "networks/"+dict["Boundary_condition"]+"/"+dict["Class_PDE"]+"/"+subdir_name+"/"+dict["Sampling_on"]+"_training/"

        self.corr_dir_name = self.dir_name+"corrections/"+self.corr_type+"/"
