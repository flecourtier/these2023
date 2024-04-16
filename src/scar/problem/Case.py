from scar.equations import Poisson2D
from scar.geometry import Geometry2D, SDFunction
from scar.utils import read_config,get_class
from scar.problem import Problem
from scimba.equations import domain

# geometry_set = get_class_name(Geometry)
# sdf_set = get_class_name(SDFunction)
# problem_set = get_class_name(Problem)

class Case:
    def __init__(self,case_file="case.json"):
        self.case_file = case_file

        # read config file
        dict_config = read_config(self.case_file)
        
        geom_class_name = dict_config["geometry"]        
        sdf_class = dict_config["sdf"]
        problem_class_name = dict_config["problem"]
        pde_class_name = dict_config["pde"]
        assert pde_class_name in "Poisson2D"

        # SDF Learned
        if isinstance(sdf_class,dict):
            sdf_class_name = sdf_class["type"]
            assert sdf_class_name in {"SDEikonal","SDEikonalReg","SDEikonalLap"}
            assert problem_class_name == "ConstantForce"
            self.form_config = sdf_class["config"]
        # SDF Analytic
        else:     
            if "SDMVP" in sdf_class:       
                if sdf_class == "SDMVP":
                    p = 1
                else:
                    p = int(sdf_class.split("SDMVP")[1])
                sdf_class_name = "SDMVP"
            else:
                sdf_class_name = sdf_class

        geom_class = get_class(geom_class_name,Geometry2D)
        sdf_class = get_class(sdf_class_name,SDFunction)
        problem_class = get_class(problem_class_name,Problem)
        pde_class = get_class(pde_class_name,Poisson2D)

        threshold = dict_config["threshold"]

        self.form = geom_class()
        if sdf_class_name in {"SDEikonal","SDEikonalReg","SDEikonalLap"}:
            self.sd_function = sdf_class(self.form,self.form_config,threshold=threshold)
            form_config = "form_"+str(self.form_config)+"/"
        elif sdf_class_name == "SDMVP":
            if p != 1:  
                sdf_class_name = "SDMVP"+str(p)
            self.sd_function = SDFunction.SDMVP(self.form,p=p,threshold=threshold)
            form_config = ""
        else:
            self.sd_function = sdf_class(self.form,threshold=threshold)
            form_config = ""
        self.problem = problem_class(self.form)
        self.bound_box = self.sd_function.bound_box
        self.xdomain = domain.SpaceDomain(2,domain.SignedDistanceBasedDomain(2, self.bound_box, self.sd_function))
        self.pde = pde_class(self.xdomain, self.problem)

        self.dir_name = "networks/"+pde_class_name+"/"+geom_class_name+"/"+sdf_class_name+"/"+problem_class_name+"/"+form_config+str(threshold)+"/"

        # corr
        self.corr_type = dict_config["correction"]
        assert self.corr_type in ["add","add_IPP"]

        self.corr_dir_name = self.dir_name+"corrections/"+self.corr_type+"/"
