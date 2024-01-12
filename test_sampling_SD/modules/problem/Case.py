from modules.utils import read_config
from modules.problem import Poisson2D,Geometry,Problem,SDFunction
from scimba.equations import domain
import inspect

# get all the class name in the module (not abstract class)
def get_class_name(module):
    class_name = []
    for name, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and not inspect.isabstract(obj):
            class_name.append(name)
    return class_name
    
# get the class by its name
def get_class(name,module):
    try:
        # Récupérer la classe par son nom
        class_ = getattr(module, name)
        return class_
    except AttributeError:
        # Gestion de l'erreur si la classe n'est pas trouvée
        print(f"Erreur : Classe {name} non trouvée dans le module {module.__name__}.")
    except Exception as e:
        # Gestion d'autres exceptions
        print(f"Une erreur s'est produite : {e}")

# geometry_set = get_class_name(Geometry)
# sdf_set = get_class_name(SDFunction)
# problem_set = get_class_name(Problem)

class Case:
    def __init__(self,case_file="case.json"):
        self.case_file = case_file

        # read config file
        dict = read_config(self.case_file)
        
        geom_class_name = dict["geometry"]        
        sdf_class_name = dict["sdf"]
        problem_class_name = dict["problem"]
        pde_class_name = dict["pde"]
        assert pde_class_name == "Poisson2D"

        if sdf_class_name != "SDCircle":
            assert geom_class_name == "Circle"

        if geom_class_name != "Circle":
            assert problem_class_name == "UnknownSolForMVP"

        geom_class = get_class(geom_class_name,Geometry)
        sdf_class = get_class(sdf_class_name,SDFunction)
        problem_class = get_class(problem_class_name,Problem)
        pde_class = get_class(pde_class_name,Poisson2D)

        threshold = dict["threshold"]

        self.form = geom_class()
        self.sd_function = sdf_class(self.form,threshold=threshold)
        self.problem = problem_class(self.form)
        bound_box = self.sd_function.bound_box
        self.xdomain = domain.SignedDistanceBasedDomain(2, bound_box, self.sd_function)
        self.pde = pde_class(self.xdomain, self.problem)

        self.dir_name = "networks/"+pde_class_name+"/"+geom_class_name+"/"+sdf_class_name+"/"+problem_class_name+"/"+str(threshold)+"/"
        
        # corr
        self.corr_type = dict["correction"]
        assert self.corr_type in ["add","mult"]

        self.corr_dir_name = self.dir_name+"corrections/"+self.corr_type+"/"
