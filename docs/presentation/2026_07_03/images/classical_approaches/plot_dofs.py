import numpy as np
from dolfinx import mesh, plot, fem
from mpi4py import MPI
import pyvista

# 1. Création du maillage
domain = mesh.create_unit_square(MPI.COMM_WORLD, 18, 18, mesh.CellType.triangle)

# 2. Création de l'espace de fonction CG P1 (Continuous Galerkin, degré 1)
k = 1
V = fem.functionspace(domain, ("Lagrange", k))

# 3. Extraction des coordonnées des Degrés de Liberté (DOFs)
dof_coords = V.tabulate_dof_coordinates()

# 4. Préparation de la visualisation PyVista
plotter = pyvista.Plotter()

# --- Affichage du maillage (cellules blanches, bords noirs) ---
cells, types, x = plot.vtk_mesh(domain, domain.topology.dim)
grid = pyvista.UnstructuredGrid(cells, types, x)
plotter.add_mesh(grid, show_edges=True, color="white", edge_color="black")

# --- Affichage des DOFs (croix rouges) ---
# On crée un objet PolyData pour les points des DOFs
dof_points = pyvista.PolyData(dof_coords)
plotter.add_mesh(dof_points, color="red", point_size=15.0, render_points_as_spheres=False)

# Ajustement de la vue
plotter.view_xy()
# plotter.screenshot("mon_maillage.png") # Ne plantera plus
plotter.show()
