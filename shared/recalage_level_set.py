import numpy as np
import matplotlib.pyplot as plt
import random
import os
import dolfin as df
import mshr

df.parameters["ghost_mode"] = "shared_facet"
df.parameters["form_compiler"]["cpp_optimize"] = True
df.parameters["form_compiler"]["optimize"] = True
df.parameters["allow_extrapolation"] = True
df.parameters["form_compiler"]["representation"] = "uflacs"


def build_polygon_points(phi):
    mesh_macro = df.UnitSquareMesh(1024 - 1, 1024 - 1)
    V = df.FunctionSpace(mesh_macro, "CG", 1)
    phi_fenics = df.interpolate(phi, V)

    mesh_macro.init(0, 1)
    boundary_points = []
    for v in df.vertices(mesh_macro):
        if phi_fenics(v.point()) <= 1e-5 and phi_fenics(v.point()) >= -3.0e-5:
            boundary_points.append(np.array([v.point().x(), v.point().y()]))

    boundary_points = np.array(boundary_points)
    box_x_min, box_x_max = (
        np.min(boundary_points[:, 0]) - 10 * mesh_macro.hmax(),
        np.max(boundary_points[:, 0]) + 10 * mesh_macro.hmax(),
    )
    box_y_min, box_y_max = (
        np.min(boundary_points[:, 1]) - 10 * mesh_macro.hmax(),
        np.max(boundary_points[:, 1]) + 10 * mesh_macro.hmax(),
    )

    min_point = np.array([box_x_min, box_y_min])
    max_point = np.array([box_x_max, box_y_max])

    mesh_macro_2 = df.RectangleMesh(df.Point(min_point), df.Point(max_point), 512, 512)
    V = df.FunctionSpace(mesh_macro_2, "CG", 1)
    phi_fenics_2 = df.interpolate(phi, V)
    V_vector = df.VectorFunctionSpace(mesh_macro_2, "CG", 1, 2)
    grad_phi = df.grad(phi_fenics_2)
    grad_phi = df.project(grad_phi, V_vector)
    mesh_macro_2.init(0, 1)
    boundary_points = []
    for v in df.vertices(mesh_macro_2):
        if phi_fenics_2(v.point()) <= 1e-12 and phi_fenics_2(v.point()) >= -0.0005:
            boundary_points.append(np.array([v.point().x(), v.point().y()]))
            break
    boundary_points = np.array(boundary_points)
    ordered_points = []
    point = df.Point(boundary_points[0, 0], boundary_points[0, 1])
    point_x, point_y = boundary_points[0, 0], boundary_points[0, 1]
    N = 0
    while (
        np.absolute(phi_fenics_2(point)) > 1e-14
        and N < 25
        and np.absolute(grad_phi(point)[0] ** 2 + grad_phi(point)[1] ** 2) > 1e-7
    ):
        point_x = point.x() - phi_fenics_2(point) * grad_phi(point)[0] / np.absolute(
            grad_phi(point)[0] ** 2 + grad_phi(point)[1] ** 2
        )
        point_y = point.y() - phi_fenics_2(point) * grad_phi(point)[1] / np.absolute(
            grad_phi(point)[0] ** 2 + grad_phi(point)[1] ** 2
        )
        point = df.Point(point_x, point_y)
        N += 1
    ordered_points.append(np.array([point_x, point_y]))

    alpha = 0.001
    while (
        np.linalg.norm(ordered_points[0] - ordered_points[-1], ord=2) == 0.0
        or np.linalg.norm(ordered_points[0] - ordered_points[-1], ord=2) > alpha
        or len(ordered_points) < 10
    ):
        origin = ordered_points[-1]
        point = df.Point(origin[0], origin[1])

        grad_p = grad_phi(point)
        ortho = np.array([-grad_p[1], grad_p[0]])
        p_1 = origin + alpha * ortho / (grad_p[0] ** 2 + grad_p[1] ** 2)

        point = df.Point(p_1[0], p_1[1])
        point_x, point_y = p_1[0], p_1[1]
        N = 0
        while (
            np.absolute(phi_fenics_2(point)) > 1e-14
            and N < 25
            and np.absolute(grad_phi(point)[0] ** 2 + grad_phi(point)[1] ** 2) > 1e-7
        ):
            point_x = point.x() - phi_fenics_2(point) * grad_phi(point)[
                0
            ] / np.absolute(grad_phi(point)[0] ** 2 + grad_phi(point)[1] ** 2)
            point_y = point.y() - phi_fenics_2(point) * grad_phi(point)[
                1
            ] / np.absolute(grad_phi(point)[0] ** 2 + grad_phi(point)[1] ** 2)
            point = df.Point(point_x, point_y)
            N += 1
        if len(ordered_points) > 10:
            v1 = np.linalg.norm(ordered_points[0] - ordered_points[-1], ord=2)
            v2 = np.linalg.norm(
                ordered_points[-1] - np.array([point_x, point_y]), ord=2
            )
            v3 = np.linalg.norm(ordered_points[0] - np.array([point_x, point_y]), ord=2)
            if (v2 >= np.max(np.array([v1, v3]))) and (v1 < alpha or v3 < alpha):
                break

        ordered_points.append(np.array([point_x, point_y]))
    ordered_points = np.array(ordered_points)
    plt.figure()
    plt.plot(ordered_points[:, 0], ordered_points[:, 1], "-+", color="blue")
    plt.plot(ordered_points[0, 0], ordered_points[0, 1], "+", color="red")
    plt.plot(ordered_points[1, 0], ordered_points[1, 1], "+", color="red")
    plt.plot(ordered_points[2, 0], ordered_points[2, 1], "+", color="red")
    plt.plot(ordered_points[3, 0], ordered_points[3, 1], "+", color="red")
    plt.plot(ordered_points[-2, 0], ordered_points[-2, 1], "*", color="purple")
    plt.plot(ordered_points[-1, 0], ordered_points[-1, 1], "*", color="purple")
    plt.show()
    return ordered_points


def build_mesh_from_polygon(ordered_points, size, plot_mesh=False):
    fenics_points = [
        df.Point(ordered_points[i, 0], ordered_points[i, 1])
        for i in range(0, len(ordered_points), 1)
    ]
    fenics_points += [fenics_points[0]]
    if size == None:
        min_size = 0.001
        H_mesh = 600

    else:
        min_size = size  # df.UnitSquareMesh(size - 1, size - 1).hmax()
        H_mesh = 20
        alpha = int(min_size / 0.0006)
        fenics_points = [fenics_points[i] for i in range(0, len(fenics_points), alpha)]

    try:
        polygon = mshr.Polygon(fenics_points)
    except:
        polygon = mshr.Polygon(fenics_points[::-1])

    fenics_points = []  # free memory
    mesh = mshr.generate_mesh(polygon, H_mesh)
    while mesh.hmax() > min_size:
        if size == None:
            H_mesh += 20
        else:
            H_mesh += 2
        mesh = mshr.generate_mesh(polygon, H_mesh)
        print(f"{H_mesh =}    {mesh.hmax() =}")
    if plot_mesh:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(ordered_points[:, 0], ordered_points[:, 1], "-+")
        plt.subplot(1, 2, 2)
        df.plot(mesh)
        plt.tight_layout()
        plt.show()
    return mesh


def convert_numpy_matrix_to_fenics(X, nb_vert, degree=2):
    """Function to convert a matrix to a FEniCS function, for degree = 1, 2

    Args:
        X (array): input array, of size nb_dof_x x nb_dof_y
        nb_vert (int): number of vertices in each direction

    Returns:
        X_FEniCS (function FEniCS): output function that can be used with FEniCS
    """
    boxmesh = df.UnitSquareMesh(nb_vert - 1, nb_vert - 1)
    V = df.FunctionSpace(boxmesh, "CG", degree)
    coords = V.tabulate_dof_coordinates()
    coords = coords.T
    new_matrix = np.zeros(
        (
            np.shape(V.tabulate_dof_coordinates())[0],
            np.shape(V.tabulate_dof_coordinates())[1] + 1,
        )
    )

    new_matrix[:, 0] = np.arange(0, np.shape(V.tabulate_dof_coordinates())[0])
    new_matrix[:, 1] = coords[0]
    new_matrix[:, 2] = coords[1]
    sorted_mat = np.array(sorted(new_matrix, key=lambda k: [k[2], k[1]]))
    mapping = sorted_mat[:, 0]
    X_FEniCS = df.Function(V)
    X_FEniCS.vector()[mapping] = X.flatten()

    return X_FEniCS
