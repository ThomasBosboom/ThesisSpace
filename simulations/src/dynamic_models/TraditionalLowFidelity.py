import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from tudatpy.kernel import constants
from tudatpy.kernel.astro import time_conversion


class TraditionalLowFidelity:
    def __init__(self, G, m1, m2, a):

        self.G  = G
        self.m1 = m1
        self.m2 = m2
        self.a  = a

        self.mu    = self.m2/(self.m1+self.m2)
        self.lstar = self.a
        self.tstar = np.sqrt(self.lstar**3/(self.G*(self.m1+self.m2)))/constants.JULIAN_DAY
        self.tstar = 4.348377505921948
        self.rotation_rate = 1/self.tstar

        self.state_m1 = np.array([-self.mu, 0, 0, 0, 0, 0])
        self.state_m2 = np.array([(1-self.mu), 0, 0, 0, 0, 0])

        self.state_L1 = np.array([self.a*(1-(self.m2/(3*self.m1))**(1/3))/self.lstar, 0, 0, 0, 0, 0])
        self.state_L2 = np.array([self.a*(1+(self.m2/(3*self.m1))**(1/3))/self.lstar, 0, 0, 0, 0, 0])


    def get_equations_of_motion(self, state, t):

        x     = state[0]
        y     = state[1]
        z     = state[2]
        x_dot = state[3]
        y_dot = state[4]
        z_dot = state[5]

        x_ddot = x+2*y_dot-((1-self.mu)*(x+self.mu))/((x+self.mu)**2+y**2+z**2)**(3/2)-(self.mu*(x-(1-self.mu)))/((x-(1-self.mu))**2+y**2+z**2)**(3/2)
        y_ddot = y-2*x_dot-((1-self.mu)*y)/((x+self.mu)**2+y**2+z**2)**(3/2)-(self.mu*y)/((x-(1-self.mu))**2+y**2+z**2)**(3/2)
        z_ddot = -((1-self.mu)*z)/((x+self.mu)**2+y**2+z**2)**(3/2)-(self.mu*z)/((x-(1-self.mu))**2+y**2+z**2)**(3/2)

        return np.array([x_dot, y_dot, z_dot, x_ddot, y_ddot, z_ddot])


    def get_state_history(self, state_rotating_bary_0, start, stop, step):

        self.t = np.arange(start, stop/(self.tstar)+step/(self.tstar), step/(self.tstar))

        return self.t*self.tstar, odeint(self.get_equations_of_motion, state_rotating_bary_0, self.t)


    # def get_jacobi_constant_history(self, state_rotating_barycenter):

    #     X_rot     = np.array(state_rotating_barycenter[:, 0])
    #     Y_rot     = np.array(state_rotating_barycenter[:, 1])
    #     Z_rot     = np.array(state_rotating_barycenter[:, 2])
    #     X_dot_rot = np.array(state_rotating_barycenter[:, 3])
    #     Y_dot_rot = np.array(state_rotating_barycenter[:, 4])
    #     Z_dot_rot = np.array(state_rotating_barycenter[:, 5])

    #     return 2*((1-self.mu)/np.sqrt((X_rot+self.mu)**2+Y_rot**2+Z_rot**2)+self.mu/np.sqrt((X_rot+self.mu-1)**2+Y_rot**2+Z_rot**2))+X_rot**2+Y_rot**2-(X_dot_rot**2+Y_dot_rot**2+Z_dot_rot**2)


    # def convert_state_nondim_to_dim(self, state_nondim, state_primary_to_secondary):

    #     state_history_dim = np.empty((np.shape(state_nondim)[0],6))
    #     for epoch, state in enumerate(state_nondim):
    #         lstar = np.linalg.norm(state_primary_to_secondary[epoch,:3])
    #         tstar = np.sqrt(lstar**3/(self.G*(self.m1+self.m2)))
    #         state_history_dim[epoch] = np.array([lstar, lstar, lstar, lstar/tstar, lstar/tstar, lstar/tstar])*state

    #     return state_history_dim


    # def convert_state_dim_to_nondim(self, state_dim):

    #     return np.array([1/self.lstar, 1/self.lstar, 1/self.lstar, 1/(self.lstar/self.tstar), 1/(self.lstar/self.tstar), 1/(self.lstar/self.tstar)])*state_dim/1000


    # def convert_state_barycentric_to_body(self, state_barycentric, body, state_type="inertial"):

    #     if state_type == "inertial":
    #         if body == "primary":
    #             return state_barycentric - self.state_m1*state_barycentric
    #         if body == "secondary":
    #             return state_barycentric - self.state_m2*state_barycentric

    #     elif state_type == "rotating":
    #         state_body = state_barycentric
    #         if body == "primary":
    #             for epoch, state in enumerate(state_body):
    #                 state_body[epoch, 0] = state_body[epoch, 0] - (self.mu)
    #             return state_body
    #         if body == "secondary":
    #             for epoch, state in enumerate(state_body):
    #                 state_body[epoch, 0] = state_body[epoch, 0] - (1-self.mu)
    #             return state_body


    # def convert_state_body_to_barycentric(self, state_body, body, state_type="inertial"):

    #     if state_type == "inertial":
    #         if body == "primary":
    #             return state_body + self.state_m1*state_body
    #         if body == "secondary":
    #             return state_body + self.state_m2*state_body

    #     elif state_type == "rotating":
    #         state_barycentric = state_body
    #         if body == "primary":
    #             for epoch, state in enumerate(state_barycentric):
    #                 state_barycentric[epoch, 0] = state_barycentric[epoch, 0] + (self.mu)
    #             return state_barycentric
    #         if body == "secondary":
    #             for epoch, state in enumerate(state_barycentric):
    #                 state_barycentric[epoch, 0] = state_barycentric[epoch, 0] + (1-self.mu)
    #             return state_barycentric


    # def convert_state_rotating_to_inertial(self, state_rotating):

    #     state_inertial_barycenter = np.empty((np.shape(state_rotating)[0], np.shape(state_rotating)[1]))
    #     for i, time in enumerate(self.t):

    #         rotation_matrix = np.array([[np.cos(time), -np.sin(time), 0],
    #                                     [np.sin(time),  np.cos(time), 0],
    #                                     [0,             0,            1]])

    #         rotation_matrix_dot = np.array([[-np.sin(time), -np.cos(time), 0],
    #                                         [ np.cos(time), -np.sin(time), 0],
    #                                         [ 0,             0,            1]])

    #         transformation_matrix = np.block([[rotation_matrix,     np.zeros((3,3))],
    #                                           [rotation_matrix_dot, rotation_matrix]])

    #         state_inertial_barycenter[i] = np.dot(transformation_matrix, state_rotating[i,:])

    #     return state_inertial_barycenter


    # def convert_state_inertial_to_rotating(self, state_inertial):

    #     state_rotating  = np.empty((np.shape(state_inertial)[0], np.shape(state_inertial)[1]))
    #     for i, time in enumerate(self.t):

    #         rotation_matrix = np.array([[np.cos(time), -np.sin(time), 0],
    #                                     [np.sin(time),  np.cos(time), 0],
    #                                     [0,             0,            1]])

    #         rotation_matrix_dot = np.array([[-np.sin(time), -np.cos(time), 0],
    #                                         [ np.cos(time), -np.sin(time), 0],
    #                                         [ 0,             0,            1]])

    #         transformation_matrix = np.block([[rotation_matrix,     np.zeros((3,3))],
    #                                           [rotation_matrix_dot, rotation_matrix]])

    #         transformation_matrix_inverse = np.linalg.inv(transformation_matrix)

    #         state_rotating[i] = np.dot(transformation_matrix_inverse,state_inertial[i,:])

    #     return state_rotating


# G  = 6.67408E-11
# m1 = 5.97219E+24
# m2 = 7.34767E+22
# a  = 3.84747963e8
# state_rotating_bary_LUMIO_0 = [1.1473302, 0, -0.15142308, 0, -0.21994554, 0]
# state_rotating_bary_LPF_0   = [0.98512134, 0.00147649, 0.00492546, -0.87329730, -1.61190048, 0]
# start = 0
# stop = 28
# step = 0.005



# system   = CRTBP(G, m1, m2, a)
# state_rotating_bary_LUMIO = system.get_state_history(state_rotating_bary_LUMIO_0, start, stop, step)[1]
# state_rotating_bary_LPF   = system.get_state_history(state_rotating_bary_LPF_0, start, stop, step)[1]
# t = system.get_state_history(state_rotating_bary_LUMIO_0, start, stop, step)[0]

# print("Moon frame: ", state_rotating_bary_LUMIO_0[0]-(1-system.mu))

# state_rotating_primary = np.multiply(np.ones((np.shape(t)[0],6)), system.state_m1)
# state_rotating_secondary = np.multiply(np.ones((np.shape(t)[0],6)), system.state_m2)


# ### Rotating Frame Plot #########################################################
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_title('Trajectories in Barycentric Rotating Frame CRTBP')
# ax.set_xlabel('x [ND]')
# ax.set_ylabel('y [ND]')
# ax.set_zlabel('z [ND]')
# ax.plot3D(state_rotating_bary_LUMIO[:,0], state_rotating_bary_LUMIO[:,1], state_rotating_bary_LUMIO[:,2], c='red', label="LUMIO")
# ax.plot3D(state_rotating_bary_LPF[:,0], state_rotating_bary_LPF[:,1], state_rotating_bary_LPF[:,2], c='blue', label="LPF")
# ax.plot3D(system.state_m1[0], system.state_m1[1], system.state_m1[2], c='green', marker='o', label="Earth")
# ax.plot3D(system.state_m2[0], system.state_m2[1], system.state_m2[2], c='gray', marker='o', label="Moon")
# ax.plot3D(system.state_L2[0], 0, 0, c='black', marker='o', label="EML2")
# plt.legend()
# xyzlim = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
# ax.axes.set_xlim3d(left=min(xyzlim[0]), right=max(xyzlim[0]))
# ax.axes.set_ylim3d(bottom=min(xyzlim[1]), top=max(xyzlim[1]))
# ax.axes.set_zlim3d(bottom=min(xyzlim[2]), top=max(xyzlim[2]))
# plt.show()


# # ##### Body-fixed Frame Plots ###################################################

# ## Barycentric frame
# state_inertial_bary_LUMIO     = system.convert_state_rotating_to_inertial(state_rotating_bary_LUMIO)
# state_inertial_bary_LPF       = system.convert_state_rotating_to_inertial(state_rotating_bary_LPF,)
# state_inertial_primary   = system.convert_state_rotating_to_inertial(state_rotating_primary)
# state_inertial_secondary = system.convert_state_rotating_to_inertial(state_rotating_secondary)

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_title('Trajectories in Barycentric Inertial Frame CRTBP')
# ax.set_xlabel('X [ND]')
# ax.set_ylabel('Y [ND]')
# ax.set_zlabel('Z [ND]')
# ax.plot3D(state_inertial_bary_LUMIO[:,0], state_inertial_bary_LUMIO[:,1], state_inertial_bary_LUMIO[:,2], c='red', label="LUMIO")
# ax.plot3D(state_inertial_bary_LPF[:,0], state_inertial_bary_LPF[:,1], state_inertial_bary_LPF[:,2], c='blue', label="LPF")
# ax.plot3D(state_inertial_primary[:,0], state_inertial_primary[:,1], state_inertial_primary[:,2], c='green', marker='o', label="Earth")
# ax.plot3D(state_inertial_secondary[:,0], state_inertial_secondary[:,1], state_inertial_secondary[:,2], c='gray', marker='o', label="Moon")
# plt.legend()
# plt.axis('equal')
# plt.show()

# ## Earth-fixed frame
# state_inertial_primary_LUMIO = system.convert_state_barycentric_to_body(state_inertial_bary_LUMIO, "primary")
# state_inertial_primary_LPF = system.convert_state_barycentric_to_body(state_inertial_bary_LPF, "primary")

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_title('Trajectories in Earth Body-Fixed Frame CRTBP Orbit')
# ax.set_xlabel('X [ND]')
# ax.set_ylabel('Y [ND]')
# ax.set_zlabel('Z [ND]')
# ax.plot3D(state_inertial_primary_LUMIO[:,0], state_inertial_primary_LUMIO[:,1], state_inertial_primary_LUMIO[:,2], c='red', label="LUMIO")
# ax.plot3D(state_inertial_primary_LPF[:,0], state_inertial_primary_LPF[:,1], state_inertial_primary_LPF[:,2], c='blue', label="LPF")
# plt.legend()
# plt.axis('equal')
# # plt.show()

# ## Moon-fixed frame
# state_inertial_secondary_LUMIO = system.convert_state_barycentric_to_body(state_inertial_bary_LUMIO, "secondary")
# state_inertial_secondary_LPF = system.convert_state_barycentric_to_body(state_inertial_bary_LPF, "secondary")
# state_rotating_secondary_LUMIO = system.convert_state_barycentric_to_body(state_rotating_bary_LUMIO, "secondary", state_type="rotating")
# state_rotating_secondary_LPF = system.convert_state_barycentric_to_body(state_rotating_bary_LPF, "secondary", state_type="rotating")
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.set_title('Trajectories in Moon Body-Fixed Frame CRTBP Orbit')
# ax.set_xlabel('X [ND]')
# ax.set_ylabel('Y [ND]')
# ax.set_zlabel('Z [ND]')
# ax.plot3D(state_rotating_secondary_LUMIO[:,0], state_rotating_secondary_LUMIO[:,1], state_rotating_secondary_LUMIO[:,2], c='red', label="LUMIO")
# ax.plot3D(state_rotating_secondary_LPF[:,0], state_rotating_secondary_LPF[:,1], state_rotating_secondary_LPF[:,2], c='blue', label="LPF")
# plt.legend()
# plt.show()

# # Comparison on the models
# system.t = system.t[0:int(0.5*stop/step)]
# state_rotating_bary_LUMIO = state_rotating_bary_LUMIO[0:int(0.5*stop/step)]
# state_rotating_bary_erdem = validation_LUMIO.get_state_history_erdem()[0:int(0.5*stop/step)]

# system.t = np.array([0])
# print(np.shape(system.t))
# print(state_rotating_bary_LUMIO[-1], np.shape(state_rotating_bary_LUMIO))
# print(state_rotating_bary_erdem[-1], np.shape(state_rotating_bary_erdem))

# ax = plt.figure().add_subplot(projection='3d')
# plt.plot(state_rotating_bary_LUMIO[:,0], state_rotating_bary_LUMIO[:,1], state_rotating_bary_LUMIO[:,2], label="Orbit own simulation")
# plt.plot(state_rotating_bary_erdem[:,0], state_rotating_bary_erdem[:,1], state_rotating_bary_erdem[:,2], label="LUMIO Erdem")
# plt.title("LUMIO orbit in synodic frame")
# plt.axis('equal')
# ax.set_xlabel('x [m]')
# ax.set_ylabel('y [m]')
# ax.set_zlabel('z [m]')
# plt.legend()


# fig = plt.figure()
# plt.plot(np.linalg.norm(state_rotating_bary_LUMIO[:,:3]-state_rotating_secondary[0:int(0.5*stop/step), :3], axis=1)*system.a)
# plt.plot(np.linalg.norm(state_rotating_bary_erdem[:,:3]-state_rotating_secondary[0:int(0.5*stop/step), :3], axis=1)*system.a)
# plt.show()
