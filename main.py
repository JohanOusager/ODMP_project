from __future__ import division, print_function
from dmp_position import PositionDMP
from DMP_obstacled import ObstacleDMP
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


if __name__ == '__main__':
    # Load a demonstration file containing robot positions.
    demo = np.loadtxt("demo.dat", delimiter=" ", skiprows=1)

    tau = 0.002 * len(demo)
    t = np.arange(0, tau, 0.002)
    demo_p = demo[:, 0:3]   #position part

    N = 50  # TODO: Try changing the number of basis functions to see how it affects the output.

    ###DMP
    dmp = PositionDMP(n_bfs=N, alpha=48.0)
    dmp.train(demo_p, t, tau)

    ###train regular DMP (done early so we can place obstacles along its path)
    dmp_p, dmp_dp, dmp_ddp = dmp.rollout(t, tau)

    ###ADDING SOME OBSTACLES
    obstacles = []
   # for p in range(500, 1751, 200):
   #     obstacles.append(demo_p[p])

    #for p in range(600, 1751, 150):
    #    obstacles.append(dmp_p[p])


    ###HARD CODED GOOD OBSTACLES
    #obstacles.append([0.53623471, 0.38135131, 0.43055046])
    #obstacles.append([0.52626427, 0.41151957, 0.43269037])
    #obstacles.append([0.70712357, 0.16850636, 0.29812683])
    obstacles.append([0.63440407, 0.19807578, 0.43654489])
    #obstacles.append([0.61440407, 0.19807578, 0.43654489])


    #DMP WITH OBSTACLE AVOIDANCE
    odmp = ObstacleDMP(n_bfs=N, alpha=48.0, obstacles=obstacles)
    odmp.train(demo_p, t, tau)


    # TODO: different starting point
    #dmp.p0 = [0.7, 0.2, 0.6]
    #odmp.p0 = dmp.p0

    # TODO: different goal
    #dmp.gp = [0.3, 0.5, 0.2]
    #odmp.gp = dmp.gp

    # TODO: different time constant:
    #tau = tau * 0.5


    # Train DMP with obstacle avoidance
    odmp_p, odmp_dp, odmp_ddp, av = odmp.rollout(t, tau)


    # 2D plot the ODMP and DMP against the original demonstration
    fig1, axs = plt.subplots(3, 1, sharex=True)
    axs[0].plot(t, demo_p[:, 0], label='Demonstration')
    axs[0].plot(t, dmp_p[:, 0], label='DMP')
    axs[0].plot(t, odmp_p[:, 0], label='ODMP')
    axs[0].set_xlabel('t (s)')
    axs[0].set_ylabel('X (m)')

    axs[1].plot(t, demo_p[:, 1], label='Demonstration')
    axs[1].plot(t, dmp_p[:, 1], label='DMP')
    axs[1].plot(t, odmp_p[:, 1], label='ODMP')
    axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('Y (m)')

    axs[2].plot(t, demo_p[:, 2], label='Demonstration')
    axs[2].plot(t, dmp_p[:, 2], label='DMP')
    axs[2].plot(t, odmp_p[:, 2], label='ODMP')
    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('Z (m)')
    axs[2].legend()
    plt.suptitle("Positions")

    # 3D plot the ODMP and DMP against the original demonstration
    fig2 = plt.figure(2, figsize=(16, 10))
    ax = plt.axes(projection='3d')
    ax.plot3D(demo_p[:, 0], demo_p[:, 1], demo_p[:, 2], label='Demonstration')
    ax.plot3D(dmp_p[:, 0], dmp_p[:, 1], dmp_p[:, 2], label='DMP')
    ax.plot3D(odmp_p[:, 0], odmp_p[:, 1], odmp_p[:, 2], label='ODMP')
    for obs in obstacles:
        ax.plot([obs[0]], [obs[1]], [obs[2]], markerfacecolor='k', markeredgecolor='k', marker='*', markersize=5, alpha=0.6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.suptitle("Positions")


    # 2D plot of the avoidance acceleration
    fig3, axs = plt.subplots(3, 1)

    axs[0].plot(t, av[:, 0])
    axs[0].set_xlabel('t (s)')
    axs[0].set_ylabel('X (m/s^2)')

    axs[1].plot(t, av[:, 1])
    axs[1].set_xlabel('t (s)')
    axs[1].set_ylabel('Y (m/s^2)')

    axs[2].plot(t, av[:, 2])
    axs[2].set_xlabel('t (s)')
    axs[2].set_ylabel('Z (m/s^2)')

    plt.suptitle("Avoidance acceleration")


    #plot 3D paths with illustrative avoidance acceleration vectors
    fig4 = plt.figure(4, figsize=(16, 10))
    ax = plt.axes(projection='3d')
    up_from = 0
    up_to = len(demo_p)
    av = av[up_from:up_to]
    odmp_p = odmp_p[up_from:up_to]
    dmp_p = dmp_p[up_from:up_to]
    demo_p = demo_p[up_from:up_to]
    ax.plot3D(demo_p[:, 0], demo_p[:, 1], demo_p[:, 2], label='Demonstration')
    ax.plot3D(dmp_p[:, 0], dmp_p[:, 1], dmp_p[:, 2], label='DMP')
    ax.plot3D(odmp_p[:, 0], odmp_p[:, 1], odmp_p[:, 2], label='ODMP')
    for obs in obstacles:
        ax.plot([obs[0]], [obs[1]], [obs[2]], markerfacecolor='k', markeredgecolor='k', marker='*', markersize=5, alpha=0.6)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    av = av/100
    av = av[::10]
    odmp_p = odmp_p[::10]
    ax.quiver(odmp_p[:,0], odmp_p[:,1], odmp_p[:,2], av[:,0], av[:,1], av[:,2], color="red")
    plt.suptitle("Positions and avoidance acceleration")

    plt.show()