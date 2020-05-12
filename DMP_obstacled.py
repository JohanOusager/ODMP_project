from dmp_position import PositionDMP
import numpy as np
from scipy.spatial.transform import Rotation as rot

class ObstacleDMP(PositionDMP):

    def __init__(self, n_bfs=10, alpha=48.0, beta=None, cs_alpha=None, cs=None):
        super().__init__(n_bfs, alpha, beta, cs_alpha, cs)


    def step(self, x, dt, tau, obstacles=None):
        def fp(xj):
            psi = np.exp(-self.h * (xj - self.c) ** 2)
            return self.Dp.dot(self.w.dot(psi) / psi.sum() * xj)

        # DMP system acceleration
        def avoidance(obs, gamma=1000, beta=20/np.pi): #beta 20/pi is default
            if all(np.isclose(self.dp, 0)):
                return 0

            #phi is the rotation angle from dp to (p - obs)
            phi = np.arccos(np.transpose(obs-self.p) * self.dp/(np.linalg.norm(obs-self.p)*np.linalg.norm(self.dp)))

            #dphi determines the magnitude of the avoidance force
            #dphi = c_1 * phi * exp(-c_2 * abs(phi)
            dphi = gamma * phi * np.exp(-beta * np.linalg.norm(phi))

            #R determines the direction of the force
            #R: axis = crossproduct((obs - p),  dp), rotation = pi/2
            R = rot.from_rotvec(np.pi/2 * np.cross((obs - self.p), self.dp))

            #ddp_obs is the resulting oriented force
            #ddp_obs = R*dp*dphi
            ddp_obs = R.apply(self.dp * dphi)

            return ddp_obs

        self.ddp = (self.alpha * (self.beta * (self.gp - self.p) - tau * self.dp) + fp(x)) / tau

        self.avoidance_term = [0, 0, 0]
        if obstacles:
            for obs in obstacles:
                # TODO: bigger gamma for bigger objects???
                # TODO: moving objects
                # TODO: tune gamma value
                self.avoidance_term += avoidance(obs, gamma=50000, beta=10/np.pi) / tau

        self.ddp += self.avoidance_term

        # Integrate acceleration to obtain velocity
        self.dp += self.ddp * dt

        # Integrate velocity to obtain position
        self.p += self.dp * dt

        if obstacles:
            return self.p, self.dp, self.ddp, self.avoidance_term, obstacles[0]-self.p, np.cross(obstacles[0] - self.p,  self.dp)
        else:
            return self.p, self.dp, self.ddp


    def rollout(self, ts, tau, obstacles=None):
        self.reset()

        if np.isscalar(tau):
            tau = np.full_like(ts, tau)

        x = self.cs.rollout(ts, tau)  # Integrate canonical system
        dt = np.gradient(ts) # Differential time vector

        n_steps = len(ts)
        p = np.empty((n_steps, 3))
        dp = np.empty((n_steps, 3))
        ddp = np.empty((n_steps, 3))
        av = np.empty((n_steps, 3))
        obsv = np.empty((n_steps, 3))
        ax = np.empty((n_steps, 3))

        if obstacles:
            for i in range(n_steps):
                p[i], dp[i], ddp[i], av[i], obsv[i], ax[i] = self.step(x[i], dt[i], tau[i], obstacles)
            return p, dp, ddp, av, obsv, ax

        else:
            for i in range(n_steps):
                p[i], dp[i], ddp[i] = self.step(x[i], dt[i], tau[i])
            return p, dp, ddp

