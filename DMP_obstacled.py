from dmp_position import PositionDMP
import numpy as np
from scipy.spatial.transform import Rotation as rot

class ObstacleDMP(PositionDMP):

    def __init__(self, n_bfs=10, alpha=48.0, beta=None, cs_alpha=None, cs=None, obstacles=None):
        super().__init__(n_bfs, alpha, beta, cs_alpha, cs)

        if obstacles is not None:
            self.obstacles = obstacles
        else:
            self.obstacles = []

    def step(self, x, dt, tau):
        def fp(xj):
            psi = np.exp(-self.h * (xj - self.c) ** 2)
            return self.Dp.dot(self.w.dot(psi) / psi.sum() * xj)

        # DMP system acceleration
        # TODO: add the obstacle term
        def avoidance(obs, gamma=1000, beta=20/np.pi): #beta 20/pi is default
            if all(self.dp == 0):
                return 0

            #phi is the rotation angle from dp to (p - obs)
            phi = np.arccos(np.transpose(obs - self.p) * self.dp / np.dot(np.abs(obs - self.p), np.abs(self.dp)) )

            #dphi determines the magnitude of the avoidance force
            #dphi = c_1 * phi * exp(-c_2 * abs(phi))
            dphi = gamma * phi * np.exp(-beta * phi)

            #R determines the direction of the force
            #R: axis = crossproduct((obs - p),  dp), rotation = pi/2
            R = rot.from_rotvec(np.pi/2 * np.cross((obs - self.p), self.dp))

            #ddp_obs is the resulting oriented force
            #ddp_obs = R*dp*dphi
            ddp_obs = R.apply(self.dp * dphi)

            return ddp_obs

        self.ddp = (self.alpha * (self.beta * (self.gp - self.p) - tau * self.dp) + fp(x)) / tau

        self.avoidance_term = [0, 0, 0]
        for obs in self.obstacles:
            # TODO: bigger gamma for bigger objects???
            # TODO: moving objects
            # TODO: tune gamma value
            self.avoidance_term += avoidance(obs, gamma=4000, beta=20/np.pi) / tau

        self.ddp += self.avoidance_term

        # Integrate acceleration to obtain velocity
        self.dp += self.ddp * dt

        # Integrate velocity to obtain position
        self.p += self.dp * dt

        return self.p, self.dp, self.ddp, self.avoidance_term

    def rollout(self, ts, tau):
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

        for i in range(n_steps):
            p[i], dp[i], ddp[i], av[i] = self.step(x[i], dt[i], tau[i])

        return p, dp, ddp, av
