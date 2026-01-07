import jax
import jax.numpy as jnp

import equinox as eqx
import diffrax

jax.config.update("jax_enable_x64", True)

class DoublePendulum(eqx.Module):
    equation: diffrax.ODETerm
    solver: diffrax.AbstractAdaptiveSolver
    stepsize_controller: diffrax.AbstractAdaptiveStepSizeController
    saveat: diffrax.SaveAt
    ts : jax.Array

    def __init__(self, ts, rtol = 1e-5, atol = 1e-5):
        self.ts = ts
        self.equation = diffrax.ODETerm(self.DP_equations)
        self.solver = diffrax.Tsit5()
        self.stepsize_controller = diffrax.PIDController(rtol=rtol, atol=atol)
        self.saveat = diffrax.SaveAt(ts=ts)

    @staticmethod
    def DP_equations(t, y, args):
        dy = {}
        dy['theta1'] = y['theta1dot']
        dy['theta2'] = y['theta2dot']
        alpha1 = jnp.cos(y['theta1']-y['theta2'])/2
        alpha2 = 2.0 * alpha1
        f1 = -0.5*y['theta2dot']**2*jnp.sin(y['theta1']-y['theta2']) - 9.8/args['L']*jnp.sin(y['theta1'])
        f2 = y['theta1dot']**2*jnp.sin(y['theta1']-y['theta2']) - 9.8/args['L']*jnp.sin(y['theta2'])
        dy['theta1dot'] = (f1 - alpha1 * f2)/(1 - alpha1*alpha2)
        dy['theta2dot'] = (-alpha2 * f1 + f2)/(1 - alpha1*alpha2)
        return dy

    def __call__(self, init_cond, args):
        solution = diffrax.diffeqsolve(
            self.equation,
            self.solver,
            t0=self.ts[0],
            t1=self.ts[-1],
            dt0=self.ts[1]-self.ts[0],
            y0=init_cond,
            args=args,
            saveat=self.saveat,
            stepsize_controller=self.stepsize_controller,
            max_steps=int(1e7)
        )
        return solution.ys

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    DP = DoublePendulum(ts=jnp.linspace(0.0,20.0,1000))

    init_cond = {
        'theta1':jnp.pi/2.0,
        'theta2':0.0,
        'theta1dot':0.0,
        'theta2dot':0.0,
        }
    
    args = {'L' : 1.0}

    ys = DP(init_cond, args)

    plt.plot(DP.ts, ys['theta1'])
    plt.plot(DP.ts, ys['theta2'])

    init_cond = {
        'theta1':jnp.pi/2.0+1e-2,
        'theta2':0.0,
        'theta1dot':0.0,
        'theta2dot':0.0,
        }
    args = {'L' : 1.0}

    ys = DP(init_cond, args)

    plt.plot(DP.ts, ys['theta1'])
    plt.plot(DP.ts, ys['theta2'])

    plt.show()