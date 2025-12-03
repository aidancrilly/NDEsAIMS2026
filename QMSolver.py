import jax
import jax.numpy as jnp

import diffrax

jax.config.update("jax_enable_x64", True)

hbar = 1e-1 # Planck's constant in alternative universe

class Stepper(diffrax.Euler):
    """

    :param cfg:
    """

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        del solver_state, made_jump
        y1 = terms.vf(t0, y0, args | {"dt": t1 - t0})
        dense_info = dict(y0=y0, y1=y1)
        return y1, None, dense_info, None, diffrax.RESULTS.successful

def TDSEUpdate(R, I, potential, args):
    I_ghost = jnp.concatenate([I[:1], I, I[-1:]])
    R_ghost = jnp.concatenate([R[:1], R, R[-1:]])
    R_np1 = R + args['dt']*(-0.5*hbar**2*(I_ghost[2:]-2*I_ghost[1:-1]+I_ghost[:-2]) / args['dx']**2 / args['m'] + potential(args['xcs'], jnp.zeros_like(R), args) * I) / hbar
    I_np1 = I - args['dt']*(-0.5*hbar**2*(R_ghost[2:]-2*R_ghost[1:-1]+R_ghost[:-2]) / args['dx']**2 / args['m'] + potential(args['xcs'], jnp.zeros_like(R), args) * R) / hbar
    P_np1 = R_np1**2 + I_np1*I
    return R_np1, I_np1, P_np1

def get_y0_TDSE(args):
    b = 0.5 / args['x0_width']**2
    k0 = args['m'] * args['v0']
    centred_x = args['xcs'] - args['x0_center']

    psi0 = (b/jnp.pi)**0.25 * jnp.exp(1j * k0 * centred_x / hbar) * jnp.exp(-0.5*b*centred_x**2)
    R0 = jnp.real(psi0)
    I0 = jnp.imag(psi0)
    P0 = R0**2 + I0**2

    # Advance by 1/2 time step for I0
    t = 0.5 * args['dt0_QM']
    # See https://physics.stackexchange.com/questions/386332/analytical-solution-for-a-gaussian-wave-packet-free-particle
    denom = 1 + 1j * hbar * b * t / (2 * args['m'])
    psi_phalf = psi0 / jnp.sqrt(denom) * jnp.exp(- 1j * k0**2 * t /(2 * hbar * args['m']) * (1 + 1j * hbar * b * centred_x**2 / k0)**2 / denom)

    I0 = jnp.imag(psi_phalf)
    y0 = {'R': R0, 'I': I0, 'P': P0}
    return y0


def get_QM_solution(args, potential, dt0_QM = 1e-5):
    def _wrapped_TDSE(t, y, args):
        R, I, _ = y['R'], y['I'], y['P']
        R_np1, I_np1, P_np1 = TDSEUpdate(R, I, potential, args)
        y_next = {'R': R_np1, 'I': I_np1, 'P': P_np1}
        return y_next
    
    QM_args = args | {'dt0_QM': dt0_QM, 'dt' : jnp.nan}
    equation = diffrax.ODETerm(_wrapped_TDSE)
    solver = Stepper()
    stepsize_controller = diffrax.ConstantStepSize()
    saveat = diffrax.SaveAt(ts=jnp.linspace(args['t0'], args['t1'], args['Nt']))

    y0 = get_y0_TDSE(QM_args)

    solution = diffrax.diffeqsolve(
        equation,
        solver,
        t0=QM_args['t0'],
        t1=QM_args['t1'],
        dt0=QM_args['dt0_QM'],
        y0=y0,
        args=QM_args,
        saveat=saveat,
        stepsize_controller=stepsize_controller,
        max_steps=int(1e7)
    )

    P = solution.ys['P']

    return P, solution