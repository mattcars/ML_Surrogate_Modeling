import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
import math as math

''' Local Stiffness Matrices '''
def element_matrix(p0, p1, A=1, E=1):

    # Geometry of member
    x0 = p0[0]
    y0 = p0[1]
    x1 = p1[0]
    y1 = p1[1]
    L = ((x1-x0)**2 + (y1-y0)**2)**0.5

    # Tranformation Matrix
    s = (y1-y0)/L
    c = (x1-x0)/L
    Q = np.array([[c, s, 0, 0], [-s, c, 0, 0], [0, 0, c, s], [0, 0, -s, c]])

    k = Q.T @ np.array([[1, 0, -1, 0], [0, 0, 0, 0], [-1, 0, 1, 0], [0, 0, 0, 0]]) @ Q
    k = E*A/L * k

    return k, Q, L


''' Get Displaced Nodes Matrix '''
def get_displacements(nodes, u):
    u_nodes = np.zeros(nodes.shape)

    for i in range(nodes.shape[0]):
        for j in range(2):
            u_nodes[i, j] = nodes[i, j] + u[2*i+j]

    return u_nodes


''' Internal Forces '''
def int_force(members, Ke, Qe, u):
    N = np.zeros(len(Ke))

    for i in range(len(Ke)):
        m = members[i]

        s = m[0]
        e = m[1]
        dof = [2*s, 2*s+1, 2*e, 2*e+1]

        fe = Ke[i] @ u[dof]
        fl = Qe[i] @ fe

        N[i] = -fl[0]

    return N


def plot_truss(nodes, members, u_nodes=[], N=[], A=[], ax=0):
    if ax==0:
        fig, ax = plt.subplots()

    # alpha parameters
    amin = 0.3
    if len(N)>0:
        Fx = max(N)
        fm = min(N)

    # line width parameters
    lm = .5
    lx = 4
    if len(A)>0:
        Amax = max(A)
        Amin = min(A)

    for i, m in enumerate(members):
        s = m[0]
        e = m[1]

        # Color according to internal force
        if len(N)>0:
            if np.round(N[i], 5) < 0:
                c = 'r'
            elif np.round(N[i], 5) == 0:
                c = 'g'
            else:
                c = 'b'
            a = (1-amin)/(Fx-fm) * (N[i]-fm) + amin

        else:
            c = 'b'
            a = 1

        # Width according to area
        if len(A)>0:
            lw = (lx-lm)/(Amax-Amin) * (A[i]-Amin) + lm
        else:
            lw = 1


        # Plot members
        ax.plot([nodes[s, 0], nodes[e, 0]], [nodes[s, 1], nodes[e, 1]], c, alpha=a, linewidth=lw)

        if len(u_nodes)>0:
            ax.plot([u_nodes[s, 0], u_nodes[e, 0]], [u_nodes[s, 1], u_nodes[e, 1]], c+'--')

    ax.scatter(nodes[:, 0], nodes[:, 1], c='r')




def DSM(nodes, members, P=-1, Pdf=9, u0=[0, 0], i0 = [0, 1], A=1, E=1):
    nf = 2 * nodes.shape[0]
    K = np.zeros((nf, nf))

    # Stiffness Matrix
    Qe = []
    Ke = []
    Le = []
    for m in members:
        s = m[0]
        e = m[1]
        dof = np.ix_([2*s, 2*s+1, 2*e, 2*e+1], [2*s, 2*s+1, 2*e, 2*e+1])

        ke, qe, le = element_matrix(nodes[s], nodes[e], A=A, E=E)
        K[dof] = K[dof] + ke
        Qe.append(qe)
        Ke.append(ke)
        Le.append(le)

    # Forces
    f = np.zeros(nf)
    R = np.zeros(nf)
    if type(P)==list or type(P)==np.ndarray:
        for i in range(len(P)):
            f[Pdf[i]] = f[Pdf[i]] + P[i]
    else:
        f[Pdf] = P


    # Degrees of Freedom
    A = i0
    B = np.delete(np.arange(0, nf, 1), i0)

    AA = np.ix_(A, A)
    AB = np.ix_(A, B)
    BA = np.ix_(B, A)
    BB = np.ix_(B, B)

    # Displacements
    u = np.zeros(nf)
    for i in range(len(u0)):
        u[i0[i]] = u0[i]

    # Find Unknown Displacements
    u[B] = la.solve(K[BB], f[B] - K[BA]@u[A])


    # Find Reactions
    R[A] = K[AA] @ u[A] + K[AB] @ u[B]

    # Total Force Vector
    f = f + R

    u_nodes = get_displacements(nodes, u)
    N = int_force(members, Ke, Qe, u)


    return u_nodes, N, Le


''' Allowable Stress Requirement '''
def req_A(f_int, L, sT, sC, E=1, r=1):

    A = []
    for i,f in enumerate(f_int):
        if f<0:
            A_stress = abs(f)/sC

            if r==0:
                A_euler=0
            else:
                I = req_I(f, L[i], E=E)
                A_euler = (r*12*I)**0.5

            A.append(max(A_stress, A_euler))
        else:
            A.append(f/sT)


    return np.array(A)


''' Euler Buckling Requirement '''
def req_I(f, L, E=1):

    I = []
    if f<0:
        I=abs(f)*L**2/(E**2 * math.pi**2)
    else:
        I=0

    return I
