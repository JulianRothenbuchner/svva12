# -*- coding: utf-8 -*-


import numpy as np
import numpy.matlib as matlib
import matplotlib.pyplot as plt


def ConstGeom(inp):
    # This script constructs the geometry of the parts, as input for MechRes

    # angles dish
    phi1 = 0
    phi2 = inp['phi']/3/180*np.pi
    phi3 = 2*phi2
    phi4 = inp['phi']/180*np.pi
    # shared x-coordinates
    x1 = -inp['R']*np.sin(phi1)
    x2 = -inp['R']*np.sin(phi2)
    x3 = -inp['R']*np.sin(phi3)
    x4 = -inp['R']*np.sin(phi4)
    # shared y-coordinates
    y1 = inp['cl']+inp['R']*(1-np.cos(phi1))
    y2 = inp['cl']+inp['R']*(1-np.cos(phi2))
    y3 = inp['cl']+inp['R']*(1-np.cos(phi3))
    y4 = inp['cl']+inp['R']*(1-np.cos(phi4))
    # angle legs
    theta = np.arctan((inp['b']+x3)/y3)
    # create structure for selected type
    out = {}  # initialize output
    if inp['type']:
        # ---- beam structure ----
        # x-coordinates
        x5 = -inp['b']
        # y-coordinates
        y5 = 0
        # assemble points
        out['points'] = np.array([[x1, x2, x3, x4, x5], [y1, y2, y3, y4, y5]])
        out['parts'] = np.array([[1, 2, 3, 3], [2, 3, 4, 5]])
        # element type
        out['type'] = np.ones((max(out['parts'].shape)), dtype=int)
    else:
        # ---- rod structure ----
        # x-coordinates
        x5 = x3 + inp['cl']/2*np.cos(np.pi/3+theta)
        x6 = x3 - inp['cl']/2*np.cos(np.pi/3-theta)
        x7 = -inp['b']
        # y-coordinates
        y5 = y3 - inp['cl']/2*np.sin(np.pi/3+theta)
        y6 = y3 - inp['cl']/2*np.sin(np.pi/3-theta)
        y7 = 0
        # assemble points
        out['points'] = np.array(
            [[x1, x2, x3, x4, x5, x6, x7], [y1, y2, y3, y4, y5, y6, y7]])

        out['parts'] = np.array([[1, 2, 3, 1, 2, 3, 3, 4, 5, 5, 6], [
                                2, 3, 4, 5, 5, 5, 6, 6, 6, 7, 7]])
        # element type
        out['type'] = np.zeros((max(out['parts'].shape)), dtype=int)

    # initialise seed for every part to 0
    out['seed'] = np.zeros((max(out['parts'].shape)))
    # initialise material for every part to 0
    out['material'] = np.zeros((max(out['parts'].shape)), dtype=int)
    # initialise section for every part to 0
    out['section'] = np.zeros((max(out['parts'].shape)), dtype=int)

    # create boundary conditions
    # 1 means d.o.f. is restrained, 0 means d.o.f. is free
    out['bc'] = np.zeros((3*max(out['points'].shape)))
    # final point is clamped
    out['bc'][-3:] = 1

    # create loading vector
    out['P'] = np.zeros((3*max(out['points'].shape), 1))

    # create thermal vector
    out['T'] = np.zeros((max(out['points'].shape), 1))
    if inp['type']:
        # first 4 nodes are exposed to deltaT
        out['T'][0:4] = inp['deltaT']*np.ones((4, 1))
    else:
        # all nodes are exposed to deltaT
        out['T'] = inp['deltaT']*np.ones((max(out['points'].shape)))

    # create mirror side of structure, if so selected
    if inp['sym']:
        # connectivity
        a = np.nonzero(out['parts'][0, :] == 1)[0]
        l = max(out['points'].shape)
        # points (first point should not be duplicated)
        out['points'] = np.vstack((np.concatenate((out['points'][0, :], -out['points'][0, 1:]), axis=0).reshape(
            1, 2*l-1), np.concatenate((out['points'][1, :], out['points'][1, 1:]), axis=0).reshape(1, 2*l-1)))
        # boundary conditions (first point should not be duplicated)
        out['bc'] = np.concatenate((out['bc'], out['bc'][3:]))
        # loading vector
        out['P'] = np.concatenate((out['P'], out['P'][3:]))
        # thermal vector
        out['T'] = np.concatenate((out['T'], out['T'][1:]))
        # advancing the node numbers to be connected apart for node 1
        c = (l-1)*np.ones((np.shape(out['parts'])), dtype=int)
        c[0, a] = np.zeros((1, len(a)), dtype=int)
        out['parts'] = np.concatenate((out['parts'], c+out['parts']), axis=1)
        # double the initialized vectors
        out['type'] = np.concatenate((out['type'], out['type']))
        out['seed'] = np.concatenate((out['seed'], out['seed']))
        out['material'] = np.concatenate((out['material'], out['material']))
        out['section'] = np.concatenate((out['section'], out['section']))
    else:
        # apply symmetry boundary conditions at point 1
        out['bc'][0] = 1   # 1 means d.o.f. is restrained, 0 means d.o.f. is free
        out['bc'][2] = 1

    # convert theta to degrees
    out['theta'] = theta*180/np.pi
    # store input data to output
    out['inp'] = inp
    out['mat'] = inp['mat']
    out['sec'] = inp['sec']
    out['usr'] = inp['usr']
    out['name'] = inp['name']
    out['plotScale'] = inp['plotScale']
    return out

    # TODO: write the input settings to a text file


def MechRes2D(inp):

    # =============================================================================
    #  AE3212-II Structures assignment 2021-2022
    # =============================================================================
    # ============================Start of preamble================================
    # This Python-function belongs to the AE3212-II Sumulation, Verfication and
    # Validation Structures assignment. The file contains a function that
    # calculates the mechanical response of a 2D structure that has been
    # specified by the user.
    #
    # The function uses a structure as input and returns a structure as output.
    #
    # Proper functioning of the code can be checked with the following input:
    #
    # [input generated by ConstGeom; Constgeom will not be made available to
    # the student groups. They just get a printed set of input instructions]
    #
    # Written by Julien van Campen
    # Aerospace Structures and Computational Mechanics
    # TU Delft
    # October 2021 â€“ January 2022
    # ==============================End of Preamble================================

    # !!! the input below is just for testing, will be removed for students !!!

    # name of the structure

    inp = {}  # initializing input
    inp['name'] = 'SpaceTelescope'

    # testing input to be removed later
    # Unit set used: kg - mm - GPa, N.B.: any consistent set will do
    inp['R'] = 2400  # [mm] Radius of the telescope dish
    inp['cl'] = 600  # [mm] Clearance of telescope with base plate
    inp['b'] = 700  # [mm] Base of standing legs telescope
    inp['phi'] = 30  # [deg] opening angle of dish
    inp['type'] = 1  # [-] 1 if beam, 0 if rod structure is selected
    # [-] 1 if full structure is portrayed, 0 if only half of structure is used
    inp['sym'] = 0

    # delta T (will disappear in real input)
    inp['deltaT'] = 5e2

    # user defined seed
    inp['usr'] = {}
    inp['usr']['seed'] = 8  # [-] results in n + 2 nodes per part
    # [-] 1 if lumped mass matrix is to be used, 0 if full mass matrix is to be used (only for rod structure)
    inp['usr']['lumped'] = 1

    inp['mat'] = [{}, {}, {}]  # initialize list of dict
    # material 1: aluminium
    inp['mat'][0]['name'] = 'aluminium'
    inp['mat'][0]['E'] = 70  # [GPa]
    inp['mat'][0]['rho'] = 2.700e-06  # [kg/mm^3]
    inp['mat'][0]['alpha'] = 24e-06  # [mm/(mm deg)]
    # material 2: steel
    inp['mat'][1]['name'] = 'steel'
    inp['mat'][1]['E'] = 210  # [GPa]
    inp['mat'][1]['rho'] = 7.850e-06  # [kg/mm^3]
    inp['mat'][1]['alpha'] = 12.5e-06  # [mm/(mm deg)]
    # material 3: cfrp
    inp['mat'][2]['name'] = 'cfrp'
    inp['mat'][2]['E'] = 200  # [GPa]
    inp['mat'][2]['rho'] = 1.800e-06  # [kg/mm^3]
    inp['mat'][2]['alpha'] = 6e-06  # [mm/(mm deg)]

    inp['sec'] = [{}, {}, {}]  # initialize list of dict
    # section 1:
    inp['sec'][0]['A'] = 100  # [mm^2]
    inp['sec'][0]['I'] = 10000  # [mm^4]
    # section 2:
    inp['sec'][1]['A'] = 400  # [mm^2]
    inp['sec'][1]['I'] = 8000000  # [mm^4]
    # section 3:
    inp['sec'][2]['A'] = 500  # [mm^2]
    inp['sec'][2]['I'] = 250000  # [mm^4]

    # factor to scale plot with
    inp['plotScale'] = 1  # [-]

    # call the function that creates the geometry
    inp = ConstGeom(inp)

    # force on point 3 in negative y-direction
    inp['P'][3*(3-1)+1] = 0  # [N]
    # number of eigenfrequencies returned
    inp['nFreq'] = 5  # [-]

    # !!! the input above is just for testing, will be removed for students !!!

    # initialize output
    out = {}
    out['name'] = inp['name']

    # %% 1. Reading the input structure and txt file
    # Check the provided input
    # --------------------------------------------------------------------------
    if 'parts' in inp.keys() == False:
        print('Error: inp.parts has not been specified by user')
       # return

    # Display the structure for the user to check it visually
    # --------------------------------------------------------------------------
    plt.figure(1)
    # plot the parts
    for ix in range(len(inp['parts'][0, :])):

        plt.plot(inp['points'][0, inp['parts'][:, ix]-1],
                 inp['points'][1, inp['parts'][:, ix]-1], 'k-', marker='o', linewidth=2, markersize=12)

    # plot the points
    plt.plot(inp['points'][0, :], inp['points']
             [1, :], 'or', linewidth=4, markersize=10)
    # format the axis of the plot
    plt.xlim([min(inp['points'][0, :])-0.5, max(inp['points'][0, :])+0.5])
    plt.ylim([min(inp['points'][1, :])-0.5, max(inp['points'][1, :])+0.5])
    plt.rcParams.update({'font.size': 20})
    plt.savefig('mesh_1.png')

    # %% 2. Creating a mesh per part of the structure
    # adjust the seed according to user specification
    # --------------------------------------------------------------------------
    if inp['usr']['seed'] != 0:
        if np.all(inp['type'] == 1):
            inp['seed'] = inp['usr']['seed']*np.ones(np.size(inp['seed']))

    # write input to output
    out['inp'] = inp

    # Loop over the specified parts and create nodes and elements per part
    # --------------------------------------------------------------------------
    # initialize counter for nodes
    iN = max(inp['points'].shape)

    mesh = {}  # initialize mesh
    mesh['part'] = [dict() for i_dic in range(len(inp['parts'][0, :]))]
    # loop over parts
    for ix in range(len(inp['parts'][0, :])):
        # length of the part
        mesh['part'][ix]['length'] = np.sqrt(sum(
            (inp['points'][:, inp['parts'][1, ix]-1]-inp['points'][:, inp['parts'][0, ix]-1])**2, 1))
        # unit vector of the part (local x-axis)
        mesh['part'][ix]['direction'] = (inp['points'][:, inp['parts'][1, ix]-1] -
                                         inp['points'][:, inp['parts'][0, ix]-1])/mesh['part'][ix]['length']
        # rotation of element w.r.t. global coord. system [radians]
        mesh['part'][ix]['rotation'] = np.arctan(
            mesh['part'][ix]['direction'][1]/mesh['part'][ix]['direction'][0])
        # element length is obtained by dividing the length of the elemnt by
        # the number of seeded nodes
        mesh['part'][ix]['elementLength'] = mesh['part'][ix]['length'] / \
            (inp['seed'][ix]+1)
        # create an array with the coordinates of the nodes
        multiplierVector = np.arange(
            0, 1+(1/(inp['seed'][ix]+1))/2, (1/(inp['seed'][ix]+1)))*mesh['part'][ix]['length']

        mesh['part'][ix]['nodes'] = inp['points'][:, inp['parts'][0, ix]-1].reshape((2, 1)) @\
            np.ones((len(multiplierVector)))[
            np.newaxis] + mesh['part'][ix]['direction'].reshape((2, 1))@multiplierVector[np.newaxis]
        # number nodes in the part
        # first node
        firstNode = inp['parts'][0, ix]
        # last node
        lastNode = inp['parts'][1, ix]
        # intermediate nodes
        nInterNodes = len(multiplierVector)-2
        if nInterNodes < 1:
            interNodes = []
        else:
            interNodes = [k for k in range(iN+1, iN+nInterNodes+1)]

        # advance counter node number
        iN = iN+nInterNodes
        # assign node numbers to part
        mesh['part'][ix]['nodeNumbers'] = [firstNode] + interNodes + [lastNode]
        mesh['part'][ix]['nNodes'] = 2+nInterNodes
        # number of elements in the part
        mesh['part'][ix]['nElements'] = len(multiplierVector)-1
        # temperature of the part (only if first node and last node have equal
        # non-zero temperature
        if (inp['T'][firstNode-1] > 0) and (inp['T'][firstNode-1] == inp['T'][lastNode-1]):
            mesh['part'][ix]['DeltaT'] = inp['T'][firstNode-1]
        else:
            mesh['part'][ix]['DeltaT'] = 0

        # plot the nodes in figure 1 for visual inspection

        plt.plot(mesh['part'][ix]['nodes'][0, :],
                 mesh['part'][ix]['nodes'][1, :], 'ok', linewidth=4, markerfacecolor='None', markersize=5)

    # store total amount of nodes
    mesh['nNodes'] = iN

    # transfer part nodes to global nodes
    # --------------------------------------------------------------------------
    # Initialise array for nodal coordinates
    # (this is done here, because before the total amount of nodes was unknown)
    mesh['nodes'] = np.zeros((2, iN))
    # Each of the specified points is a node. These nodes have the lowest node
    # numbers
    mesh['nodes'][:, :max(inp['points'].shape)] = inp['points']
    # loop over parts to collect nodal coordinates
    for ix in range(len(inp['parts'][0, :])):
        # only the intermediate nodes need to be added to mesh.nodes
        if mesh['part'][ix]['nNodes'] > 2:
            for jx in range(2, mesh['part'][ix]['nNodes']):
                mesh['nodes'][:, mesh['part'][ix]['nodeNumbers']
                              [jx-1]-1] = mesh['part'][ix]['nodes'][:, jx-1]
    # plot the nodes in a separate figure
    plt.figure(2)
    for ix in range(len(inp['parts'][0, :])):
        plt.plot(inp['points'][0, inp['parts'][:, ix]-1], inp['points']
                 [1, inp['parts'][:, ix]-1], '0.8', linewidth=2)

    plt.plot(mesh['nodes'][0, :], mesh['nodes'][1, :], 'ok',
             linewidth=4, markerfacecolor='None', markersize=5)
    # format the axis of the plot

    plt.xlim([min(inp['points'][0, :])-0.5, max(inp['points'][0, :])+0.5])
    plt.ylim([min(inp['points'][1, :])-0.5, max(inp['points'][1, :])+0.5])
    plt.savefig('mesh_2.png')
    plt.show()
    # plt.rcParams.update({'font.size': 20})
   

    # pause for user to do visual inspection of structure and mesh
    # --------------------------------------------------------------------------
    input('Please inspect the mesh shown in figures 1 and 2. \
          They have also been saved in the working path as "mesh_1.png" and "mesh_2.png". \
          Press enter to continue.')

    # %% 3. Assigning element properties
    # Loop over the specified parts and assign properties to the elements
    # --------------------------------------------------------------------------
    # inintialize counter for elements
    iE = 0
    # initialize list of dict with large number of empty dict (here 1000)
    mesh['element'] = [dict() for i_dic in range(1000)]
    for ix in range(len(inp['parts'][0, :])):
        # register first element number of part
        firstElementNumber = iE+1
        # loop over elements in part
        for jx in range(mesh['part'][ix]['nElements']):
            # advance element count
            iE = iE+1
            # number of the part that the element belongs to
            mesh['element'][iE]['partNumber1'] = ix
            # node numbers belonging to the element
            mesh['element'][iE]['nodeNumber1'] = mesh['part'][ix]['nodeNumbers'][jx]
            mesh['element'][iE]['nodeNumber2'] = mesh['part'][ix]['nodeNumbers'][jx+1]
            # retrieve properties
            mesh['element'][iE]['E'] = inp['mat'][inp['material'][ix]]['E']
            mesh['element'][iE]['rho'] = inp['mat'][inp['material'][ix]]['rho']
            mesh['element'][iE]['alpha'] = inp['mat'][inp['material'][ix]]['alpha']
            mesh['element'][iE]['A'] = inp['sec'][inp['section'][ix]]['A']
            mesh['element'][iE]['I'] = inp['sec'][inp['section'][ix]]['I']
            mesh['element'][iE]['type'] = inp['type'][ix]
            mesh['element'][iE]['rotation'] = mesh['part'][ix]['rotation']
            mesh['element'][iE]['length'] = mesh['part'][ix]['elementLength']
            # divide the mass of the element over its two nodes
            mesh['element'][iE]['mass'] = mesh['element'][iE]['length'] * \
                mesh['element'][iE]['A']*mesh['element'][iE]['rho']
            mesh['element'][iE]['lumpedMassNodeNumber1'] = mesh['element'][iE]['mass']/2
            mesh['element'][iE]['lumpedMassNodeNumber2'] = mesh['element'][iE]['mass']/2
            # element temperature
            mesh['element'][iE]['DeltaT'] = mesh['part'][ix]['DeltaT']

        # register last element number of part
        lastElementNumber = iE
        mesh['part'][ix]['elementNumbers'] = range(
            firstElementNumber, lastElementNumber+1)
    # store total amount of elements
    mesh['nElements'] = iE
    # keep only filled dict
    mesh['element'] = mesh['element'][1:iE+1]

    # Create a stiffness matrix, thermal loadvector and mass matrix per element (in global coordinate system)
    # --------------------------------------------------------------------------
    for ix in range(mesh['nElements']):
        # Local stiffness matrix
        # ----------------------------------------------------------------------
        mesh['element'][ix]['Kbar'] = ((mesh['element'][ix]['E']*mesh['element'][ix]['A']) /
                                       mesh['element'][ix]['length']) * \
            np.concatenate((np.array([1, 0, 0, -1, 0, 0]).reshape(1, 6), np.zeros(
                (2, 6)), np.array([-1, 0, 0, 1, 0, 0]).reshape(1, 6), np.zeros((2, 6))), axis=0)
        # Add terms for bending stiffness if the element is a beam element
        if mesh['element'][ix]['type']:
            mesh['element'][ix]['Kbar'] = mesh['element'][ix]['Kbar'] + ((mesh['element'][ix]['E']*mesh['element'][ix]['I'])/(mesh['element'][ix]['length']**3)) * \
                np.concatenate((np.zeros((1, 6)),
                                np.array([0, 12, 6*mesh['element'][ix]['length'], 0, -12,
                                         6*mesh['element'][ix]['length']]).reshape(1, 6),
                                np.array([0, 6*mesh['element'][ix]['length'], 4*mesh['element'][ix]['length']**2, 0, -
                                         6*mesh['element'][ix]['length'], 2*mesh['element'][ix]['length']**2]).reshape(1, 6),
                                np.zeros((1, 6)),
                                np.array([0, -12, -6*mesh['element'][ix]['length'], 0,
                                          12, -6*mesh['element'][ix]['length']]).reshape(1, 6),
                                np.array([0, 6*mesh['element'][ix]['length'], 2*mesh['element'][ix]['length']**2, 0, -6*mesh['element'][ix]['length'], 4*mesh['element'][ix]['length']**2]).reshape(1, 6)))

        # Thermal load vector
        # ----------------------------------------------------------------------
        mesh['element'][ix]['Qbar'] = mesh['element'][ix]['E']*mesh['element'][ix]['A']*mesh['element'][ix]['alpha'] * \
            mesh['element'][ix]['DeltaT'] * \
            np.array([[1], [0], [0], [-1], [0], [0]], dtype=float)
        # Rotation matrix
        # ----------------------------------------------------------------------
        T = np.diag([np.cos(mesh['element'][ix]['rotation']), np.cos(mesh['element'][ix]['rotation']), 1, np.cos(
            mesh['element'][ix]['rotation']), np.cos(mesh['element'][ix]['rotation']), 1])
        T[0, 1] = np.sin(mesh['element'][ix]['rotation'])
        T[1, 0] = -np.sin(mesh['element'][ix]['rotation'])
        T[3, 4] = np.sin(mesh['element'][ix]['rotation'])
        T[4, 3] = -np.sin(mesh['element'][ix]['rotation'])
        # Rotate the local stiffness matrix to the global coordinate system
        mesh['element'][ix]['K'] = np.transpose(
            T)@mesh['element'][ix]['Kbar']@T
        # Rotate the local thermal load vector to the global coordinate system
        mesh['element'][ix]['Q'] = np.transpose(T)@mesh['element'][ix]['Qbar']
        # Mass matrix
        # ----------------------------------------------------------------------
        rhoAL = mesh['element'][ix]['rho'] * \
            mesh['element'][ix]['A']*mesh['element'][ix]['length']
        rhoIL = mesh['element'][ix]['rho'] * \
            mesh['element'][ix]['I']/mesh['element'][ix]['length']
        if mesh['element'][ix]['type']:
            # mass matrix for beam element
            mRhoA = rhoAL/420*np.concatenate((np.array([140, 0, 0, 70, 0, 0]).reshape(1, 6),
                                              np.array([0, 156, 22*mesh['element'][ix]['length'], 0,
                                                       54, -13*mesh['element'][ix]['length']]).reshape(1, 6),
                                              np.array([0, 22*mesh['element'][ix]['length'], 4*mesh['element'][ix]['length']**2, 0,
                                                       13*mesh['element'][ix]['length'], -3*mesh['element'][ix]['length']**2]).reshape(1, 6),
                                              np.array([70, 0, 0, 140, 0, 0]).reshape(
                                                  1, 6),
                                              np.array([0, 54, 13*mesh['element'][ix]['length'], 0,
                                                        156, -22*mesh['element'][ix]['length']]).reshape(1, 6),
                                              np.array([0, -13*mesh['element'][ix]['length'], -3*mesh['element'][ix]['length']**2, 0, -22*mesh['element'][ix]['length'], 4*mesh['element'][ix]['length']**2]).reshape(1, 6)))
            mRhoI = rhoIL/30*np.concatenate((np.zeros((1, 6)),
                                             np.array([0, 36, 3*mesh['element'][ix]['length'], 0, -36*mesh['element']
                                                      [ix]['length'], 3*mesh['element'][ix]['length']]).reshape(1, 6),
                                             np.array([0, 3*mesh['element'][ix]['length'], 4*mesh['element'][ix]['length']**2, 0, -
                                                      3*mesh['element'][ix]['length'], -mesh['element'][ix]['length']**2]).reshape(1, 6),
                                             np.zeros((1, 6)),
                                             np.array([0, -36*mesh['element'][ix]['length'], -3*mesh['element'][ix]
                                                       ['length']**2, 0, 36, -3*mesh['element'][ix]['length']]).reshape(1, 6),
                                             np.array([0, 3*mesh['element'][ix]['length'], -mesh['element'][ix]['length']**2, 0, -3*mesh['element'][ix]['length'], 4*mesh['element'][ix]['length']**2]).reshape(1, 6)))
            mesh['element'][ix]['m'] = mRhoA + mRhoI
        else:
            # mass matrix for rod element
            if inp['usr']['lumped']:
                # lumped mass matrix
                mesh['element'][ix]['m'] = rhoAL/2*np.eye(6)
            else:
                # regular mass matrix
                mesh['element'][ix]['m'] = rhoAL/6*np.concatenate((np.concatenate((2*np.eye(
                    3), np.eye(3)), axis=1), np.concatenate((np.eye(3), 2*np.eye(3)), axis=1)))

    # %% 4. Assembling the structure
    # Assemble the Stiffness matrix
    # --------------------------------------------------------------------------
    # initialise the global stiffness matrix
    # each node has 3 degrees of freedom: u,v, and theta
    mesh['K'] = np.zeros((3*mesh['nNodes'], 3*mesh['nNodes']))
    for ix in range(len(inp['parts'][0, :])):
        # assemlbe the global stiffness matrix
        for jx in mesh['part'][ix]['elementNumbers']:
            # bounds
            lb1 = 3*(mesh['element'][jx-1]['nodeNumber1']-1)+1
            lb2 = 3*mesh['element'][jx-1]['nodeNumber1']
            ub1 = 3*(mesh['element'][jx-1]['nodeNumber2']-1)+1
            ub2 = 3*mesh['element'][jx-1]['nodeNumber2']
            bounds = np.concatenate(
                (np.arange(lb1, lb2+1), np.arange(ub1, ub2+1)))-1
            # assemble
            mesh['K'][np.ix_(bounds, bounds)] = mesh['K'][np.ix_(
                bounds, bounds)]+mesh['element'][jx-1]['K']

    # Assemble the Thermal load vector
    # --------------------------------------------------------------------------
    out['Q'] = np.zeros((np.shape(mesh['K'])[0], 1))
    for ix in range(len(inp['parts'][0, :])):
        # assemlbe the thermal load vector
        for jx in mesh['part'][ix]['elementNumbers']:
            # bounds
            lb1 = 3*(mesh['element'][jx-1]['nodeNumber1']-1)+1
            lb2 = 3*mesh['element'][jx-1]['nodeNumber1']
            ub1 = 3*(mesh['element'][jx-1]['nodeNumber2']-1)+1
            ub2 = 3*mesh['element'][jx-1]['nodeNumber2']
            bounds = np.concatenate(
                (np.arange(lb1, lb2+1), np.arange(ub1, ub2+1)))-1
            # assemble
            out['Q'][bounds] = out['Q'][bounds]+mesh['element'][jx-1]['Q']

    # Assemble the Mass matrix
    # --------------------------------------------------------------------------
    # initialise the global mass matrix
    mesh['m'] = np.zeros((3*mesh['nNodes'], 3*mesh['nNodes']))
    for ix in range(len(inp['parts'][0, :])):
        # % assemlbe the global stiffness matrix
        for jx in mesh['part'][ix]['elementNumbers']:
            # bounds
            lb1 = 3*(mesh['element'][jx-1]['nodeNumber1']-1)+1
            lb2 = 3*mesh['element'][jx-1]['nodeNumber1']
            ub1 = 3*(mesh['element'][jx-1]['nodeNumber2']-1)+1
            ub2 = 3*mesh['element'][jx-1]['nodeNumber2']
            bounds = np.concatenate(
                (np.arange(lb1, lb2+1), np.arange(ub1, ub2+1)))-1
            # assemble
            mesh['m'][np.ix_(bounds, bounds)] = mesh['m'][np.ix_(
                bounds, bounds)]+mesh['element'][jx-1]['m']

    # %% 5. Applying Loads and Boundary Conditions
    # Assemble the loading vector
    # --------------------------------------------------------------------------
    # applied loads
    out['P'] = np.zeros((np.shape(mesh['K'])[0], 1))
    out['P'][:max((inp['points']).shape)*3] = inp['P']

    # displacements of entire system
    out['U'] = np.zeros((np.shape(mesh['K'])[0], 1))

    # reaction forces of entire system
    out['R'] = np.zeros((np.shape(mesh['K'])[0], 1))

    # reduce the amount of degrees of freedom if rod element is selected
    # --------------------------------------------------------------------------
    if np.all(inp['type'] == 1):
        # nothing happens
        pass
    else:
        # find the indices of the remaining DOFs
        remainDF = np.nonzero(matlib.repmat(
            [1, 1, 0], 1, int(max(mesh['K'].shape)/3)))[1]
        remainBC = np.nonzero(matlib.repmat(
            [1, 1, 0], 1, int(max(inp['bc'].shape)/3)))[1]
        # reduce vector with boundary conditions
        inp['bc'] = inp['bc'][remainBC]
        #  reduce stiffness and mass matrices
        mesh['K'] = mesh['K'][np.ix_(remainDF, remainDF)]
        mesh['m'] = mesh['m'][np.ix_(remainDF, remainDF)]
        # reduce displacement, load, thermal load and reaction force vectors
        out['U'] = out['U'][remainDF]
        out['P'] = out['P'][remainDF]
        out['Q'] = out['Q'][remainDF]
        out['R'] = out['R'][remainDF]

    # Remove blocked degrees of freedom
    # --------------------------------------------------------------------------
    # active degrees of freedom
    # 1 means d.o.f. is restrained, 0 means d.o.f. is free
    activeDF = np.ones((np.shape(mesh['K'])[0]))
    # find the clamped nodes
    inactiveDF = np.nonzero(inp['bc'])[0]
    # inactive degrees of freedom
    activeDF[inactiveDF] = 0
    inactiveDF = np.ones((np.shape(mesh['K'])[0])) - activeDF
    # convert to zeros and ones to indices
    activeDF = np.nonzero(activeDF)[0]
    inactiveDF = np.nonzero(inactiveDF)[0]

    # reduced stiffness matrices
    Kr = mesh['K'][np.ix_(activeDF, activeDF)]
    Ksr = mesh['K'][np.ix_(inactiveDF, activeDF)]
    # reduced load vectors
    Pr = out['P'][activeDF]
    Ps = out['P'][inactiveDF]
    Qr = out['Q'][activeDF]
    Qs = out['Q'][inactiveDF]
    # reduced mass matrix
    mr = mesh['m'][np.ix_(activeDF, activeDF)]

    # return mesh as output
    out['mesh'] = mesh

    # %% 6. Performing the analysis
    # displacements and reaction forces
    # --------------------------------------------------------------------------
    # displacements of reduced system
    KrInv = np.linalg.inv(Kr)
    Ur = KrInv@(Pr-Qr)  # ok<MINV>
    # displacements of entire system
    out['U'][activeDF] = Ur
    # reaction forces
    Rs = Ksr@Ur-(Ps-Qs)
    # reaction forces of entire system
    out['R'][inactiveDF] = Rs

    # eigenfrequency analysis
    # --------------------------------------------------------------------------
    # compute eigenvalues
    E = np.linalg.eig(-KrInv@mr)[0]
    # proces first userdefined number of eigenvalues
    E = np.sqrt(np.abs(E[:inp['nFreq']]**(-1)))
    # return eigenfrequencies to output.
    out['eigenfrequency'] = E
    
    # %% 7. Plotting results
    if np.all(inp['type'] == 1):
        # reshape 3 displacements per node
        locDisp3D = out['U'].reshape(3, mesh['nNodes'], order='F')
    else:
        # reshape 2 displacements per node
        locDisp3D = out['U'].reshape(2, mesh['nNodes'], order='F')

    # 2D locations of the displaced nodes
    locDisp = mesh['nodes'] + inp['plotScale']*locDisp3D[0:2, :]

    plt.figure(3)
    for ix in range(max(inp['parts'][0, :].shape)):
        plt.plot(inp['points'][0, inp['parts'][:, ix]-1], inp['points'][1, inp['parts'][:, ix]-1],
                 '0.8', linewidth=2)

    plt.plot(mesh['nodes'][0, :], mesh['nodes'][1, :], 'o', '0.8',
             linewidth=2, markerfacecolor='None', markersize=5)

    plt.plot(locDisp[0, :], locDisp[1, :], 'or', linewidth=2,
             markerfacecolor='None', markersize=5)

    # format the axis of the plot
    plt.xlim([min(min(inp['points'][0, :]), min(locDisp[1, :])) -
             50, max(inp['points'][0, :])+50])
    plt.ylim([min(inp['points'][1, :])-50,
             max(max(inp['points'][1, :]), max(locDisp[1, :]))+50])

    # %% 8. Storing results to txt file and output structure

    # TODO: writing to output file
    return out


inp = []
out = MechRes2D(inp)
