# -*- coding: utf-8 -*-

import sys
import math
import itertools


class Node(object):

    """
    Any kind of node (pickup, delivery, start, end, ...)

    Parameters
    ----------
    x_coord : float
        X coordinate.
    y_coord : float
        Y coordinate.
    tw_start : int
        Start of time window.
    tw_end : int
        End of time window.
    servicetime : int
        Service duration.
    load : int
        Load to be handled at this node. Pickup node have a positive load,
        while delivery nodes have a negative load.
    priority : int, optional
        Priority score. Lower values represent higher priority. (The
        default value is the maximum int value, which implies that
        nodes without priority score actually have the lowest possible
        priority)
    identity : int, optional
        Positive external ID to represent the node. (The default value
        is -1, which implies that no external ID was provided)

    Attributes
    ----------
    x_coord : float
        X coordinate of the node.
    y_coord : float
        Y coordinate of the node.
    tw_start : int
        Start of time window. The unit of measurement can be chosen by the
        developper.
    tw_end : int
        End of the time window. The unit of measurement can be chosen by the
        developper.
    servicetime : int
        Duration of the service at this node.
    load : int
        Load to be handled at this node. Pickup node have a positive load,
        while delivery nodes have a negative load.
    priority : int, optional
        A priority score for this node. Nodes with lower priority scores
        should be visited first.
    identity: int
        External ID to identify this node.
    """

    def __init__(self, identity, x_coord, y_coord, load, tw_start, tw_end,
                 servicetime, priority=sys.maxint):

        """
        Initialize a new node.
        """
        self.identity = identity
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.tw_start = tw_start
        self.tw_end = tw_end
        self.servicetime = servicetime
        self.load = load
        self.priority = priority

    def __repr__(self):
        result = "Node "
        if self.identity is not -1:
            result += str(self.identity)
        result += (" (%r,%r): TW:[%r - %r]; service time: %r; load: %r" %
                   (self.x_coord, self.y_coord, self.tw_start,
                    self.tw_end, self.servicetime, self.load))
        if self.priority is not sys.maxint:
            result += "; priority: %r" % (self.priority)
        return result

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.x_coord == other.x_coord and
                self.y_coord == other.y_coord and
                self.tw_start == other.tw_start and
                self.tw_end == other.tw_end and
                self.servicetime == other.servicetime and
                self.priority == other.priority)

    def __neq__(self, other):
        return not self.__eq__(other)


class Vehicle(object):

    """
    A Vehicle object

    Parameters
    ----------
    capacity : int
        The capacity of the vehicle.
    start_terminal : `Node`
        The start node of the vehicle.
    end_terminal : `Node`
        The end node of the vehicle.
    accessible_nodes : [int...], optional
        A list of the nodes which can be served by the vehicle, represented
        by their external id's. (Default value is an empty list)
    identity : int, optional
        Positive external ID of identifying the vehicle. (Default value is -1)

    Attributes
    ----------
    capacity : int
        Capacity of the vehicle.
    start_terminal : `Node`
        Node where the vehicle starts its route from.
    end_terminal : `Node`
        Node where the vehicle ends its route.
    accessible_nodes : [int...], optional
        A list of the nodes which can be served by the vehicle, represented by
        their external id's.
    identity : int, optional
        External ID for identifying the vehicle.
    """

    def __init__(self, capacity, start_terminal, end_terminal,
                 accessible_nodes=[], identity=-1):

        """
        Create a new vehicle
        """

        self.capacity = capacity
        self.start_terminal = start_terminal
        self.end_terminal = end_terminal
        self.accessible_nodes = accessible_nodes
        self.identity = identity

    def __repr__(self):
        result = "Vehicle "
        if self.identity is not -1:
            result += str(self.identity)
        result += (" capacity: %r, start terminal: %r, end terminal: %r" %
                   (self.capacity, self.start_terminal, self.end_terminal))
        if self.accessible_nodes is not []:
            result += "; accessible nodes: %r" % (self.accessible_nodes)
        return result


class Request(object):

    """
    A request consisting of a pickup and delivery node.

    Parameters
    ----------
    pickup_node : `Node`
        The pickup node in the request.
    delivery_node : `Node`
        The delivery node in the request.

    Attributes
    ----------
    pickup_node : `Node`
        The pickup node in the request.
    delivery_node : `Node`
        The delivery node in the request.
    id : int
        The external ID of the request, which is by default set to the external
        ID of the *pickup* node.
    """

    def __init__(self, pickup_node, delivery_node):
        """
        Create a new Request object.
        """
        self.pickup_node = pickup_node
        self.delivery_node = delivery_node
        self.id = pickup_node.identity

    def __repr__(self):
        result = "Request "
        result += (" pickup node: %r, delivery node: %r" %
                   (self.pickup_node, self.delivery_node))
        return result

    def __eq__(self, other):
        return (isinstance(other, self.__class__) and
                self.pickup_node == other.pickup_node and
                self.delivery_node == other.delivery_node)

    def __neq__(self, other):
        return not self.__eq__(other)


class Problem(object):

    """
    The optimization problem

    The optimization problem consists of two main elements, i.e. the
    set of requests and the set of vehicles.

    Parameters
    ----------
    requests : [`Request`]
        A list of requests that need to be served in the optimization
        problem.
    vehicles : [`Vehicle`]
        A list of vehicles available to serve the requests.
    runtime : float
        It is the maximum CPU time the algorithm is allowed to run (stop criterion)

    Attributes
    ----------
    K : {int : `Vehicle`}
        A dictionary mapping the internal id to a specific `Vehicle` object.
        The id's run from 0 to (m-1), with m the number of vehicles.
    m : int
        The number of vehicles in the problem
    P : {int : `Node`}
        A dictionary mapping the internal id to a specific *pickup* `Node`
        object. The id's run from 0 to (n-1), with n the number of pickup
        nodes.
    n : int
        The number of requests
    D : {int : `Node`}
        A dictionary mapping the internal id to a specific *delivery* `Node`
        object. The id's run from n to (2n-1), with n the number of pickup
        nodes.
    Pk : {int : [int]}
        A dictionary mapping the vehicle's internal id to a list of the
        internal id's of the *pickup* nodes that can be serviced by the
        specific vehicle.
    Dk : {int : [int]}
        A dictionary mapping the vehicle's internal id to a list of the
        internal id's of the *delivery* nodes that can be serviced by the
        specific vehicle.
    N : {int : `Node`}
        A dictionary mapping the internal id to a specific *pickup* or
        *delivery* Node.
    Nk : {int : [int]}
        A dictionary mapping the vehicle's internal id to a list of the
        internal id's of the *pickup* and *delivery* nodes that can be
        serviced by the specific vehicle.
    tau_k : {int : 'Node'}
        A dictionary mapping the start terminal node for each vehicle.
    tau_k_bis : {int : 'Node'}
        A dictionary mapping the end terminal node for each vehicle.
    V : {'Node'}
        A dictionary mapping all nodes (pickup, delivery, start terminal,
        end teminal).
    A : [(int,int),(int,int),...]
        A list containing all arcs between any two nodes of V, stored as
        tuples.
    Vk : {int : [int]}
        A dictionary  mapping the vehicle's internal id to a list of the
        internal id's of the pickup, delivery, start terminal and end
        terminal nodes that can be serviced by the specific vehicle.
    Ak : {int : [(int,int),(int,int),...]}
        A dictionary mapping the vehicle's internal id to a list containing
        all arcs between any two accessible nodes of Vk, stored as tuples.
    distancematrix: {(int,int) : int}
        A dictionary containing the corresponding distance for each arc
        (stored as a tuple).
    timematrix : {(int,int) : int}
        A dictionary containing the corresponding travel time for each arc
        (stored as a tuple).
    neighbor_graph : {(int,int) : int}
        A dictionary containing the weight f*(u,v) for each arc (u,v). f*(u,v)
        indicates the best solution found so far, in a solution which used
        edge (u,v). Initially, f*(u,v) is set to infinity and each time a new
        solution is found, the weights f*(u,v) of all edges used in the given
        solution are updated. These weights are used in the "historical node-
        pair removal" heuristic (destroy_factory.py).
        (stored as a tuple)
    """

    def __init__(self, requests, vehicles, runtime=60):
        """
        Initialize a new optimization problem
        """
        self.requests = requests
        self.vehicles = vehicles
        self.runtime = runtime
        # Dictionary of vehicles {[0:m-1] --> vehicle}
        self.K = self.create_vehicle_dict(vehicles)
        self.m = len(vehicles)
        # Dictionary of pickup nodes {[0:n-1]-->node}
        self.P = self.create_pickup_dict(requests)
        self.n = len(requests)
        # Dictionary of delivery nodes{(id[n,2n-1]-->node}
        self.D = self.create_delivery_dict(requests)
        # Dictionary of pickup nodes serviced by vehicle k {k[1:m]--> list of
        # pickup nodes ids [0:n-1]} Note that no nodes are stored!
        self.Pk = self.create_pk()
        # Dictionary of pickup nodes serviced by vehicle k {k[1:m]--> list of
        # delivery node ids [n:2n-1]} Note that no nodes are stored!
        self.Dk = self.create_dk()
        # Dictionary of all nodes (P + D) {[0:2n-1]--> node}
        self.N = dict(self.P.items() + self.D.items())
        # Dictionary with a list of all nodes (P+D) serviced by vehicle k
        # {k[0:m-1] --> list of node ids [0:2n-1]} Note that no nodes are
        # stored!
        self.Nk = self.create_nk()
        # Dictionary with the start terminal node for each vehicle
        # {i[2n:2n+m-1]--> Node}
        self.tau_k = self.create_tau_k()
        # Dictionary with the end terminal node for each vehicle
        # {i[2n+m:2n+2m-1]--> Node}
        self.tau_k_bis = self.create_tau_k_bis()
        # Dictionary with all nodes (N + tau_k + tau_k_bis). This contains
        # the actual nodes!
        self.V = dict(self.N.items() + self.tau_k.items()
                      + self.tau_k_bis.items())
        # List with all arcs between any two nodes of V (stored as tuples:
        # [(node1_id, node1_id), (node1_id, node2_id), ...]
        self.A = list(itertools.product(self.V.keys(), self.V.keys()))
        # Dictionary with the id's of all accessible nodes for each vehicle k
        # {k[0:m-1] --> list of node ids [0:2n+2m-1]}. Note that no nodes are
        # stored!
        self.Vk = self.create_Vk()
        # Dictionary with all arcs between any two accessible nodes of Vk for
        # each vehicle. {k[0:=m-1]--> list of arcs}
        self.Ak = self.create_Ak()
        # Dictionary with corresponding distance for each arc
        # {(node1_id , node2_id): distance}
        self.distancematrix = {}
        self.distancematrix = self.calculate_distancematrix()
        # Dictionary with corresponding travel time for each arc
        # {(node1_id , node2_id): travel time}
        self.timematrix = self.calculate_timematrix()
        # Dictionary with best solution value found for each arc
        # For each arc, this value is initialised to infinity
        # {(node1_id , node2_id): solution value}
        self.neighbor_graph = {}
        self.neighbor_graph = self.initialise_neighbor_graph()

    def create_vehicle_dict(self, vehicles):
        """
        Transform a list of vehicles to the appropriate dictionary structure.

        Parameters
        ----------
        vehicles : [int...]
            A list of available vehicles, represented by their external id's.

        Returns
        ----------
        result : {[0:m-1] --> 'Vehicle'}
            Dictionary of vehicles

        """
        result = {}
        for i in range(len(vehicles)):
            result[i] = vehicles[i]
        return result

    def create_pickup_dict(self, requests):
        """
        Transform a list of pickup nodes to the appropriate dictionary
        structure.

        Parameters
        ----------
        requests : [('Node', 'Node')...]
            A list of requests that need to be served. Each request is
            represented by a tuple consisting of the pickup node and
            delivery node making up the request.

        Returns
        ----------
        result : {[0:n-1]--> 'Node'}
            Dictionary of pickup nodes

        """
        result = {}
        for i in range(len(requests)):
            result[i] = requests[i].pickup_node
        return result

    def create_delivery_dict(self, requests):
        """
        Transform a list of delivery nodes to the appropriate dictionary
        structure.

        Parameters
        ----------
        requests : [('Node', 'Node')...]
            A list of requests that need to be served. Each request is
            represented by a tuple consisting of the pickup node and
            delivery node making up the request.

        Returns
        ----------
        result : {[n,2n-1]--> 'Node'}
            Dictionary of delivery nodes
        """
        result = {}
        n = len(requests)
        for i in range(n):
            result[n+i] = requests[i].delivery_node
        return result

    def create_pk(self):
        """
        Create dictionary of servicable pickup nodes per vehicle.
        CURRENTLY IMPLEMENTED AS IF ALL VEHICLES CAN SERVICE ALL PICKUP NODES

        Returns
        ----------
        result : {k [1:m]--> pickup nodes ids [0:n-1]}
            Dictionary of servicable pickup nodes per vehicle
        """
        result = {}
        for i in range(self.m):
            result[i] = self.P.keys()
        return result

    def create_dk(self):
        """
        Create dictionary of servicable delivery nodes per vehicle.
        CURRENTLY IMPLEMENTED AS IF ALL VEHICLES CAN SERVICE ALL DELIVERY NODES

        Returns
        ----------
        result : {k [1:m]--> delivery nodes ids [n:2n-1]}
            Dictionary of servicable delivery nodes per vehicle
        """
        result = {}
        for i in range(self.m):
            result[i] = self.D.keys()
        return result

    def create_nk(self):
        """
        Create dictionary of servicable nodes per vehicle.

        Returns
        ----------
        result : {k [1:m]--> nodes ids [0:2n-1]}
            Dictionary of servicable nodes per vehicle
        """
        result = {}
        for i in range(self.m):
            result[i] = self.Pk[i] + self.Dk[i]
        return result

    def create_tau_k(self):
        """
        Create dictionary with the start terminal node for each vehicle.

        Returns
        ----------
        result : {i [2n:2n+m-1]--> 'Node'}
            Dictionary with the start terminal node for each vehicle
        """
        result = {}
        for k in range(self.m):
            result[2*self.n+k] = self.K[k].start_terminal
        return result

    def create_tau_k_bis(self):
        """
        Create dictionary with the end terminal node for each vehicle.

        Returns
        ----------
        result : {i [2n+m:2n+2m-1]--> 'Node'}
            Dictionary with the end terminal node for each vehicle
        """
        result = {}
        for k in range(self.m):
            result[2*self.n+self.m+k] = self.K[k].end_terminal
        return result

    def create_Vk(self):
        """
        Create dictionary with the id's of all accessible nodes for each
        vehicle.

        Returns
        ----------
        result : {k [0:m-1]--> nodes ids [0:2n+2m-1]}
            Dictionary with the id's of all accessible nodes for each vehicle.
        """
        result = {}
        for k in range(self.m):
            result[k] = (self.Nk[k] + [2 * self.n + k]
                         + [2 * self.n + self.m + k])
        return result

    def create_Ak(self):
        """
        Create dictionary with the list of arcs between any two accessible
        nodes of Vk for each vehicle.

        Returns
        ----------
        result : {k [0:m-1]--> list of arcs}
            Dictionary with the list of arcs between any two accessible
            nodes of Vk for each vehicle.
        """
        result = {}
        for k in range(self.m):
            result[k] = list(itertools.product(self.Vk[k], self.Vk[k]))
        return result

    def calculate_distance(self, node1, node2):
        """
        Calculate the Euclidean distance between two nodes.

        Parameters
        ----------
        node1, node2 : 'Node'
            A node either pickup, delivery, start- or endterminal

        Returns
        ----------
        distance : int
            Euclidean distance between two nodes
        """
        distance = math.sqrt((node1.x_coord - node2.x_coord) ** 2
                             + (node1.y_coord - node2.y_coord) ** 2)
        return distance

    def calculate_distancematrix(self):
        """
        Create a dictionary with the corresponding distance for each arc in A.

        Returns
        ----------
        result : {(node1_id , node2_id): distance}
            Dictionary with the corresponding distance for each arc in A.
        """
        result = {}
        for arc in self.A:
            node1 = self.V[arc[0]]
            node2 = self.V[arc[1]]
            result[arc] = self.calculate_distance(node1, node2)
        return result

    def calculate_timematrix(self, time_function=None):
        """
        Create a dictionary with the corresponding traveltime for each arc in
        A.

        Returns
        ----------
        result : {(node1_id , node2_id): traveltime}
            Dictionary with the corresponding traveltime for each arc in A.
        """
        if time_function == None:
            time_function = lambda x: x
        return {key: time_function(value) for key, value
                in self.distancematrix.iteritems()}

    def initialise_neighbor_graph(self):
        """
        Create a dictionary with the infinity value for each arc in A.

        Returns
        ----------
        result : {(node1_id , node2_id): infinity}
            Dictionary with the infinity value for each arc in A.
        """
        result = {}
        for arc in self.A:
            result[arc] = float('inf')
        return result


    def __repr__(self):
        result = "Pickup Nodes:\n-------------\n"
        for i, node in self.P.iteritems():
            result += "%s: %r \n" % (i, node)
        result += "\nDelivery Nodes:\n---------------\n"
        for i, node in self.D.iteritems():
            result += "%s: %r \n" % (i, node)
        result += "\nStart Terminals:\n----------------\n"
        for i, node in self.tau_k.iteritems():
            result += "%s: %r \n" % (i, node)
        result += "\nEnd Terminals:\n--------------\n"
        for i, node in self.tau_k_bis.iteritems():
            result += "%s: %r \n" % (i, node)
        result += "\nVehicles:\n--------\n"
        for i, node in self.K.iteritems():
            result += "%s: %r \n" % (i, node)
        return result
