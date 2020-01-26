#!/usr/bin/env python

import random
import numpy as np 
import rospy 
import PyKDL as kdl
import kdl_parser_py.urdf
from tf.transformations import *
from deap import base, creator, tools
from std_msgs.msg import Float64
from visualization_msgs.msg import Marker, MarkerArray
import cv2 as cv

class RobotArm:
    def __init__(self):
        robot_description = rospy.get_param("/robot_description")
        (ok, tree) = kdl_parser_py.urdf.treeFromParam(robot_description)
        # KDL chain
        self.arm_chain = tree.getChain("link1", "end_effector_link")
        arm_chain_n_joints = self.arm_chain.getNrOfJoints()
        self.n_joints = arm_chain_n_joints
        self.joints_limit = [(-np.pi*0.9, np.pi*0.9), (-np.pi*0.57, np.pi*0.5), (-np.pi*0.3, np.pi*0.44), (-np.pi*0.57, np.pi*0.65)]
        rospy.loginfo("Arm n-Joint : %d", arm_chain_n_joints)
        for i in range(arm_chain_n_joints):
            rospy.loginfo("Segment : %s", self.arm_chain.getSegment(i).getName())
            rospy.loginfo("Joint : %s", self.arm_chain.getSegment(i).getJoint().getName())
        # Forward Kinematics
        self.arm_fk = kdl.ChainFkSolverPos_recursive(self.arm_chain)
        # x, y, z, qx, qy, qz, qw
        self.target_pose = []

    # Forward Kinematics
    def forward_kinematics(self, joints_angle):
        _joints_angle = kdl.JntArray(len(joints_angle))
        for i in range(len(joints_angle)):
            _joints_angle[i] = joints_angle[i]
        eef_frame = kdl.Frame()
        self.arm_fk.JntToCart(_joints_angle, eef_frame)
        x, y, z = eef_frame.p 
        qx, qy, qz, qw = eef_frame.M.GetQuaternion()
        return [x, y, z, qx, qy, qz, qw]

    def generate_random_pose(self):
        acceptable = False
        while not acceptable:
            joints_angle = []
            for i in range(self.n_joints):
                lower, upper = self.joints_limit[i]
                q = random.uniform(lower, upper)
                joints_angle.append(q)
            pose = self.forward_kinematics(joints_angle)
            if pose[0] > 0.05 and pose[1] > 0.05 and pose[2] > 0.05:
                acceptable = True
        return pose, joints_angle

    # Here is objective function 
    # GA will minimize this function
    def pose_error(self, joints_angle):
        # Position Error
        current_pose = self.forward_kinematics(joints_angle)
        pos_diff = np.array(self.target_pose[0:3]) - np.array(current_pose[0:3])
        pos_diff = pos_diff.reshape(3,1)
        # Orientation Error
        _, current_pitch, _ = euler_from_quaternion([current_pose[3], current_pose[4], current_pose[5], current_pose[6]])
        _, target_pitch, _ = euler_from_quaternion([self.target_pose[3], self.target_pose[4], self.target_pose[5], self.target_pose[6]])
        pitch_diff = np.array([target_pitch-current_pitch]).reshape(1,1)
        diff  = np.vstack((pos_diff, pitch_diff))
        error = np.linalg.norm(pos_diff)
        return error

class GAIKSolver():
    def __init__(self):
        rospy.init_node('ga_ik_solver_node')
        self.robot = RobotArm()

    def init_publisher(self):
        self.joint1_pub = rospy.Publisher("/open_manipulator/joint1_position/command", Float64, queue_size=1)
        self.joint2_pub = rospy.Publisher("/open_manipulator/joint2_position/command", Float64, queue_size=1)
        self.joint3_pub = rospy.Publisher("/open_manipulator/joint3_position/command", Float64, queue_size=1)
        self.joint4_pub = rospy.Publisher("/open_manipulator/joint4_position/command", Float64, queue_size=1)
        self.marker_pub = rospy.Publisher("/visualization_marker_array", MarkerArray, queue_size=1)

    def init_subscriber(self):
        pass

    def init_genetic_algorithm(self):
        self.encoding_len = 16
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)
        self.toolbox = base.Toolbox()
        self.toolbox.register("attribute", self.random_binary)
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attribute, n=self.encoding_len*self.robot.n_joints)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutFlipBit, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=20)
        self.toolbox.register("evaluate", self.evaluate)

    def random_binary(self):
        return random.randint(0,1)

    def decode(self, individual, l=8, limits=[(-np.pi, np.pi)]):
        joints_angle = []
        for i in range(len(limits)):
            lower, upper = limits[i]
            precission = (upper - lower) / (2**l - 1)
            _sum = 0
            cnt = 0
            for j in range(i*l, i*l+l):
                _sum += individual[j] * 2**cnt 
                cnt += 1
            phenotype = _sum * precission + lower
            joints_angle.append(phenotype)
        return joints_angle

    def evaluate(self, individual):
        _limits = [(-np.pi*0.9, np.pi*0.9), (-np.pi*0.57, np.pi*0.5), (-np.pi*0.3, np.pi*0.44), (-np.pi*0.57, np.pi*0.65) ]
        joints_angle = self.decode(individual, l=self.encoding_len, limits=_limits)
        error = self.robot.pose_error(joints_angle)
        return error,

    def train(self, min_error=1e-4, max_gen=500):
        pop = self.toolbox.population(n=50)
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        CXPB, MUTPB = 1.0, 0.2

        fits = [ind.fitness.values[0] for ind in pop]
        generation = 0
        best_fitness = 999
        finished = False
        while not finished:
            # Select the next generation individuals
            offspring = self.toolbox.select(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))
            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            # Apply mutation
            for mutant in offspring:
                if random.random() < MUTPB:
                    self.toolbox.mutate(mutant)
                    del mutant.fitness.values
            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            pop[:] = offspring
        
            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]
            idx = np.argmin(fits)
            best_fitness = min(fits)
            if generation > max_gen or best_fitness <= min_error:
                break
            generation = generation + 1

        _limits = [(-np.pi*0.9, np.pi*0.9), (-np.pi*0.57, np.pi*0.5), (-np.pi*0.3, np.pi*0.44), (-np.pi*0.57, np.pi*0.65) ]
        joints_angle = self.decode(pop[idx], l=self.encoding_len, limits=_limits)
        rospy.loginfo("IK Solution")
        rospy.loginfo("Joints Angle :%s", joints_angle)
        rospy.loginfo("Best Fitness :%s", best_fitness)
        return joints_angle

    def set_joints(self, joints_angle):
        self.joint1_pub.publish(joints_angle[0])
        self.joint2_pub.publish(joints_angle[1])
        self.joint3_pub.publish(joints_angle[2])
        self.joint4_pub.publish(joints_angle[3])

    def publish_marker(self):
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "/world"
        marker.type = marker.CUBE
        marker.action = marker.ADD
        marker.scale.x = 0.015
        marker.scale.y = 0.015
        marker.scale.z = 0.015
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.position.x = self.robot.target_pose[0]
        marker.pose.position.y = self.robot.target_pose[1]
        marker.pose.position.z = self.robot.target_pose[2]
        marker.pose.orientation.x = self.robot.target_pose[3]
        marker.pose.orientation.y = self.robot.target_pose[4]
        marker.pose.orientation.z = self.robot.target_pose[5]
        marker.pose.orientation.w = self.robot.target_pose[6]
        marker_array.markers.append(marker)
        id = 0
        for marker in marker_array.markers:
            marker.id = id
            id += 1
        self.marker_pub.publish(marker_array)

    def initialize(self):
        self.init_publisher()
        self.init_subscriber()
        self.init_genetic_algorithm()

    def run(self):
        count = 0
        self.initialize()
        # Main code
        random_pose, _ = self.robot.generate_random_pose()
        self.robot.target_pose = random_pose
        rospy.loginfo("Target Pose :%s", self.robot.target_pose)
        # Visualize with marker in RVIZ
        for i in range(100):
            self.publish_marker()
            rospy.sleep(0.01)
        
        joints_angle = self.train()
        rospy.loginfo("Actual Pose :%s", self.robot.forward_kinematics(joints_angle))
        for i in range(100):
            self.set_joints(joints_angle)
            rospy.sleep(0.01)
        self.end()

    def end(self):
        pass

if __name__ == "__main__":
    ga_ik = GAIKSolver()
    ga_ik.run()