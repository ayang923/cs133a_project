'''GeneratorNode.py

   This creates a trajectory generator node

   from GeneratorNode import GeneratorNode
   generator = GeneratorNode(name, rate, TrajectoryClass)

      Initialize the node, under the specified name and rate.  This
      also requires a trajectory class which must implement:

         trajectory = TrajectoryClass(node)
         jointnames = trajectory.jointnames()
         (q, qdot)  = trajectory.evaluate(t, dt)

      where jointnames, q, qdot are all python lists of the joint's
      name, position, and velocity respectively.  The dt is the time
      since the last evaluation, to be used for integration.

      If trajectory.evaluate() return None, the node shuts down.

   Node:        /generator
   Publish:     /joint_states           sensor_msgs/JointState

'''

import rclpy
import numpy as np

from asyncio            import Future
from rclpy.node         import Node
from sensor_msgs.msg    import JointState

from tf2_ros                    import TransformBroadcaster
from geometry_msgs.msg          import TransformStamped

from rclpy.node                 import Node
from rclpy.qos                  import QoSProfile, DurabilityPolicy
from rclpy.time                 import Duration
from geometry_msgs.msg          import Point, Vector3, Quaternion
from std_msgs.msg               import ColorRGBA
from visualization_msgs.msg     import Marker
from visualization_msgs.msg     import MarkerArray

from cs133a_project.TransformHelpers import *

#
#   Trajectory Generator Node Class
#
#   This inherits all the standard ROS node stuff, but adds
#     1) an update() method to be called regularly by an internal timer,
#     2) a spin() method, aware when a trajectory ends,
#     3) a shutdown() method to stop the timer.
#
#   Take the node name, the update frequency, and the trajectory class
#   as arguments.
#
class RobotNode(Node):
    # Initialization.
    def __init__(self, name, rate, Trajectory):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Initialize the transform broadcaster
        self.broadcaster = TransformBroadcaster(self)

        # Set up a trajectory.
        self.trajectory = Trajectory(self)
        self.jointnames = self.trajectory.jointnames()

        # Add a publisher to send the joint commands.
        self.pub = self.create_publisher(JointState, '/joint_states', 10)

        # Wait for a connection to happen.  This isn't necessary, but
        # means we don't start until the rest of the system is ready.
        self.get_logger().info("Waiting for a /joint_states subscriber...")
        while(not self.count_subscribers('/joint_states')):
            pass

        # Create a future object to signal when the trajectory ends,
        # i.e. no longer returns useful data.
        self.future = Future()

        # Set up the timing so (t=0) will occur in the first update
        # cycle (dt) from now.
        self.dt    = 1.0 / float(rate)
        self.t     = -self.dt
        self.start = self.get_clock().now()+rclpy.time.Duration(seconds=self.dt)

        # Create a timer to keep calculating/sending commands.
        self.timer = self.create_timer(self.dt, self.update)
        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.dt, rate))

    # Shutdown
    def shutdown(self):
        # Destroy the timer, then shut down the node.
        self.timer.destroy()
        self.destroy_node()

    # Spin
    def spin(self):
        # Keep running (taking care of the timer callbacks and message
        # passing), until interrupted or the trajectory is complete
        # (as signaled by the future object).
        rclpy.spin_until_future_complete(self, self.future)

        # Report the reason for shutting down.
        if self.future.done():
            self.get_logger().info("Stopping: " + self.future.result())
        else:
            self.get_logger().info("Stopping: Interrupted")


    # Update - send a new joint command every time step.
    def update(self):
        # To avoid any time jitter enforce a constant time step and
        # integrate to get the current time.
        self.t += self.dt

        # Determine the corresponding ROS time (seconds since 1970).
        now = self.start + rclpy.time.Duration(seconds=self.t)

        # Compute position/orientation of the pelvis (w.r.t. world).
        ppelvis = pxyz(0.0, 0.5, 1.5 + 0.5 * np.sin(self.t/2))
        Rpelvis = Rotz(np.sin(self.t))
        Tpelvis = T_from_Rp(Rpelvis, ppelvis)

        # Build up and send the Pelvis w.r.t. World Transform!
        trans = TransformStamped()
        trans.header.stamp    = now.to_msg()
        trans.header.frame_id = 'world'
        trans.child_frame_id  = 'pelvis'
        trans.transform       = Transform_from_T(Tpelvis)
        self.broadcaster.sendTransform(trans)

        # Compute the desired joint positions and velocities for this time.
        desired = self.trajectory.evaluate(self.t, self.dt)
        if desired is None:
            self.future.set_result("Trajectory has ended")
            return
        (q, qdot) = desired

        # Check the results.
        if not (isinstance(q, list) and isinstance(qdot, list)):
            self.get_logger().warn("(q) and (qdot) must be python lists!")
            return
        if not (len(q) == len(self.jointnames)):
            self.get_logger().warn("(q) must be same length as jointnames!")
            return
        if not (len(q) == len(self.jointnames)):
            self.get_logger().warn("(qdot) must be same length as (q)!")
            return
        if not (isinstance(q[0], float) and isinstance(qdot[0], float)):
            self.get_logger().warn("Flatten NumPy arrays before making lists!")
            return

        # Build up a command message and publish.
        cmdmsg = JointState()
        cmdmsg.header.stamp = now.to_msg()      # Current time for ROS
        cmdmsg.name         = self.jointnames   # List of joint names
        cmdmsg.position     = q                 # List of joint positions
        cmdmsg.velocity     = qdot              # List of joint velocities
        self.pub.publish(cmdmsg)

#
#   Demo Node Class
#
class BallNode(Node):
    # Initialization.
    def __init__(self, name, rate):
        # Initialize the node, naming it as specified
        super().__init__(name)

        # Prepare the publisher (latching for new subscribers).
        quality = QoSProfile(durability=DurabilityPolicy.TRANSIENT_LOCAL,
                             depth=1)
        self.pub = self.create_publisher(
            MarkerArray, '/visualization_marker_array', quality)

        # Initialize the ball position, velocity, set the acceleration.
        self.radius = 0.1

        self.p = np.array([0.0, 0.0, self.radius]).reshape((3,1))
        self.v = np.array([1.0, 0.1,  5.0       ]).reshape((3,1))
        self.a = np.array([0.0, 0.0, -9.81      ]).reshape((3,1))

        # Create the sphere marker.
        diam        = 2 * self.radius
        self.marker = Marker()
        self.marker.header.frame_id  = "world"
        self.marker.header.stamp     = self.get_clock().now().to_msg()
        self.marker.action           = Marker.ADD
        self.marker.ns               = "point"
        self.marker.id               = 1
        self.marker.type             = Marker.SPHERE
        self.marker.pose.orientation = Quaternion()
        self.marker.pose.position    = Point_from_p(self.p)
        self.marker.scale            = Vector3(x = diam, y = diam, z = diam)
        self.marker.color            = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.8)
        # a = 0.8 is slightly transparent!

        # Create the marker array message.
        self.mark = MarkerArray()
        self.mark.markers.append(self.marker)

        # Set up the timing so (t=0) will occur in the first update
        # cycle (dt) from now.
        self.dt    = 1.0 / float(rate)
        self.t     = -self.dt
        self.start = self.get_clock().now() + Duration(seconds=self.dt)

        # Create a timer to keep calling update().
        self.create_timer(self.dt, self.update)
        self.get_logger().info("Running with dt of %f seconds (%fHz)" %
                               (self.dt, rate))

    # Shutdown
    def shutdown(self):
        # Destroy the node, including cleaning up the timer.
        self.destroy_node()

    # Update - send a new joint command every time step.
    def update(self):
        # To avoid any time jitter enforce a constant time step and
        # integrate to get the current time.
        self.t += self.dt

        # Integrate the velocity, then the position.
        self.v += self.dt * self.a
        self.p += self.dt * self.v

        # Check for a bounce - not the change in x velocity is non-physical.
        if self.p[2,0] < self.radius:
            self.p[2,0] = self.radius + (self.radius - self.p[2,0])
            self.v[2,0] *= -1.0
            self.v[0,0] *= -1.0   # Change x just for the fun of it!

        # Update the ID number to create a new ball and leave the
        # previous balls where they are.
        #####################
        # self.marker.id += 1
        #####################

        # Update the message and publish.
        now = self.start + Duration(seconds=self.t)
        self.marker.header.stamp  = now.to_msg()
        self.marker.pose.position = Point_from_p(self.p)
        self.pub.publish(self.mark)