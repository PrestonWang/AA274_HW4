import numpy as np
import pdb

EPSILON_OMEGA = 1e-3

def compute_dynamics(xvec, u, dt, compute_jacobians=True):
    """
    Compute Turtlebot dynamics (unicycle model).

    Inputs:
                     xvec: np.array[3,] - Turtlebot state (x, y, theta).
                        u: np.array[2,] - Turtlebot controls (V, omega).
        compute_jacobians: bool         - compute Jacobians Gx, Gu if true.
    Outputs:
         g: np.array[3,]  - New state after applying u for dt seconds.
        Gx: np.array[3,3] - Jacobian of g with respect to xvec.
        Gu: np.array[3,2] - Jacobian of g with respect to u.
    """
    ########## Code starts here ##########
    # TODO: Compute g, Gx, Gu
    # HINT: To compute the new state g, you will need to integrate the dynamics of x, y, theta
    # HINT: Since theta is changing with time, try integrating x, y wrt d(theta) instead of dt by introducing om
    # HINT: When abs(om) < EPSILON_OMEGA, assume that the theta stays approximately constant ONLY for calculating the next x, y
    #       New theta should not be equal to theta. Jacobian with respect to om is not 0.
    
    om = u[1]
    if np.abs(om) < EPSILON_OMEGA:
        g = np.array([xvec[0]+u[0]*np.cos(xvec[2])*dt,xvec[1]+u[0]*np.sin(xvec[2])*dt,xvec[2]+dt*u[1]])
        Gx = np.zeros((3,3))
        Gx[0,0] = 1
        Gx[0,2] = -1*u[0]*np.sin(xvec[2])*dt
        Gx[1,1] = 1
        Gx[1,2] = u[0]*np.cos(xvec[2])*dt
        Gx[2,2] = 1
       
        Gu = np.zeros((3,2))
        Gu[0,0] = np.cos(xvec[2])*dt
        Gu[1,0] = np.sin(xvec[2])*dt
        Gu[0,1] = -(u[0]/2)*np.sin(xvec[2])*(dt**2)
        Gu[1,1] = (u[0]/2)*np.cos(xvec[2])*(dt**2)
        Gu[2,1] = dt
    else:
     	theta_next = xvec[2]+dt*om
     	g = np.array([xvec[0]+u[0]/u[1]*(np.sin(theta_next)-np.sin(xvec[2])), xvec[1]+u[0]/u[1]*(np.cos(xvec[2])-np.cos(theta_next)),xvec[2]+dt*u[1]])
    
     	Gx = np.zeros((3,3))
     	Gx[0,0] = 1
     	Gx[0,2] = u[0]/u[1]*(np.cos(theta_next)-np.cos(xvec[2]))
     	Gx[1,1] = 1
     	Gx[1,2] = u[0]/u[1]*(np.sin(theta_next)-np.sin(xvec[2]))
     	Gx[2,2] = 1
    
     	Gu = np.zeros((3,2))
     	Gu[0,0] = 1.0/u[1]*(np.sin(theta_next)-np.sin(xvec[2]))
     	Gu[1,0] = 1.0/u[1]*(np.cos(xvec[2])-np.cos(theta_next))
     	Gu[0,1] = (u[0]/(u[1]**2))*(-np.sin(theta_next) + np.sin(xvec[2]) + u[1]*dt*np.cos(theta_next))
     	Gu[1,1] = (u[0]/(u[1]**2))*(-np.cos(xvec[2]) + np.cos(theta_next) + u[1]*dt*np.sin(theta_next))
     	Gu[2,1] = dt

	#Just used for debugging Gu
	#pdb.set_trace()    
    
    ########## Code ends here ##########

    if not compute_jacobians:
        return g

    return g, Gx, Gu

def transform_line_to_scanner_frame(line, x, tf_base_to_camera, compute_jacobian=True):
    """
    Given a single map line in the world frame, outputs the line parameters
    in the scanner frame so it can be associated with the lines extracted
    from the scanner measurements.

    Input:
                     line: np.array[2,] - map line (alpha, r) in world frame.
                        x: np.array[3,] - pose of base (x, y, theta) in world frame.
        tf_base_to_camera: np.array[3,] - pose of camera (x, y, theta) in base frame.
         compute_jacobian: bool         - compute Jacobian Hx if true.
    Outputs:
         h: np.array[2,]  - line parameters in the scanner (camera) frame.
        Hx: np.array[2,3] - Jacobian of h with respect to x.
    """
    
    alpha, r = line

    ########## Code starts here ##########
    # TODO: Compute h, Hx
    # HINT: Calculate the pose of the camera in the world frame (x_cam, y_cam, th_cam), a rotation matrix may be useful.
    # HINT: To compute line parameters in the camera frame h = (alpha_in_cam, r_in_cam), 
    #       draw a diagram with a line parameterized by (alpha,r) in the world frame and 
    #       a camera frame with origin at x_cam, y_cam rotated by th_cam wrt to the world frame
    # HINT: What is the projection of the camera location (x_cam, y_cam) on the line r? 
    # HINT: To find Hx, write h in terms of the pose of the base in world frame (x_base, y_base, th_base)

    
    #Rotation matrix from the world frame to the base frame
    c, s = np.cos(x[2]), np.sin(x[2])
    R_world2base = np.array([[c, -s], [s, c]])
    
    #Translation vector from the world frame to the base frame
    t_world2base = x[0:2]
    t_world2base = np.asarray(t_world2base)
    t_world2base = np.reshape(t_world2base, (2, 1))
    
    #Rotation matrix from the base frame to the camera frame
    c, s = np.cos(tf_base_to_camera[2]), np.sin(tf_base_to_camera[2])
    R_base2cam = np.array([[c, -s], [s, c]])
    
    #Translation vector from the base frame to the camera frame
    t_base2cam = tf_base_to_camera[0:2] 
    t_base2cam = np.asarray(t_base2cam)
    t_base2cam = np.reshape(t_base2cam, (2, 1))
    #Pose of the camera in the world frame
    # pos_cam = np.dot(np.linalg.inv(R_world2base), t_base2cam) + t_world2base
    pos_cam = np.matmul((R_world2base), t_base2cam) + t_world2base
    
    theta_cam = x[2] + tf_base_to_camera[2]

    
    alpha_cam = line[0] - theta_cam
    d = np.linalg.norm(pos_cam, 2)
    theta_world2cam = np.arctan2(pos_cam[1], pos_cam[0])
    
    # print(theta_world2cam)
    
    r_cam = line[1] - d*np.cos(line[0]-theta_world2cam)
    
    
    
    Hx = np.zeros((2, 3))
    Hx[0, 2] = -1
    Hx[1, 0] = -np.cos(line[0])
    Hx[1, 1] = -np.sin(line[0])
    d = np.linalg.norm(t_base2cam, 2)
    beta = np.arctan2(t_base2cam[1], t_base2cam[0])
    
    Hx[1, 2] = d*np.sin(x[2]+beta-line[0])
    
    
    h = np.array([alpha_cam, r_cam])


    ########## Code ends here ##########

    if not compute_jacobian:
        return h

    return h, Hx


def normalize_line_parameters(h, Hx=None):
    """
    Ensures that r is positive and alpha is in the range [-pi, pi].

    Inputs:
         h: np.array[2,]  - line parameters (alpha, r).
        Hx: np.array[2,n] - Jacobian of line parameters with respect to x.
    Outputs:
         h: np.array[2,]  - normalized parameters.
        Hx: np.array[2,n] - Jacobian of normalized line parameters. Edited in place.
    """
    alpha, r = h
    if r < 0:
        alpha += np.pi
        r *= -1
        if Hx is not None:
            Hx[1,:] *= -1
    alpha = (alpha + np.pi) % (2*np.pi) - np.pi
    h = np.array([alpha, r])

    if Hx is not None:
        return h, Hx
    return h