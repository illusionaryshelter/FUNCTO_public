# POIL: Point-based One-Shot Imitation Learning with Stable Dynamical Systems

Sang Min Kim<sup>1</sup>, Jinwoo Seo<sup>1</sup>, Hyeongjun Heo<sup>1</sup>, Junho Lee<sup>1</sup>, Yonghyeon Lee<sup>2∗</sup>, and Young Min Kim<sup>1∗</sup> 

Abstract— We present POIL, a point-based one-shot imitation learning framework with stable dynamical systems. While one-shot imitation avoids collecting extensive demonstrations, successful one-shot manipulation requires not only transferring a demonstrated trajectory to a novel object but also executing it robustly under changing scene conditions, grasp configurations, and external disturbances. POIL addresses both problems through a shared representation: a set of 3D points on the object’s functional part, used jointly for trajectory transfer and closed-loop execution. The one-shot transfer from the demonstrated trajectory is enabled with point correspondences. POIL grounds the shared functional part with a multi-modal large language model, and transfers the trajectory across viewpoint, pose, and object category changes. During execution, multi-view tracking observes the same points online, and Point-set BCSDM drives them in closed loop by projecting per-point velocities onto a single rigid-body twist computed from the tracked points alone. This extends stable dynamical models from an SE(3) pose to a point set without requiring a known 3D model or pose estimator. We show that at the goal the controller becomes a gradient flow on the classical SO(3) potential, so its terminal phase inherits the almost-global convergence of that potential under a rigid-object assumption. Across simulation and realrobot experiments, POIL transfers a single demonstration across object category, grasp pose, and goal geometry, while recovering from external disturbances during execution. Project page: https://sangminkim-99.github.io/poil/ 

## I. INTRODUCTION

While recent data-driven methods perform tasks successfully in a novel environment, one still needs to acquire sufficient demonstration trajectories to adapt general models to the current setting [1], [2], [3]. One-shot imitation learning minimizes the requirement to a single demonstration, and aims to generalize the manipulation skills to unseen scenarios [4], [5], [6]. Table I compares existing approaches along four axes: action-labeled pretraining, object generalization, closed-loop execution, and a convergence guarantee. The first axis is whether the method requires action-labeled pretraining beyond the single demonstration itself. Such pretraining is a powerful route to generalization, but the data remains costly to collect, and a method free of it stays applicable where the data is unavailable. The second axis is how far the single demonstration generalizes over objects. Learning-based approaches adapt a learned policy prior to the given demonstration [4], [5], [6], inheriting both the pretraining requirement and generalize only within its data coverage. Trajectory-transfer approaches instead directly map the demonstrated motion to a new scene through semantic correspondence [7], [8], [9], [10], and can transfer a trajectory to novel geometry, appearance, or even a novel object category by identifying corresponding functional parts. For the first two axes, we find that trajectory transfer is a natural fit. 

Successful manipulation, however, requires more than transferring the demonstrated trajectory. Most prior works on trajectory transfer focus on establishing static correspondences between the demonstration and the target scene, and execute the transferred sequence of actions as an open-loop replay [7], [8], [9], [10]. The transferred actions must still adapt to the current scene context and execution conditions, where feasible grasp configurations, object placements, and external disturbances may cause the demonstrated execution to become infeasible or deviate from the desired motion. Closed-loop control observes the object online and corrects such deviations from feedback. Feedback alone, however, offers no assurance that the corrected motion actually reaches the goal, especially from states far outside the demonstration. Stable dynamical systems provide exactly this convergence guarantee, continuously correcting deviations while ensuring convergence [11], [12], [13]. However, these methods have primarily focused on robust execution given an appropriate state representation and reference trajectory, whereas oneshot trajectory transfer has focused on constructing such trajectories for novel objects. The integration of these two complementary capabilities, which would encompass all four axes in Table I, is still largely unexplored. 

To bridge trajectory transfer and stable execution, we present POIL, a framework built on a unified representation: a set of 3D points on the object. By anchoring these points directly to the object’s functional part rather than the gripper, the policy deduces the desired movements of the designated part in the object-centric space, and can naturally generalize across different embodiments or object categories. After the corresponding points are established, the representation compactly summarizes the spatial configurations for physical execution, detached from the semantic reasoning. Tracking these points under a stable controller closes the loop and provides a convergence guarantee at the goal without requiring dense 3D object geometry. 

Building on the point-based representation, POIL comprises two stages: transfer and execution. The transfer stage grounds the task-specific functional part with a Multi-modal Large Language Model (MLLM) and assigns corresponding points to it. For example, we can transfer the skill of hanging a mug to a teapot by prompting the functional part, handles, of them, instead of matching the most similar visual feature of the entire object [14]. Grounding by name stays robust to large viewpoint changes and broadens the class of target objects a single demonstration can reach. The point motion is then extracted from the demonstration with a multi-view point tracker robust to large rotations [15] and transferred to the target. When the target goal geometry differs, we adapt the trajectory to the new fixture with a few user-clicked correspondences. The execution stage tracks the same points online and drives them along the transferred trajectory with Point-set BCSDM, our extension of Behavior Controllable Stable Dynamical Model (BCSDM) [12] from an SE(3) pose to points. It applies BCSDM to each point and projects the per-point velocities onto a single rigid-body twist for the gripper, and we show that this projection is a gradient descent on a point-set alignment cost, so the terminal phase inherits the almost-global convergence of the classical SO(3) potential without access to the full 3D model. Since the controller acts on object points rather than the demonstrated gripper pose, POIL can start from any feasible grasp; we select one by simulating rollouts in advance and rejecting grasps that violate joint limits or collide with the scene. 


TABLE I



ONE-SHOT IMITATION LEARNING APPROACHES FOR MANIPULATION, COMPARED ALONG THE FOUR AXES DISCUSSED IN SEC. I.


<table><tr><td>Approach</td><td>No action pretrain.</td><td>Object gen.</td><td>Closed loop</td><td>Conv. guarantee</td></tr><tr><td>Meta / in-context [4], [6]</td><td>✗</td><td>in-dist.</td><td>√</td><td>✗</td></tr><tr><td>Correspondence [9], [10]</td><td>√</td><td>func. part‡</td><td>✗</td><td>✗</td></tr><tr><td>Alignment [7], [8]</td><td>√</td><td>intra-cat.</td><td><eq>\checkmark^†</eq></td><td>✗</td></tr><tr><td>Analytical SDS [11], [12]</td><td>√</td><td>✗</td><td>√</td><td>√</td></tr><tr><td>POIL (ours)</td><td>√</td><td>func. part‡</td><td>√</td><td>√</td></tr></table>


<sup>†</sup>Closed-loop alignment to an intermediate pose, then open-loop replay. <sup>‡</sup>Requires geometrically similar parts. 


Experiments in simulation and on a real robot demonstrate that POIL transfers a single demonstration across object category, grasp pose, and goal geometry, while recovering from external disturbances. Our contributions are threefold: (i) We propose POIL, which attains both generalization and a convergence guarantee from a single demonstration by sharing the point-set representation between transfer and execution. (ii) We introduce Point-set BCSDM, extending BCSDM [12] from SE(3) pose trajectories to points, and show that its terminal phase is almost-globally stable. (iii) We design a name-based functional-part transfer with MLLMs that establishes correspondence under large pose change, where dense feature matching struggles. 

## II. RELATED WORKS

## A. One-Shot Imitation Learning

One-shot imitation transfers a skill from a single demonstration to novel scenes. Existing approaches broadly fall into two directions: learning a generalizable policy from prior data, or directly transferring the demonstrated trajectory to a novel scene. Learning-based methods adapt from a single demonstration via meta-learning or in-context conditioning [5], [4], [6]. Their generalization comes from pretraining on large-scale action-labeled data, which is far more costly to collect than image or text data. Thus, the policy’s generalizability is still restricted by the coverage of this data, and there is no convergence guarantee. In contrast, other methods transfer the demonstration without training. Correspondence-based methods map the demonstration to the target via a shared functional part, generalizing even across categories, but replay the motion open-loop, without correcting deviations during execution [9], [10]. Alignmentbased methods close the loop only to reach a predefined starting pose, then replay open-loop, and their instance-level alignment confines them to intra-category objects [7], [8]. 

## B. Keypoint Representation for Robotic Manipulation

Keypoints provide a compact and object-centric interface for manipulation [16], [17], [18], [19], [20], [21]. Obtaining them often relies on supervision specific to a task or object category: kPAM [16] trains a category-specific detector, KETO [22] needs tool-specific demonstrations, and K-VIL [18] transfers only within an object category. More recently, foundation models build keypoint-based task structure with less supervision — spatio-temporal constraints in ReKep [19], affordance marks in MOKA [20] — inferring the task without demonstrations, orthogonal to transferring a demonstrated skill. Closest to ours, RoboTAP [23] track points to follow a demonstrated trajectory in closed loop, but on a fixed instance, and Point Policy [21] uses points for both observation and control, but learns a behaviorcloning policy from many demonstrations; neither offers a convergence guarantee. 

## C. Stable Dynamical Systems

A stable dynamical system encodes a demonstration as a time-independent velocity field whose solutions converge to the goal from any initial state, correcting deviations along the way. Built on these formulations, stable dynamical systems have deduced convergence guarantees and been widely adopted in robotic manipulation [24]. These methods primarily address the execution problem, assuming that an appropriate state representation and reference trajectory are already available. Applied to manipulation, the state is a single rigid-body SE(3) pose assigned to one body: either the object, which requires a known 3D model and runtime pose estimation, or the gripper, under the assumption that the object is rigidly held in a fixed grasp [13], [12]. Behavior-Controllable Stable Dynamical Models (BCSDM) [12] realize such convergence on SE(3) by driving the gripper pose, and interpolates between mimicking and contracting behaviors through a single parameter. We instead combine stable execution with one-shot trajectory transfer through an objectcentric point representation, extending the convergence guarantee to transferred point trajectories without requiring estimated poses or full 3D geometry. With our semantic transfer stage, the formulation extends the convergence guarantee to novel objects; we also observe empirically that execution tolerates mild non-rigidity (Sec. IV-B, Hanging Bag), though our analysis assumes a rigid object. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-11/784c5ca8-2b0a-4805-8bfc-f3dcb4b8e5df/c221433626a1ae0f7a0b439f15d839d9e3fa1407b3fcfd60465c5eb151a0378a.jpg)



Fig. 1. Overview of POIL. The transfer stage grounds the functional part, establishes point correspondences, and transfers the demonstrated point trajectory to the target scene. The execution stage tracks the same points online and drives them with Point-set BCSDM.


## III. METHOD

POIL takes a single demonstration and transfers the manipulation skill to a novel target object that shares a functional part, executing the task via a closed-loop controller with convergence guarantees (see Fig. 1). The two stages share one representation – a set of 3D points on the object’s functional part – so that a single abstraction yields both the object generalization of the transfer stage and the object-centric convergence guarantee of the execution stage. Formally, the demonstration is a multi-view RGB-D video sequence composed of frames of image and depth observations $\{ ( I _ { n , \tau } ^ { \mathrm { s r c } } , D _ { n , \tau } ^ { \mathrm { s r c } } ) \}$ , where $n \in [ 1 , N ]$ is the index to a camera and $\tau \in [ 1 , T ]$ is the time stamp. During the demonstration, the robot or a human hand manipulates an object. At test time, we observe a target scene $( I _ { n } ^ { \mathrm { t g t } } , D _ { n } ^ { \mathrm { t g t } } )$ with a novel object, where the robot tries to grasp the target object and drives the functional part along the transferred trajectory. 

## A. Point Trajectory Transfer

POIL transfers demonstrations through functional-part point trajectories, which capture task-relevant object motion while remaining independent of gripper pose and wholeobject appearance. We obtain this object-centric trajectory by detecting the functional part, aligning the demonstration and target part point clouds, and recovering point motion from the demonstration. When the target fixture differs from the demonstration, we further adapt the trajectory to the new fixture. 

We localize the functional part by its semantic label rather than its appearance, whereas dense image features drift and fail to match across such changes [14]. Given demonstration and target object labels and a shared part label, we detect the part with Rex-Omni [25], an MLLM-based open-vocabulary detector that returns a bounding box from an image and a text label. In each scene, we query every view with the object label to crop the object. Querying each crop separately for the part, however, often returns a spurious box when the part is self-occluded in that view (Fig. 2). We therefore introduce multi-view prompting: we concatenate all N object crops into a single image grid and issue one part query, which suppresses these errors. We convert the bounding boxes into part masks using SAM2 [26], and unproject them with depth into 3D part point clouds $\mathcal { P } _ { \mathrm { d e m o } }$ and $\mathcal { P } _ { \mathrm { t g t } } . \operatorname { O n } \mathcal { P } _ { \mathrm { d e m o } }$ we sample $N _ { k }$ points $\{ \bar { \bf x } _ { i } ^ { \mathrm { d e m o } } \} _ { i = 1 } ^ { N _ { k } }$ by farthest-point sampling, the objectcentric point-set we transfer in the next steps. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-11/784c5ca8-2b0a-4805-8bfc-f3dcb4b8e5df/feadaa6d3a686f922a668fd6a580d5832ef5772eb0ba0a34df14813cd6d616ff.jpg)



Fig. 2. Handle detection from multi-view mug images. Multi-view prompting enables Rex-Omni to localize the handle using cross-view context.


With the functional part localized, the remaining problem is geometric: assuming the functional part keeps a similar structure across objects, a dense correspondence between the demonstration and target parts exists to transfer the points. We recover this correspondence by non-rigid registration. Without a shared frame, however, it is ill-posed, collapsing to the least-deformation solution instead of the correct one. We therefore first fit a coarse transform $T - \mathbf { a }$ rigid transform $( R , \mathbf { t } ) \in S E ( 3 )$ with anisotropic scaling along the principal axes of $\mathcal { P } _ { \mathrm { d e m o } }$ . We initialize T by searching rotation and translation candidates, scoring each by Chamfer distance between $\mathcal { P } _ { \mathrm { d e m o } }$ and $\mathcal { P } _ { \mathrm { t g t } } .$ , then refine the top K by gradient descent and select the best as $T$ [27]. We then refine the alignment with occlusion-aware non-rigid registration (OAReg) [28], which optimizes a deformation field over the coarsely aligned parts. Passing the sampled demonstration points $\left\{ \mathbf { x } _ { i } ^ { \mathrm { d e m o } } \right\} _ { i = 1 } ^ { N _ { k } }$ through this field yields the target points 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-11/784c5ca8-2b0a-4805-8bfc-f3dcb4b8e5df/8e308d388ee651bfd3c651cab6e612b01ee3227458830ce1387788ba2961d5ae.jpg)



Fig. 3. Trajectory adaptation to a new fixture. User-clicked correspondences between the demonstration’s straight rack (left) and the target’s L-shaped rack (right) define a TPS warp Φ. Applying Φ to every waypoint of the reference trajectory yields the adapted trajectory in the target scene.


$\{ \mathbf { x } _ { i } ^ { \mathrm { t g t } } \} _ { i = 1 } ^ { N _ { k } }$ 

After establishing which functional-part points correspond, we recover how those points move in the demonstration. This temporal point trajectory is the object-centric motion that will later be followed by the controller. Because the object can undergo large out-of-plane rotations, single-view point trackers often lose these points [29], [30]. We therefore track them with MVTracker [15], a multi-view RGB-D tracker stable under such motion. At each timestep, we rigidly align the target points $\{ \mathbf { x } _ { i } ^ { \mathrm { t g t } } \} _ { i = 1 } ^ { N _ { k } }$ to the tracked demonstration configuration by the Kabsch method [31], yielding transferred point trajectories $\{ \mathbf { x } _ { i } ^ { \mathrm { t r } } ( \tau ) \} _ { i = 1 } ^ { N _ { k } }$ . When the target fixture matches the demonstration, these trajectories directly become the controller reference, $\mathbf { x } _ { i } ^ { * } ( \tau ) = \mathbf { x } _ { i } ^ { \mathrm { t r } } ( \tau )$ 

When the fixture changes shape or pose, directly reusing this trajectory can miss the target geometry, so we non-rigidly adapt it to the new fixture (Fig. 3). Large fixture deformations make fully automatic registration unreliable [28], so we ask the user to click a few corresponding point pairs on the two fixtures. We use the clicked points to segment the fixtures with SAM2 [26] and depth unprojection, then run OAReg [28] on the resulting fixture point clouds. During OAReg optimization, we add a sparse correspondence loss that aligns the clicked 3D point pairs. The optimized deformation field yields dense correspondences between the fixtures. We fit a Thin Plate Spline (TPS) [32] $\Phi : \mathbb { R } ^ { 3 }  \mathbb { R } ^ { 3 }$ to these correspondences and apply it to every point in the transferred trajectories, yielding the adapted reference $\mathbf { x } _ { i } ^ { * } ( \tau ) = \Phi ( \mathbf { x } _ { i } ^ { \mathrm { t r } } ( \tau ) )$ . 

## B. Closed-Loop Execution via Point-set BCSDM

By tracking and controlling functional-part points directly, POIL executes the transferred trajectory in closed loop with a convergence guarantee, without a known 3D model or runtime pose estimation. The execution stage proceeds in three steps: (i) offline grasp selection to identify a feasible initial grasp, (ii) online point tracking to update the functional part point states in real time, and (iii) Point-set BCSDM control to drive the object along the transferred trajectory. 

Grasp selection. POIL’s object-centric point representation provides inherent grasp invariance during execution. Because the controller acts on object points rather than the demonstrated gripper pose, the robot can select any feasible grasp on the novel object, avoiding the demonstrated grasp when it is in collision or violates joint limits. When the functional part is symmetric under a $1 8 0 ^ { \circ }$ rotation (detected by a Chamfer test after flipping about each axis) about one of its principal axes, the point correspondence from Sec. III-A is ambiguous: the mirrored matching fits the part equally well but yields a different reference trajectory. We therefore roll out each candidate grasp under every symmetric matching as well. Along each simulated path we check inverse-kinematics feasibility and approximate robot-scene collision between robot link spheres and the scene point cloud. Each graspmatching pair is scored by joint-limit and collision violations, and the lowest-cost pair is selected for execution. 

Online point tracking. The point-based representation gives the controller a feedback state on a novel object without estimating its full pose or 3D model. We track the transferred points online with the same multi-view point tracker [15] used in the transfer stage. To improve robustness, we additionally track points sampled on the whole object as support points [29], which stabilize the part-point estimates under occlusion and large rotations, while only the part points drive the controller. Because the tracker runs at a lower frequency than the control loop, we propagate the point positions between tracker updates using the end-effector motion computed from robot forward kinematics (FK), and reset them to the tracker’s output whenever a new frame arrives. 

Point-set BCSDM controller. Point-set BCSDM turns the reference point trajectory into closed-loop velocities for the functional-part points. Given reference trajectories $\{ x _ { i } ^ { * } ( \tau ) \} _ { i = 1 } ^ { N _ { k } }$ and current point positions $\{ x _ { i } \} _ { i = 1 } ^ { N _ { k } }$ , we index the targets by τ and define 

$$
\tau^ {*} = \arg \min _ {\tau} \sum_ {i} \left\| x _ {i} - x _ {i} ^ {*} (\tau) \right\| ^ {2}\tag{1}
$$

as the closest target configuration to the current one. Each point is driven by a velocity field 

$$
v _ {i} = \dot {x} _ {i} ^ {*} (\tau^ {*}) + \eta \left(x _ {i} ^ {*} (\tau^ {*}) - x _ {i}\right),\tag{2}
$$

which interpolates between mimicking the target velocity $( \eta ~  ~ 0 )$ and contracting toward it (large $\eta ) .$ , following BCSDM [12]. The robot holds the object with a fixed grasp, so it can only move all points together as one rigid motion. We therefore find the single twist $( v , \omega )$ that best matches the per-point velocities $\{ v _ { i } \}$ in the least-squares sense, 

$$
(v, \omega) = \underset {v, \omega} {\arg \min} \sum_ {i} \left| \left| v + \omega \times (x _ {i} - \bar {x}) - v _ {i} \right| \right| ^ {2},\tag{3}
$$

about the centroid $\begin{array} { r } { \bar { x } = \frac { 1 } { N _ { k } } \sum _ { i } x _ { i } } \end{array}$ . In implementation, we robustly integrate this fit with RANSAC [33] to reject outlier point velocities before the final least-squares solve. We then map this twist to the gripper wrist through the fixed grasp transform and send it to the robot. 

Stability. We assume a rigid object, $x _ { i } = R x _ { i } ^ { \mathrm { b } } + t$ with fixed body-frame points $x _ { i } ^ { \mathrm { b } }$ , and analyze the terminal phase, where $\tau ^ { * } = T , \dot { x } _ { i } ^ { * } ( \tau ^ { * } ) = 0 .$ , and $\eta$ only rescales time, so we set $\eta = 1$ . We analyze the least-squares solve (3) without 

RANSAC; under the rigid-object assumption all per-point velocities agree with a single twist, so RANSAC rejects nothing. The twist decreases 

$$
D = \frac {1}{2} \sum_ {i} \left\| x _ {i} - x _ {i} ^ {*} (\tau^ {*}) \right\| ^ {2},\tag{4}
$$

which we verify below. Translation is a strictly convex quadratic and trivially stable, so we focus on rotation. Taking the body frame at the target $( x _ { i } ^ { * } = x _ { i } ^ { \mathrm { b } }$ , hence $R = I$ at the target) and centering $( \sum _ { i } x _ { i } ^ { \mathrm { b } } = 0 )$ 

$$
D (R) = \operatorname{tr} (W) - \operatorname{tr} (W R), \quad W = \sum_ {i} x _ {i} ^ {\mathrm{b}} x _ {i} ^ {\mathrm{b} ^ {\top}} \succ 0.\tag{5}
$$

Let [v] denote the skew-symmetric matrix with $[ \boldsymbol { v } ] \boldsymbol { u } = \boldsymbol { v } \times \boldsymbol { u }$ and $\bar { ( \cdot ) } ^ { \vee }$ its inverse. Substituting $v _ { i } = x _ { i } ^ { \mathrm { b } } - R x _ { i } ^ { \mathrm { b } }$ into (3), the normal equations in the body frame have Gram matrix $A = \operatorname { t r } ( W ) I - W \succ 0$ , the unit-mass inertia of the point set, and right-hand side equal to the skew part of W R. The closed-loop rotation is therefore ${ \dot { R } } = R \left[ \omega ( R ) \right]$ with 

$$
\omega (R) = A ^ {- 1} \left(R ^ {\top} W - W R\right) ^ {\vee}, \quad \dot {D} = - \| \omega (R) \| _ {A} ^ {2} \leq 0,\tag{6}
$$

where $\| u \| _ { A } ^ { 2 } = u ^ { \top } A u .$ That is, the least-squares twist is the gradient descent of $D$ in the metric A, the kineticenergy-metric gradient of [34] with the point-set inertia in place of the body inertia. D is the potential $\operatorname { t r } ( W ( I - R ) )$ proposed in [34] as a navigation function on $S O ( 3 )$ . Unlike the geodesic cost of BCSDM [12], it has three critical points besides I; it is known that none is a local minimum [35], so the gradient flow converges almost-globally [34], [36]. We restate the argument for our point-set parameterization, where W comes from tracked points only, assuming general position: W has distinct positive eigenvalues $\lambda _ { 1 } < \lambda _ { 2 } < \lambda _ { 3 }$ 

Theorem $\begin{array} { r } { I \colon \omega ( R ) = 0 } \end{array}$ iff $R \in \{ I , R _ { 1 } ^ { * } , R _ { 2 } ^ { * } , R _ { 3 } ^ { * } \}$ , where $R _ { k } ^ { * }$ is the rotation by π about the k-th principal axis of $W .$ This is the $P = W$ case of [34], due to [35]; we include a short proof. 

Proof: $\omega ( R ) = 0$ iff $W R = R ^ { \top } W = ( W R ) ^ { \top }$ , so WR is symmetric and $( W R ) ^ { 2 } = W R R ^ { \top } W = W ^ { 2 }$ . In the principal frame $W ^ { 2 }$ has distinct diagonal entries, so WR, which commutes with $W ^ { 2 }$ , is diagonal with entries $\pm \lambda _ { k }$ Then $R = W ^ { - 1 } ( W R ) = \mathrm { d i a g } ( \pm 1 , \pm 1 , \pm 1 )$ and det $R = 1$ leaves the four rotations stated. 

Lemma 2: Each $R _ { k } ^ { * }$ is a nondegenerate critical point of D and not a local minimum. 

Proof: Expanding $D ( R ^ { * } \exp ( [ \xi ] ) )$ ) to second order with $[ \xi ] ^ { 2 } = \xi \bar { \xi } ^ { \top } - \mathbf { \bar { \| } } \xi \| ^ { 2 } I$ and $\mathrm { t r } ( W R ^ { * } [ \xi ] ) = 0$ gives Hessian $H ( R ^ { * } ) ~ = ~ \mathrm { t r } ( W R ^ { * } ) I - W R ^ { * }$ the Morse-index matrix of [35]. In the principal frame $W R _ { k } ^ { * }$ is diagonal with $\lambda _ { k }$ in position k and $- \lambda _ { i } , - \lambda _ { j }$ elsewhere $\ddot { ( \{ i , j \} } = \{ 1 , 2 , 3 \} \backslash \{ k \} )$ , so $H ( R _ { k } ^ { * } )$ is diagonal with entries $- \lambda _ { i } - \lambda _ { j } < 0 , \lambda _ { k } - \lambda _ { j }$ and $\lambda _ { k } - \lambda _ { i }$ . The last two are nonzero by distinctness, so $H ( R _ { k } ^ { * } )$ is nonsingular, and the first is negative, so $R _ { k } ^ { * }$ is not a minimum. 

Theorem 3: The rotational dynamics are almost-globally asymptotically stable at $R = I$ 

Proof: By Theorem 1 and Lemma 2, and since $H ( I ) =$ $A \succ 0 , D$ is a Morse function on SO(3) with the unique minimum I, and (6) is its gradient flow in the metric A. The stable manifolds of the three non-minimum critical points have dimension at most two, so all trajectories outside a measure-zero set converge to I [34]. ■ 


TABLE II



CONTROL ROBUSTNESS UNDER VARYING INITIAL POSES. SUCCESS RATE (%) OVER 100 TRIALS. NEAR DENOTES THE REGION USED TO COLLECT DEMONSTRATIONS FOR PP-50, AND FAR SAMPLES POSES OUTSIDE IT.


<table><tr><td rowspan="2">Policy</td><td rowspan="2">#Demos</td><td colspan="2">PnP Cube</td><td colspan="2">Mug Insert.</td><td colspan="2">Reshelving</td><td colspan="2">Hanging Bag</td></tr><tr><td>Near</td><td>Far</td><td>Near</td><td>Far</td><td>Near</td><td>Far</td><td>Near</td><td>Far</td></tr><tr><td>PP [21]</td><td>1</td><td>100</td><td>54</td><td>33</td><td>25</td><td>71</td><td>31</td><td>51</td><td>45</td></tr><tr><td>PP [21]</td><td>50</td><td>100</td><td>100</td><td>100</td><td>84</td><td>100</td><td>57</td><td>100</td><td>100</td></tr><tr><td>IP [6]</td><td>1</td><td>100</td><td>100</td><td>0</td><td>0</td><td>4</td><td>3</td><td>23</td><td>19</td></tr><tr><td>Ours</td><td>1</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td><td>100</td></tr></table>


PP: Point Policy [21], IP: Instant Policy [6]. 


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-11/784c5ca8-2b0a-4805-8bfc-f3dcb4b8e5df/90df2c9860587a6cc2581102c46f56e6d72053d99de73418911410ca6df91461.jpg)



Fig. 4. Functional-part point transfer under viewpoint changes. We report part point error (left) and the fraction of transferred points within a distance threshold (right). The ablation without multi-view prompting (MVP) isolates the effect of cross-view reasoning.


## IV. EXPERIMENTS

We organize the evaluation into four studies. We isolate the performance of functional-part point transfer (Sec. IV-A) and control (Sec. IV-B) with ground-truth inputs. We then evaluate the full pipeline (Sec. IV-C) across pose, grasp, object, and goal variations, as well as external disturbances. We use $N _ { k } ~ = ~ 1 0$ functional-part points, $\eta \ = \ 3 ,$ , and 40 support points for all experiments. 

## A. Functional-part Point Transfer

We evaluate whether our functional-part point transfer remains accurate under large viewpoint and pose changes. 

Baselines. We compare against dense feature matching baselines that rely only on visual similarity. Single-view baselines lift 2D correspondences to 3D using depth, with features from DINOv2 [37] and DIFT [38]. Multi-view baselines aggregate 2D features from two views into a 3D feature point cloud following $\mathrm { D ^ { 3 } F }$ Fields [39], then transfer keypoints by nearest neighbor matching. We pair this aggregation with both DINOv2 and DIFT features. POIL is given the object and functional-part names used for prompting; this matches our setting, where the task specifies which part should transfer, and lets us isolate whether name-based part grounding improves correspondence under pose change. We also ablate multi-view prompting by querying Rex-Omni [25] independently on each view. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-11/784c5ca8-2b0a-4805-8bfc-f3dcb4b8e5df/fd9ff3bcd353afb2667606f3a831859d2211ce7572490134e6823776d596ad7d.jpg)



Fig. 5. Demonstration sequences for the control tasks. Object motion is rendered from blue to red.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-11/784c5ca8-2b0a-4805-8bfc-f3dcb4b8e5df/a49d2bba7fd00edf14f565f3a8c2b3629fabf8f58eb29794a00e13d704851cda.jpg)



Fig. 6. Success rate on Mug Insertion versus the initial pose’s xy distance to the training region (20 cm bins; shaded: in-distribution). The 50-demo Point Policy degrades with distance, while ours stays at 100%.


Evaluation setup. As no standard evaluation exists for multi-view functional-part point transfer, we construct a controlled setting. We render objects from three categories (mug, pan, scissors) at 12 viewpoints spaced 30<sup>◦</sup> apart on a circle around the object. For each object, we treat one view as the demonstration and evaluate target views at azimuth offsets from 30<sup>◦</sup> to 90<sup>◦</sup>. 

Quantitative results. Fig. 4 reports normalized point error in units of the object’s longest bounding-box dimension and the fraction of transferred points within a distance threshold. Ours achieves the lowest point error at every angular offset and the highest correct-point fraction at every threshold. This indicates that grounding the functional part by name is more robust to pose change than matching dense visual features alone. The per-view ablation confirms that multiview prompting suppresses the false detections described in Sec. III-A (Fig. 2). 

## B. Pose-Robust Control with Point-set BCSDM

We evaluate the core control property of Point-set BCSDM: reaching the demonstrated goal from poses far outside the demonstration distribution. To isolate control from perception and manipulation constraints, all policies receive ground-truth point trajectories and directly actuate the object. We compare against two learned baselines. Point Policy [21] is a behavior cloning method with a point representation; with a single demonstration it memorizes the world-frame trajectory. Instant Policy [6] conditions on the demonstration through a motion prior learned from a large synthetic dataset dominated by linear interpolation between waypoints. 

We evaluate on four tasks of increasing difficulty (Fig. 5). Pick-and-Place Cube (PnP Cube) requires a single rigid translation. Reshelving requires pulling a book out along a specific direction and placing it on an upper shelf. Mug Insertion requires inserting a mug handle onto a rack, which demands following the demonstration trajectory rather than taking a shortcut to the goal pose. Hanging Bag tests extension to a slightly deformable object. This is feasible for all three methods because they rely on points rather than object pose. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-11/784c5ca8-2b0a-4805-8bfc-f3dcb4b8e5df/53e496894d6344adf9cfdda7ca44068fe54a87dcb694372fabccb11553948cdc.jpg)



Fig. 7. End-to-end execution under object-centric closed-loop control. Object motion is rendered from blue to red. (a) Teapot pouring with perturbation recovery. (b) Mug hanging with a side-lying target mug and a different feasible grasp.


For each task, we sample initial poses from two regimes with full yaw range [−180<sup>◦</sup>, 180<sup>◦</sup>] and different xy extent. Near draws the xy position from a compact box around the demonstration start. Far extends this box, rejection-sampling every pose to lie outside the Near box. Demonstrations are collected in Near for 50-demo Point Policy. The success criteria depend on task type. For PnP Cube and Reshelving, a trial succeeds if the object center is within 1 cm of the demonstration goal. For Mug Insertion and Hanging Bag, a trial succeeds if the average distance between current and demonstration goal points is below 1 cm. 

Table II reports success rates. All three methods solve PnP Cube, which reduces to a rigid translation. Instant Policy fails on Reshelving and Mug Insertion despite solving PnP Cube, so the integration itself is functional. We attribute this to distribution shift, as its motion prior is trained on RLBench [40] scenes with near-linear trajectories, though we did not isolate which factor dominates. Point Policy follows the demonstration when poses stay near it, but with one demonstration it lacks a cue for the current timestep. Under out-of-distribution poses, it often drives toward the final pose rather than the nearest trajectory configuration, colliding with the wall or rod. With 50 demonstrations, success remains high near the training distribution but drops with distance (Fig. 6), whereas Point-set BCSDM maintains 100% success across all distances from a single demonstration. 

## C. End-to-End evaluation

We evaluate this full pipeline end to end under the main variations from the introduction: object displacement, grasp change, goal-geometry change, and cross-category transfer. In simulation, we use MuJoCo [41] with a 7-DoF Franka Panda arm. We also evaluate the hanging-rack task on hardware (Fig. 10). 


cube → box



mug → rod



plate → rack


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-11/784c5ca8-2b0a-4805-8bfc-f3dcb4b8e5df/1d0514c99aa461fcd462b9b75da92bf41c1a6774931f4e7ea93a908505c2680f.jpg)



Fig. 8. Trajectory adaptation under goal-geometry changes: a cube placed in a taller box, a mug hung on an L-hook rod, and a plate seated in a tilted rack slot. Top: demonstration. Bottom: execution along the trajectory adapted to the target fixture.


Object-centric closed-loop control. Fig. 7 (a) shows perturbation recovery in a teapot-pour task; in the final panel, the green overlay marks the state reached without perturbation. After a displacement, the gripper ends at a different pose while the teapot returns to the same final pour configuration. In Fig. 7 (b), the upright mug-hanging demonstration is transferred to a side-lying mug, where the demonstrated rim grasp is blocked by collision and the robot must choose a feasible grasp on the current object pose. Because control is defined on object points rather than the gripper pose, POIL still drives the handle to the rack from this different grasp. 

Goal transfer. When the target fixture differs from the demonstration, OAReg and TPS deform the demonstrated functional-part point trajectory to the target scene (Fig. 3). POIL then follows this adapted trajectory in closed loop, producing the executions in Fig. 8. We show three cases: a cube lifted higher than the demonstration to clear a taller box, a plate aligned to a tilted rack, and a mug rotated before insertion to align with an L-shaped rod. In the plate-rack task, the rack also changes position; because OAReg and TPS align the trajectory to the target fixture, the same mechanism handles this spatial shift. 

Pipeline failure analysis. We evaluate the side-lying mug setting in Fig. 7(b) over 50 randomized yaw angles. POIL completes 39 trials. Among the 11 failures, 7 result from part-detection failure, and 3 have no valid grasp after filtering. The remaining failure is a mug–floor collision, which is not captured by the current gripper-trajectory feasibility check. These results suggest that part detection under selfocclusion and finding a valid grasp, rather than the downstream point-based controller, are the main bottlenecks in the full pipeline. Representative executed failures are shown in Fig. 9. 

Real-world cross-category transfer. We deploy POIL on a 7-DoF Franka Emika Panda arm with two ZED 2i depth cameras. In this setup, the mug is too wide to grasp from the outside, and depth reconstructs its rim poorly. The flat tray and bags need a human to prop them up before any grasp. We therefore isolate grasping on hardware: the operator places the object in the gripper at an arbitrary grasp, and we evaluate the rest of the pipeline. Neither demonstration contains a gripper, as the simulated mug is actuated directly and the tray is moved by a human hand. Fig. 10 shows the executions. The top row transfers a simulated mughanging demonstration to a real mug, a brown handbag, and a green paper bag lying on the table. The brown handbag is an interesting case: its elongated handle makes the coarse alignment rotate the transferred points by about 90<sup>◦</sup>, yet the execution still hangs the bag, since the task tolerates the handle’s in-plane orientation. The bottom row transfers a human-hand demonstration of placing a grey tray in a dish rack to the same tray, a wooden tray, and a dark bag. Over 10 sim-to-real mug trials, POIL succeeds in 6, with 3 tracking failures and 1 infeasible IK solution, and the other transfers are shown qualitatively. We also add disturbances to this transfer (Fig. 11). Even after the mug is rotated within the grasp, the tracked points let the controller drive the handle back to the demonstrated goal. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-11/784c5ca8-2b0a-4805-8bfc-f3dcb4b8e5df/07579647c3c80dc10b8a480f479b3e6e08269354ca62354857e3a618756b9d49.jpg)



Fig. 9. Representative failures in the side-lying mug evaluation. Left: the mug is detected in all views, but the hidden handle causes Rex-Omni [25] to ground the rack as the handle (red box). Right: a feasible initial grasp leads to mug-floor collision, outside the current feasibility check.


![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-11/784c5ca8-2b0a-4805-8bfc-f3dcb4b8e5df/ff855c3f0b2f7f23751d9f630071bcd1417aebcc64a819907c668dae48fe244d.jpg)



Fig. 10. Real-world cross-category transfer from simulation (top) and human hand demonstration (bottom). Robot and object motion is rendered from blue (initial) to red (final), with tracked point trajectories overlaid.


## V. CONCLUSION AND LIMITATIONS

We introduced POIL, which employs functional-part points as a unified representation for one-shot trajectory transfer and stable closed-loop execution. This closes a gap left by prior trajectory transfer methods that execute open loop and by pose-based dynamical systems that assume a known object-pose or fixed grasp. In simulation and on a real robot, POIL transfers a single demonstration across object category, grasp pose, and goal geometry, while recovering from external disturbances. 

Several assumptions remain. First, we assume object and functional-part queries for detection, a few user-clicked correspondences when the fixture changes, and geometrically similar source and target parts for point correspondence. Second, execution requires the functional-part points to remain trackable; large motions outside the camera views, severe occlusion, or unreliable depth on transparent or specular surfaces can break tracking. Third, our experiments employ a parallel gripper and a single grasp, which can result in no collision-free, IK-feasible grasp under large pose changes, or no graspable configuration for flat objects. Future work could divide long transfers into sequential stages using regrasping, or use dexterous hands to reorient the object in hand. 

![image](https://cdn-mineru.openxlab.org.cn/result/2026-10-11/784c5ca8-2b0a-4805-8bfc-f3dcb4b8e5df/39b1ad4d6a2a3d7d37e37edf0847334d6aadf965ca2a3d327f6663c9f5d1398d.jpg)



Fig. 11. Disturbance recovery during sim-to-real mug hanging. A human pulls the gripper and rotates the mug within the grasp (red; intermediate frames below), yet the handle reaches the demonstrated goal (bottom right).


## REFERENCES



[1] B. Zitkovich et al., “Rt-2: Vision-language-action models transfer web knowledge to robotic control,” in Conference on Robot Learning. PMLR, 2023, pp. 2165–2183. 





[2] M. J. Kim et al., “Openvla: An open-source vision-language-action model,” arXiv preprint arXiv:2406.09246, 2024. 





[3] A. O’Neill et al., “Open x-embodiment: Robotic learning datasets and rt-x models: Open x-embodiment collaboration 0,” in IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 6892–6903. 





[4] C. Finn, T. Yu, T. Zhang, P. Abbeel, and S. Levine, “One-shot visual imitation learning via meta-learning,” in Conference on robot learning. PMLR, 2017, pp. 357–368. 





[5] Y. Duan et al., “One-shot imitation learning,” Advances in neural information processing systems, vol. 30, 2017. 





[6] V. Vosylius and E. Johns, “Instant policy: In-context imitation learning via graph diffusion,” in Proceedings of the International Conference on Learning Representations (ICLR), 2025. 





[7] E. Valassakis, G. Papagiannis, N. Di Palo, and E. Johns, “Demonstrate once, imitate immediately (dome): Learning visual servoing for oneshot imitation learning,” in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2022, pp. 8614–8621. 





[8] N. Di Palo and E. Johns, “Dinobot: Robot manipulation via retrieval and alignment with vision foundation models,” in IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 2798– 2805. 





[9] C. Tang et al., “Mimicfunc: Imitating tool manipulation from a single human video via functional correspondence,” arXiv preprint arXiv:2508.13534, 2025. 





[10] Y. Liu, J. Mao, J. B. Tenenbaum, T. Lozano-Perez, and L. P. Kaelbling,´ “One-shot manipulation strategy learning by making contact analogies,” in IEEE International Conference on Robotics and Automation (ICRA), 2025, pp. 15 387–15 393. 





[11] S. M. Khansari-Zadeh and A. Billard, “Learning stable nonlinear dynamical systems with gaussian mixture models,” IEEE Transactions on Robotics, vol. 27, no. 5, pp. 943–957, 2011. 





[12] B. Lee, Y. Lee, J. Ha, and F. C. Park, “Behavior-controllable stable dynamics models on riemannian configuration manifolds,” IEEE Transactions on Robotics, 2025. 





[13] T. Li, S. Sun, S. S. Aditya, and N. Figueroa, “Elastic motion policy: An adaptive dynamical system for robust and efficient one-shot imitation learning,” in IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025, pp. 9846–9853. 





[14] J. Zhang et al., “Telling left from right: Identifying geometry-aware semantic correspondence,” in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024, pp. 3076–3085. 





[15] F. Rajicˇ et al., “Multi-view 3d point tracking,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 59–68. 





[16] L. Manuelli, W. Gao, P. Florence, and R. Tedrake, “kpam: Keypoint affordances for category-level robotic manipulation,” in The International Symposium of Robotics Research. Springer, 2019, pp. 132–157. 





[17] W. Gao and R. Tedrake, “kpam 2.0: Feedback control for categorylevel robotic manipulation,” IEEE Robotics and Automation Letters, vol. 6, no. 2, pp. 2962–2969, 2021. 





[18] J. Gao, Z. Tao, N. Jaquier, and T. Asfour, “K-vil: Keypoints-based visual imitation learning,” IEEE Transactions on Robotics, vol. 39, no. 5, pp. 3888–3908, 2023. 





[19] W. Huang, C. Wang, Y. Li, R. Zhang, and L. Fei-Fei, “Rekep: Spatio-temporal reasoning of relational keypoint constraints for robotic manipulation,” arXiv preprint arXiv:2409.01652, 2024. 





[20] F. Liu, K. Fang, P. Abbeel, and S. Levine, “Moka: Open-world robotic manipulation through mark-based visual prompting,” arXiv preprint arXiv:2403.03174, 2024. 





[21] S. Haldar and L. Pinto, “Point policy: Unifying observations and actions with key points for robot manipulation,” arXiv preprint arXiv:2502.20391, 2025. 





[22] Z. Qin, K. Fang, Y. Zhu, L. Fei-Fei, and S. Savarese, “Keto: Learning keypoint representations for tool manipulation,” in IEEE International Conference on Robotics and Automation (ICRA), 2020, pp. 7278– 7285. 





[23] M. Vecerik et al., “Robotap: Tracking arbitrary points for few-shot visual imitation,” in IEEE International Conference on Robotics and Automation (ICRA), 2024, pp. 5397–5403. 





[24] M. Saveriano, F. J. Abu-Dakka, A. Kramberger, and L. Peternel, “Dynamic movement primitives in robotics: A tutorial survey,” The International Journal of Robotics Research, vol. 42, no. 13, pp. 1133– 1184, 2023. 





[25] Q. Jiang et al., “Detect anything via next point prediction,” arXiv preprint arXiv:2510.12798, 2025. 





[26] N. Ravi et al., “Sam 2: Segment anything in images and videos,” arXiv preprint arXiv:2408.00714, 2024. 





[27] J. Kim, G. Bae, E. S. Lee, and Y. M. Kim, “Learning 3d scene analogies with neural contextual scene maps,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 7828–7840. 





[28] M. Zhao, G. Meng, and D.-m. Yan, “Occlusion-aware non-rigid point cloud registration via unsupervised neural deformation correntropy,” in The Thirteenth International Conference on Learning Representations, 2025. 





[29] N. Karaev, I. Rocco, B. Graham, N. Neverova, A. Vedaldi, and C. Rupprecht, “Cotracker: It is better to track together,” in European conference on computer vision. Springer, 2024, pp. 18–35. 





[30] N. Karaev, Y. Makarov, J. Wang, N. Neverova, A. Vedaldi, and C. Rupprecht, “Cotracker3: Simpler and better point tracking by pseudolabelling real videos,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2025, pp. 6013–6022. 





[31] W. Kabsch, “A solution for the best rotation to relate two sets of vectors,” Foundations of Crystallography, vol. 32, no. 5, pp. 922–923, 1976. 





[32] F. L. Bookstein, “Principal warps: Thin-plate splines and the decomposition of deformations,” IEEE Transactions on pattern analysis and machine intelligence, vol. 11, no. 6, pp. 567–585, 1989. 





[33] M. A. Fischler and R. C. Bolles, “Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography,” Communications of the ACM, vol. 24, no. 6, pp. 381–395, 1981. 





[34] D. E. Koditschek, “The application of total energy as a lyapunov function for mechanical control systems,” Contemporary mathematics, vol. 97, no. 672, pp. 131–157, 1989. 





[35] D. Chillingworth, J. Marsden, and Y. Wan, “Symmetry and bifurcation in three-dimensional elasticity. part ii,” Archive for rational mechanics and analysis, vol. 83, no. 4, pp. 363–395, 1983. 





[36] N. A. Chaturvedi, A. K. Sanyal, and N. H. McClamroch, “Rigid-body attitude control,” IEEE control systems magazine, vol. 31, no. 3, pp. 30–51, 2011. 





[37] M. Oquab et al., “Dinov2: Learning robust visual features without supervision,” arXiv preprint arXiv:2304.07193, 2023. 





[38] L. Tang, M. Jia, Q. Wang, C. P. Phoo, and B. Hariharan, “Emergent correspondence from image diffusion,” Advances in neural information processing systems, vol. 36, pp. 1363–1389, 2023. 





[39] Y. Wang et al., “D<sup>3</sup>fields: Dynamic 3d descriptor fields for zeroshot generalizable rearrangement,” in 8th Annual Conference on Robot Learning, 2024. 





[40] S. James, Z. Ma, D. R. Arrojo, and A. J. Davison, “Rlbench: The robot learning benchmark & learning environment,” IEEE Robotics and Automation Letters, vol. 5, no. 2, pp. 3019–3026, 2020. 





[41] E. Todorov, T. Erez, and Y. Tassa, “Mujoco: A physics engine for model-based control,” in IEEE/RSJ International Conference on Intelligent Robots and Systems, 2012, pp. 5026–5033. 

