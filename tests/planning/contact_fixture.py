"""Explicit three-coordinate native contact model shared by bounded probes.

All dimensions, limits and initial state are test inputs. The thin press pad
lets its exact outward sphere cover fit inside the declared 3 mm contact
model allowance; it is not a task-specific production robot adapter.
"""

# BEGIN shared native fixtures; the historical wide pad remains the default.
_XML = """<mujoco>
  <compiler angle="radian"/>
  <option timestep=".002" gravity="0 0 0"/>
  <default><joint damping="1"/><geom friction=".8 .01 .001" solref=".004 1"/></default>
  <worldbody>
    <geom name="target_slab" type="box" pos="0 0 -.05" size=".3 .3 .05"/>
    <body name="base">
      <body name="arm" pos="0 0 .07">
        <joint name="arm_q" type="slide" axis="0 0 1" range="-.04 .04"/>
        <geom name="pad" type="box" size=".05 .05 .05" mass=".2"/>
        <site name="tcp" pos="0 0 -.05"/>
        <body name="left" pos="0 .15 .1">
          <joint name="left_q" type="slide" axis="0 1 0" range="0 .04"/>
          <geom name="left_pad" type="box" size=".01 .01 .01" mass=".02"/>
        </body>
        <body name="right" pos="0 -.15 .1">
          <joint name="right_q" type="slide" axis="0 1 0" range="-.04 0"/>
          <geom name="right_pad" type="box" size=".01 .01 .01" mass=".02"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="arm_motor" joint="arm_q" ctrllimited="true" ctrlrange="-10 10"/>
    <position name="left_servo" joint="left_q" kp="100" ctrllimited="true" ctrlrange="0 .04"/>
    <position name="right_servo" joint="right_q" kp="100" ctrllimited="true" ctrlrange="-.04 0"/>
  </actuator>
</mujoco>"""

_URDF = """<robot name="cpu_press_fixture">
  <link name="base"/><link name="arm"/><link name="left"/><link name="right"/><link name="tcp"/>
  <joint name="arm_q" type="prismatic"><parent link="base"/><child link="arm"/>
    <origin xyz="0 0 .07"/><axis xyz="0 0 1"/>
    <limit lower="-.04" upper=".04" effort="10" velocity="1"/></joint>
  <joint name="left_q" type="prismatic"><parent link="arm"/><child link="left"/>
    <origin xyz="0 .15 .1"/><axis xyz="0 1 0"/>
    <limit lower="0" upper=".04" effort="10" velocity="1"/></joint>
  <joint name="right_q" type="prismatic"><parent link="arm"/><child link="right"/>
    <origin xyz="0 -.15 .1"/><axis xyz="0 1 0"/>
    <limit lower="-.04" upper="0" effort="10" velocity="1"/></joint>
  <joint name="tcp_fixed" type="fixed"><parent link="arm"/><child link="tcp"/>
    <origin xyz="0 0 -.05"/></joint>
</robot>"""



def native_model_text(*, pad_width=.1):
    """Return matching native models for an explicitly chosen square pad."""
    if not 0 < pad_width <= .1:
        raise ValueError("The bounded test fixture requires 0 < pad_width <= .1 m")
    half = pad_width/2
    xml = _XML.replace('name="pad" type="box" size=".05 .05 .05"',
                       f'name="pad" type="box" size="{half} {half} .05"')
    return xml, _URDF
# END shared native fixture dimensions do not modify a production model.
