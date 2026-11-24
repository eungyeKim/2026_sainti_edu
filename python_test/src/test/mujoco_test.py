import mujoco

xml = """
<mujoco>
    <option timestep="0.002"/>
    <worldbody>
        <geom type="plane" size="5 5 0.1"/>
        <body name="box" pos="0 0 0.5">
            <freejoint/>
            <geom type="box" size="0.1 0.1 0.1" mass="1"/>
        </body>
    </worldbody>
</mujoco>
"""

def main():
    print("ok")
    # mujoco
    print(mujoco.__version__)
    # 변수

model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
    
    ## 자유도 free-shape -> nq  7개 (x,y,z, qx,qy,qz,qw) 회전은 쿼터니안 
    ## nv 6개 (x,y,z, wx,wy,wz) 회전은 각속도 , nu(조작) - 액츄에이터 , 모터... 로봇에서 모터 갯수만큼 nu 가 늘어남.
    

print("MuJoCo:", mujoco.__version__)
print("자유도:", model.nv)
print("위치 변수 개수:", model.nq)

print(f"초기 높이: {data.qpos[2]:.6f}")

for _ in range(100):
    mujoco.mj_step(model, data)

print(f"시뮬레이션 시간: {data.time:.3f}")
print(f"이동 후 높이: {data.qpos[2]:.6f}")

if __name__ == "__main__":
    main()