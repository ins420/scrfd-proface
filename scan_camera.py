import pyrealsense2 as rs

ctx = rs.context()

for dev in ctx.query_devices():
    print("장치:", dev.get_info(rs.camera_info.name))
    print("시리얼:", dev.get_info(rs.camera_info.serial_number))