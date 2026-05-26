import pyopencl as cl

# ============================================
# MOSTRAR PLATAFORMAS OPENCL
# ============================================

platforms = cl.get_platforms()

print("\n=== OPENCL PLATFORMS ===\n")

for i, platform in enumerate(platforms):

    print(f"Platform {i}:")
    print(platform.name)

    devices = platform.get_devices()

    for j, device in enumerate(devices):

        print(f"  Device {j}:")
        print(f"  Name: {device.name}")
        print(f"  Type: {cl.device_type.to_string(device.type)}")
        print(f"  Compute Units: {device.max_compute_units}")
        print(f"  Global Memory: {device.global_mem_size / 1e9:.2f} GB")

        print()