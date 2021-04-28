import pymlir

def declare_toolchain_version():
    print("\nCVITEK TPU Toolchain {}\n".format(pymlir.module().version))