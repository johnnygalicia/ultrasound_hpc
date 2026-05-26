import pyopencl as cl
import numpy as np


# ==================================================
# CONTEXTO OPENCL
# ==================================================

platform = cl.get_platforms()[0]

device = platform.get_devices()[0]

context = cl.Context([device])

queue = cl.CommandQueue(context)


# ==================================================
# KERNEL OPENCL
# ==================================================

kernel_code = """

__kernel void invert_image(
    __global uchar *input_image,
    __global uchar *output_image
)
{
    int idx = get_global_id(0);

    output_image[idx] = 255 - input_image[idx];
}

"""


program = cl.Program(
    context,
    kernel_code
).build()


# ==================================================
# FUNCION GPU
# ==================================================

def invert_image_gpu(image):

    # -----------------------------------------
    # Convertir a uint8
    # -----------------------------------------

    image = image.astype(np.uint8)

    flat_image = image.flatten()

    # -----------------------------------------
    # Buffers GPU
    # -----------------------------------------

    mf = cl.mem_flags

    input_buffer = cl.Buffer(
        context,
        mf.READ_ONLY | mf.COPY_HOST_PTR,
        hostbuf=flat_image
    )

    output_buffer = cl.Buffer(
        context,
        mf.WRITE_ONLY,
        flat_image.nbytes
    )

    # -----------------------------------------
    # Ejecutar kernel
    # -----------------------------------------

    global_size = (flat_image.size,)

    program.invert_image(
        queue,
        global_size,
        None,
        input_buffer,
        output_buffer
    )

    # -----------------------------------------
    # Recuperar resultado
    # -----------------------------------------

    output_image = np.empty_like(flat_image)

    cl.enqueue_copy(
        queue,
        output_image,
        output_buffer
    )

    output_image = output_image.reshape(
        image.shape
    )

    return output_image