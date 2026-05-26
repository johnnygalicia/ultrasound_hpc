import pyopencl as cl
import numpy as np


class GaussianGPU:

    def __init__(self):

        # =========================================
        # CONTEXTO OPENCL
        # =========================================

        platform = cl.get_platforms()[0]

        device = platform.get_devices()[0]

        self.context = cl.Context([device])

        self.queue = cl.CommandQueue(self.context)

        # =========================================
        # KERNEL OPENCL
        # =========================================

        kernel_code = """
        __kernel void gaussian_blur(
            __global const uchar *input,
            __global uchar *output,
            const int width,
            const int height
        )
        {
            int x = get_global_id(0);
            int y = get_global_id(1);

            if (x >= width || y >= height)
                return;

            int idx = y * width + x;

            float sum = 0.0f;
            int count = 0;

            for (int ky = -1; ky <= 1; ky++)
            {
                for (int kx = -1; kx <= 1; kx++)
                {
                    int nx = x + kx;
                    int ny = y + ky;

                    if (
                        nx >= 0 &&
                        nx < width &&
                        ny >= 0 &&
                        ny < height
                    )
                    {
                        int nidx = ny * width + nx;

                        sum += input[nidx];
                        count++;
                    }
                }
            }

            output[idx] = (uchar)(sum / count);
        }
        """

        # =========================================
        # BUILD SOLO UNA VEZ
        # =========================================

        self.program = cl.Program(
            self.context,
            kernel_code
        ).build()

        # =========================================
        # RECUPERAR KERNEL SOLO UNA VEZ
        # =========================================

        self.kernel = cl.Kernel(
            self.program,
            "gaussian_blur"
        )

    # =============================================
    # PROCESAMIENTO GPU
    # =============================================

    def apply(self, image):

        image = image.astype(np.uint8)

        height, width = image.shape

        output = np.empty_like(image)

        mf = cl.mem_flags

        input_buffer = cl.Buffer(
            self.context,
            mf.READ_ONLY | mf.COPY_HOST_PTR,
            hostbuf=image
        )

        output_buffer = cl.Buffer(
            self.context,
            mf.WRITE_ONLY,
            output.nbytes
        )

        # =========================================
        # ARGUMENTOS KERNEL
        # =========================================

        self.kernel.set_arg(0, input_buffer)
        self.kernel.set_arg(1, output_buffer)
        self.kernel.set_arg(2, np.int32(width))
        self.kernel.set_arg(3, np.int32(height))

        # =========================================
        # EJECUCIÓN
        # =========================================

        cl.enqueue_nd_range_kernel(
            self.queue,
            self.kernel,
            (width, height),
            None
        )

        cl.enqueue_copy(
            self.queue,
            output,
            output_buffer
        )

        return output