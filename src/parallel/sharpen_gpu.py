import pyopencl as cl
import numpy as np


class SharpenGPU:

    def __init__(self):

        # =========================================
        # CONTEXTO OPENCL
        # =========================================

        platform = cl.get_platforms()[0]

        device = platform.get_devices()[0]

        self.context = cl.Context([device])

        self.queue = cl.CommandQueue(
            self.context
        )

        # =========================================
        # KERNEL OPENCL
        # =========================================

        kernel_code = """

        __kernel void sharpen_filter(

            __global const uchar *input_image,
            __global uchar *output_image,

            const int width,
            const int height

        )
        {

            int x = get_global_id(0);
            int y = get_global_id(1);

            if (x >= width || y >= height)
                return;

            int idx = y * width + x;

            // =====================================
            // BORDES
            // =====================================

            if (
                x == 0 ||
                y == 0 ||
                x == width - 1 ||
                y == height - 1
            )
            {
                output_image[idx] =
                    input_image[idx];

                return;
            }

            // =====================================
            // KERNEL SHARPEN
            // =====================================

            int center =
                input_image[idx];

            int top =
                input_image[
                    (y - 1) * width + x
                ];

            int bottom =
                input_image[
                    (y + 1) * width + x
                ];

            int left =
                input_image[
                    y * width + (x - 1)
                ];

            int right =
                input_image[
                    y * width + (x + 1)
                ];

            int result =
                (5 * center)
                - top
                - bottom
                - left
                - right;

            // =====================================
            // CLAMP
            // =====================================

            if (result < 0)
                result = 0;

            if (result > 255)
                result = 255;

            output_image[idx] =
                (uchar)(result);
        }

        """

        # =========================================
        # BUILD
        # =========================================

        self.program = cl.Program(
            self.context,
            kernel_code
        ).build()

        self.kernel = self.program.sharpen_filter

    # =============================================
    # APPLY
    # =============================================

    def apply(self, image):

        image = image.astype(np.uint8)

        height, width = image.shape

        output = np.empty_like(image)

        mf = cl.mem_flags

        input_buffer = cl.Buffer(
            self.context,
            mf.READ_ONLY |
            mf.COPY_HOST_PTR,
            hostbuf=image
        )

        output_buffer = cl.Buffer(
            self.context,
            mf.WRITE_ONLY,
            output.nbytes
        )

        # =========================================
        # EJECUTAR KERNEL
        # =========================================

        self.kernel(
            self.queue,
            (width, height),
            None,

            input_buffer,
            output_buffer,

            np.int32(width),
            np.int32(height)
        )

        # =========================================
        # COPIAR RESULTADO
        # =========================================

        cl.enqueue_copy(
            self.queue,
            output,
            output_buffer
        )

        return output