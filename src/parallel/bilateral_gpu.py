import pyopencl as cl
import numpy as np


class BilateralGPU:

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

        __kernel void bilateral_filter(

            __global const uchar *input_image,
            __global uchar *output_image,

            const int width,
            const int height,

            const float sigma_space,
            const float sigma_intensity

        )
        {

            int x = get_global_id(0);
            int y = get_global_id(1);

            if (x >= width || y >= height)
                return;

            int idx = y * width + x;

            float center_pixel = input_image[idx];

            float sum = 0.0f;
            float weight_sum = 0.0f;

            int radius = 2;

            // =====================================
            // VECINDARIO
            // =====================================

            for (int dy = -radius; dy <= radius; dy++)
            {
                for (int dx = -radius; dx <= radius; dx++)
                {

                    int nx = x + dx;
                    int ny = y + dy;

                    if (nx >= 0 && nx < width &&
                        ny >= 0 && ny < height)
                    {

                        int nidx = ny * width + nx;

                        float neighbor =
                            input_image[nidx];

                        // =========================
                        // DISTANCIA ESPACIAL
                        // =========================

                        float spatial_dist =
                            (float)(dx*dx + dy*dy);

                        float spatial_weight =
                            exp(
                                -spatial_dist /
                                (2.0f * sigma_space *
                                 sigma_space)
                            );

                        // =========================
                        // DISTANCIA INTENSIDAD
                        // =========================

                        float intensity_dist =
                            neighbor - center_pixel;

                        float intensity_weight =
                            exp(
                                -(intensity_dist *
                                  intensity_dist) /
                                (2.0f *
                                 sigma_intensity *
                                 sigma_intensity)
                            );

                        // =========================
                        // PESO TOTAL
                        // =========================

                        float weight =
                            spatial_weight *
                            intensity_weight;

                        sum += neighbor * weight;

                        weight_sum += weight;
                    }
                }
            }

            output_image[idx] =
                (uchar)(sum / weight_sum);
        }

        """

        self.program = cl.Program(
            self.context,
            kernel_code
        ).build()

        self.kernel = self.program.bilateral_filter

    # =============================================
    # APLICACION
    # =============================================

    def apply(
        self,
        image,
        sigma_space=10.0,
        sigma_intensity=25.0
    ):

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
        # EJECUTAR KERNEL
        # =========================================

        self.kernel(
            self.queue,
            (width, height),
            None,

            input_buffer,
            output_buffer,

            np.int32(width),
            np.int32(height),

            np.float32(sigma_space),
            np.float32(sigma_intensity)
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